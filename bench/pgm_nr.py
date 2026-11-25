"""External Newton-Raphson power flow on Power Grid Model (PGM) input data.

This reproduces PGM's symmetric Newton-Raphson formulation closely enough to
(a) reproduce PGM's converged voltages and (b) reproduce PGM's iteration counts
for the start options PGM already has (flat, linear, average_source). It then adds
the starts PGM does not accept today: a user supplied voltage vector.

Model, following the PGM C++ sources (power_grid_model_c/power_grid_model/include/power_grid_model):
- per unit on 1 MVA (common/common.hpp base_power_3p), node u_rated as voltage base
- line, generic_branch, transformer pi models as in component/line.hpp, generic_branch.hpp,
  transformer.hpp and edge.hpp calc_param_y_sym (single connected edges included)
- link: ideal connection, nodes merged (no node injection sensors in these cases)
- source: Norton equivalent, y_ref added to the bus diagonal plus current y_ref*u_ref (source.hpp)
- shunt: constant admittance; sym_load / sym_gen: ZIP (const power, impedance, current)
- voltage_regulator on a sym_gen/sym_load: PV bus at u_ref, no q limits
- polar NR, magnitude step relative (x.v += x.v*dv), stop when max |u_new-u_old| <= tol,
  iteration count includes the final iteration (iterative_pf_solver.hpp)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

BASE_S = 1e6
DEG30 = np.pi / 6
NA_INT8 = -128


def _nan(x, default):
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), default, x)


class Network:
    def __init__(self, data: dict, frequency: float = 50.0):
        self.data = data
        nodes = data["node"]
        self.node_ids = nodes["id"]
        id2raw = {int(i): k for k, i in enumerate(self.node_ids)}
        n_raw = len(nodes)
        # merge nodes connected by links (infinite admittance)
        parent = list(range(n_raw))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        if "link" in data:
            for lk in data["link"]:
                if lk["from_status"] == 1 and lk["to_status"] == 1:
                    a, b = find(id2raw[int(lk["from_node"])]), find(id2raw[int(lk["to_node"])])
                    if a != b:
                        parent[b] = a
        roots = sorted({find(k) for k in range(n_raw)})
        root2bus = {r: b for b, r in enumerate(roots)}
        self.raw2bus = np.array([root2bus[find(k)] for k in range(n_raw)])
        self.id2bus = {i: int(self.raw2bus[k]) for i, k in id2raw.items()}
        self.n = len(roots)
        u_rated = np.zeros(self.n)
        for k in range(n_raw):
            u_rated[self.raw2bus[k]] = nodes["u_rated"][k]
        self.u_rated = u_rated

        rows, cols, vals = [], [], []
        # list of (bus_from, bus_to, tap_ratio) for phase shift propagation
        self.edges = []

        def add_branch(f, t, fs, ts, y_series, y_shunt, tap_ratio):
            bf, bt = self.id2bus[int(f)], self.id2bus[int(t)]
            tap = abs(tap_ratio)
            if fs == 1 and ts == 1:
                ytt = y_series + 0.5 * y_shunt
                yff = ytt / tap / tap
                yft = -y_series / np.conj(tap_ratio)
                ytf = -y_series / tap_ratio
                rows.extend([bf, bt, bf, bt])
                cols.extend([bf, bt, bt, bf])
                vals.extend([yff, ytt, yft, ytf])
                self.edges.append((bf, bt, tap_ratio))
            elif fs == 1 or ts == 1:
                if abs(y_shunt) < 1e-10:
                    es = 0.0
                else:
                    es = 0.5 * y_shunt + 1.0 / (1.0 / y_series + 2.0 / y_shunt)
                if fs == 1:
                    rows.append(bf); cols.append(bf); vals.append(es / tap / tap)
                else:
                    rows.append(bt); cols.append(bt); vals.append(es)

        if "line" in data:
            for ln in data["line"]:
                u1 = u_rated[self.id2bus[int(ln["from_node"])]]
                base_y = BASE_S / u1 ** 2
                ys = 1.0 / (ln["r1"] + 1j * ln["x1"]) / base_y
                ysh = 2 * np.pi * frequency * ln["c1"] / base_y * (ln["tan1"] + 1j)
                add_branch(ln["from_node"], ln["to_node"], ln["from_status"], ln["to_status"], ys, ysh, 1.0)
        if "generic_branch" in data:
            for gb in data["generic_branch"]:
                u2 = u_rated[self.id2bus[int(gb["to_node"])]]
                base_y = BASE_S / u2 ** 2
                ys = 1.0 / (gb["r1"] + 1j * gb["x1"]) / base_y
                ysh = (gb["g1"] + 1j * gb["b1"]) / base_y
                k = 1.0 if np.isnan(gb["k"]) else gb["k"]
                th = 0.0 if np.isnan(gb["theta"]) else np.fmod(gb["theta"], 2 * np.pi)
                add_branch(gb["from_node"], gb["to_node"], gb["from_status"], gb["to_status"], ys, ysh,
                           k * np.exp(1j * th))
        if "transformer" in data:
            for tr in data["transformer"]:
                u1r = u_rated[self.id2bus[int(tr["from_node"])]]
                u2r = u_rated[self.id2bus[int(tr["to_node"])]]
                tap_nom = 0 if tr["tap_nom"] == NA_INT8 else int(tr["tap_nom"])
                tap_pos = tap_nom if tr["tap_pos"] == NA_INT8 else int(tr["tap_pos"])
                tap_dir = 1 if tr["tap_max"] > tr["tap_min"] else -1
                u1, u2 = float(tr["u1"]), float(tr["u2"])
                if tr["tap_side"] == 0:
                    u1 += tap_dir * (tap_pos - tap_nom) * tr["tap_size"]
                else:
                    u2 += tap_dir * (tap_pos - tap_nom) * tr["tap_size"]
                k = (u1 / u2) / (u1r / u2r)
                sn = float(tr["sn"])

                def adj(xk, xmin, xmax):
                    # component/transformer_utils.hpp tap_adjust_impedance
                    xmin = xk if np.isnan(xmin) else float(xmin)
                    xmax = xk if np.isnan(xmax) else float(xmax)
                    tmin, tmax = int(tr["tap_min"]), int(tr["tap_max"])
                    if min(tap_nom, tmax) <= tap_pos <= max(tap_nom, tmax):
                        return xk if tmax == tap_nom else xk + (tap_pos - tap_nom) * (xmax - xk) / (tmax - tap_nom)
                    return xk if tmin == tap_nom else xk + (tap_pos - tap_nom) * (xmin - xk) / (tmin - tap_nom)

                uk = adj(float(tr["uk"]), tr["uk_min"], tr["uk_max"])
                pk = adj(float(tr["pk"]), tr["pk_min"], tr["pk_max"])
                base_y_to = BASE_S / u2r ** 2
                z_abs = abs(uk) * u2 * u2 / sn
                zr = pk * u2 * u2 / sn / sn
                zi2 = z_abs ** 2 - zr ** 2
                zs = zr + 1j * (np.sign(uk) if uk != 0 else 1.0) * (np.sqrt(zi2) if zi2 > 0 else 0.0)
                ys = (1.0 / zs) / base_y_to
                y_abs = tr["i0"] * sn / u2 / u2
                yr = tr["p0"] / u2 / u2
                yi2 = y_abs ** 2 - yr ** 2
                ysh = (yr - 1j * (np.sqrt(yi2) if yi2 > 0 else 0.0)) / base_y_to
                add_branch(tr["from_node"], tr["to_node"], tr["from_status"], tr["to_status"], ys, ysh,
                           k * np.exp(1j * (int(tr["clock"]) * DEG30)))
        for comp in data:
            if comp not in ("node", "line", "generic_branch", "transformer", "link", "shunt", "source",
                            "sym_load", "sym_gen", "voltage_regulator"):
                raise NotImplementedError(f"component {comp}")

        # source: Norton equivalent
        src = data["source"]
        self.src_bus = np.array([self.id2bus[int(s)] for s in src["node"]])
        self.src_on = src["status"] == 1
        sk = _nan(src["sk"], 1e10)
        sk = np.where(np.isinf(sk), 1e50, sk)
        rx = _nan(src["rx_ratio"], 0.1)
        z_abs = BASE_S / sk
        x1 = z_abs / np.sqrt(rx * rx + 1.0)
        self.src_y = 1.0 / (x1 * rx + 1j * x1)
        for b, y, on in zip(self.src_bus, self.src_y, self.src_on):
            if on:
                rows.append(b); cols.append(b); vals.append(y)
        self.y_branch = sp.csr_matrix((np.array(vals, dtype=complex), (rows, cols)), shape=(self.n, self.n))
        self.y_branch.sum_duplicates()

        # appliances
        self.loadgen = []
        for comp, sign in (("sym_load", -1.0), ("sym_gen", 1.0)):
            if comp in data:
                a = data[comp]
                self.loadgen.append((comp, sign, np.array([self.id2bus[int(x)] for x in a["node"]]), a["type"].astype(int)))
        self.pv_bus = np.zeros(0, dtype=int)
        self.pv_uref = np.zeros(0)
        if "voltage_regulator" in data:
            vr = data["voltage_regulator"]
            obj2bus = {}
            for comp, _, buses, _ in self.loadgen:
                for i, b in zip(data[comp]["id"], buses):
                    obj2bus[int(i)] = b
            on = vr["status"] == 1
            self.pv_bus = np.array([obj2bus[int(o)] for o in vr["regulated_object"][on]], dtype=int)
            self.pv_uref = vr["u_ref"][on].astype(float)
        self.regulated = {}
        if "voltage_regulator" in data:
            for o, st in zip(data["voltage_regulator"]["regulated_object"], data["voltage_regulator"]["status"]):
                if st == 1:
                    self.regulated[int(o)] = True
        self.phase_shift = self._phase_shift()

    def _phase_shift(self):
        adj = [[] for _ in range(self.n)]
        for bf, bt, n in self.edges:
            adj[bf].append((bt, -np.angle(n)))
            adj[bt].append((bf, np.angle(n)))
        ps = np.full(self.n, np.nan)
        for s, on in zip(self.src_bus, self.src_on):
            if on and np.isnan(ps[s]):
                ps[s] = 0.0
                stack = [s]
                while stack:
                    a = stack.pop()
                    for b, d in adj[a]:
                        if np.isnan(ps[b]):
                            ps[b] = ps[a] + d
                            stack.append(b)
        return np.nan_to_num(ps)

    # ---- per-scenario state -------------------------------------------------
    def scenario(self, data: dict):
        """Injection parameters for one scenario (data = possibly updated input dict)."""
        s_spec = {0: np.zeros(self.n, complex), 1: np.zeros(self.n, complex), 2: np.zeros(self.n, complex)}
        s_lin = np.zeros(self.n, complex)
        for comp, sign, buses, types in self.loadgen:
            a = data[comp]
            on = a["status"] == 1
            s = sign * (a["p_specified"] + 1j * a["q_specified"]) / BASE_S
            reg = np.array([int(i) in self.regulated for i in a["id"]], dtype=bool)
            np.add.at(s_lin, buses[on], np.where(reg, s.real + 0j, s)[on])
            for t, key in ((0, 0), (1, 2), (2, 1)):  # LoadGenType: 0 const_power, 1 const_impedance, 2 const_current
                m = on & (types == t)
                np.add.at(s_spec[key], buses[m], s[m])
        y_shunt = np.zeros(self.n, complex)
        if "shunt" in data:
            sh = data["shunt"]
            on = sh["status"] == 1
            b = np.array([self.id2bus[int(x)] for x in sh["node"]])
            base_y = BASE_S / self.u_rated[b] ** 2
            np.add.at(y_shunt, b[on], ((sh["g1"] + 1j * sh["b1"]) / base_y)[on])
        src = data["source"]
        u_ref = src["u_ref"] * np.exp(1j * _nan(src["u_ref_angle"], 0.0))
        i_src = np.zeros(self.n, complex)
        np.add.at(i_src, self.src_bus[self.src_on], (self.src_y * u_ref)[self.src_on])
        ybus = (self.y_branch + sp.diags(y_shunt)).tocsr()
        return dict(ybus=ybus, s0=s_spec[0], s1=s_spec[1], s2=s_spec[2], i_src=i_src, s_lin=s_lin,
                    u_ref=u_ref[self.src_on])

    # ---- starts (mirroring newton_raphson_pf_solver.hpp) --------------------
    def start_flat(self, sc):
        ang = np.angle(np.mean(sc["u_ref"] * np.exp(-1j * self.phase_shift[self.src_bus[self.src_on]])))
        u = np.exp(1j * (ang + self.phase_shift))
        for b in np.unique(self.src_bus[self.src_on]):
            m = self.src_bus[self.src_on] == b
            u[b] = np.mean(sc["u_ref"][m])
        return u

    def start_average_source(self, sc):
        avg = np.mean(sc["u_ref"] * np.exp(-1j * self.phase_shift[self.src_bus[self.src_on]]))
        return avg * np.exp(1j * self.phase_shift)

    def start_linear(self, sc):
        # every load_gen as constant admittance -conj(S) at 1 p.u.; regulated generator q left out
        y = sc["ybus"] + sp.diags(-np.conj(sc["s_lin"]))
        return spla.spsolve(y.tocsc(), sc["i_src"])

    # ---- Newton-Raphson ------------------------------------------------------
    def solve(self, sc, u0, tol=1e-8, max_iter=20):
        """Returns (u, n_iter, converged). n_iter counted as in PGM (includes the final small step)."""
        ybus = sc["ybus"]
        u = np.array(u0, dtype=complex)
        pv = np.zeros(self.n, bool)
        pv[self.pv_bus] = True
        if pv.any():
            u[self.pv_bus] = self.pv_uref * np.exp(1j * np.angle(u[self.pv_bus]))
        pq = np.where(~pv)[0]
        va = np.arange(self.n)
        n = self.n
        for it in range(1, max_iter + 1):
            vm = np.abs(u)
            ibus = ybus @ u
            s_calc = u * np.conj(ibus)
            s_inj = sc["s0"] + sc["s1"] * vm + sc["s2"] * vm ** 2 + u * np.conj(sc["i_src"])
            f = s_calc - s_inj
            # Jacobian, MATPOWER dSbus_dV convention, magnitude variable is relative (dV/V)
            du = sp.diags(u)
            dsva = 1j * du @ np.conj(sp.diags(ibus) - ybus @ du)
            dsvm_rel = du @ np.conj(ybus @ du) + np.conj(sp.diags(ibus)) @ du  # dS/dVm * Vm
            # injection derivatives
            dinj_va = 1j * u * np.conj(sc["i_src"])
            dinj_vm_rel = sc["s1"] * vm + 2 * sc["s2"] * vm ** 2 + u * np.conj(sc["i_src"])
            dsva = dsva - sp.diags(dinj_va)
            dsvm_rel = dsvm_rel - sp.diags(dinj_vm_rel)
            j11 = dsva.real[:, va]
            j12 = dsvm_rel.real[:, pq]
            j21 = dsva.imag[pq][:, va]
            j22 = dsvm_rel.imag[pq][:, pq]
            jac = sp.vstack([sp.hstack([j11, j12]), sp.hstack([j21, j22])]).tocsc()
            rhs = -np.concatenate([f.real, f.imag[pq]])
            dx = spla.spsolve(jac, rhs)
            if not np.all(np.isfinite(dx)):
                return u, it, False
            th = np.angle(u) + dx[:n]
            vm_new = vm.copy()
            vm_new[pq] += vm[pq] * dx[n:]
            u_new = vm_new * np.exp(1j * th)
            dev = np.max(np.abs(u_new - u))
            u = u_new
            if dev <= tol:
                return u, it, True
        return u, max_iter, False

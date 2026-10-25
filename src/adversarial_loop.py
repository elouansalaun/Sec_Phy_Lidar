"""
src/adversarial_loop.py: exact adversarial loop: optimiser ↔ intruder by path generation
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from src.instance import EPS, Instance, cost, evaluate, violations
from src.intruder import PROFILES, Domain, corridor_threshold, detection
from src.milp import Cuts, structural

TOL = 1e-6          # feasibility tolerance of the master on z_v ≤ Σ C_vj x_j


# =============================================================================
# Step 1: linking coefficients and intruder oracle
# =============================================================================
def coefficients(A: sparse.csr_matrix, theta: float, fusion: str = "product") -> sparse.csc_matrix:
    """Linking coefficients (configurations × nodes), stored by columns.

    fusion = "product": P_v = 1 − Π(1 − P_vj) (independence), C_vj = min(s_vj / s_θ, 1).
    fusion = "max"    : P_v = max_j P_vj (full correlation), C_vj = 1[P_vj ≥ θ].
    In both cases: P_v ≥ θ ⇔ min(Σ_{j∈T} C_vj, 1) = 1.
    """
    C = A.astype(np.float64).tocsr(copy=True)
    if fusion == "max":
        C.data = (C.data >= theta - 1e-12).astype(np.float64)
    elif fusion == "product":
        s_theta = -np.log1p(-theta)
        with np.errstate(divide="ignore"):
            C.data = np.minimum(-np.log1p(-C.data) / s_theta, 1.0)
    else:
        raise ValueError(fusion)
    C.eliminate_zeros()
    return C.tocsc()


def level(C: sparse.csc_matrix, T) -> np.ndarray:
    """u_v = min(Σ_{j∈T} C_vj, 1) = min(σ_v / s_θ, 1): u_v = 1 ⇔ P_v ≥ θ."""
    T = np.asarray(T, np.int64)
    if T.size == 0:
        return np.zeros(C.shape[1])
    return np.minimum(np.asarray(C.tocsr()[T].sum(axis=0)).ravel(), 1.0)


def intruder_oracle(dom: Domain, u: np.ndarray, useful: np.ndarray, profile: str,
                    n_paths: int) -> list[np.ndarray]:
    """Corridors where no useful node reaches θ (u < 1). Each one is a cut violated by x̂.

    After each corridor, its useful nodes are blocked (u = 1): the next one avoids the
    previous ones, which gives more varied cuts for the same solution x̂.
    """
    w = np.where(useful, u, 0.0)
    out = []
    for _ in range(n_paths):
        p, ch = corridor_threshold(dom, w, profile)
        if p >= 1.0 - TOL:
            break
        out.append(ch)
        w = w.copy()
        w[ch[useful[ch]]] = 1.0
    return out


def cost_step(inst: Instance) -> float:
    """Step q such that every cost is a multiple of q (0.1 or 0.01), otherwise 0."""
    c = np.r_[inst.cost_sensor, inst.cost_support]
    for q in (0.1, 0.01):
        if np.allclose(c / q, np.round(c / q), atol=1e-9):
            return q
    return 0.0


def useful_configurations(inst: Instance, C: sparse.csc_matrix, useful: np.ndarray,
                          dominance: bool) -> np.ndarray:
    """Configurations to keep for the minimal cost (same optimum).

    Empty: no coefficient on a useful node; it serves no cut and costs > 0.
    Dominated ("max" fusion only): same support and same type as a configuration j' with a
    sensor cost ≤, whose covered set contains its own. Replacing it by j' (or removing it
    if j' is already placed) keeps (C3), (C4) and feasibility, without costing more.
    Under "product" fusion, contributions add up: the argument does not hold.
    """
    Cu = C.tocsr()[:, np.flatnonzero(useful)].tocsr()
    Cu.data = (Cu.data > 0).astype(np.float64)
    size = np.diff(Cu.indptr)
    ok = size > 0
    if not dominance:
        return ok
    for s_ in np.unique(inst.support):
        J = np.flatnonzero((inst.support == s_) & ok)
        if J.size < 2:
            continue
        S = Cu[J]
        inter = (S @ S.T).toarray()                        # |S_a ∩ S_b|
        t = size[J]
        for a_, ja in enumerate(J):
            sub = inter[a_] == t[a_]                       # S_a ⊆ S_b
            sub &= inst.kind[J] == inst.kind[ja]
            sub &= inst.cost_sensor[J] <= inst.cost_sensor[ja] + 1e-12
            equal = (t == t[a_]) & sub
            strict = sub & ~equal
            if strict.any() or (equal & (J < ja)).any():   # on a tie, keep the smallest index
                ok[ja] = False
    return ok


# =============================================================================
# Step 2: master and loop
# =============================================================================
@dataclass
class LoopResult:
    status: str                   # "optimal", "infeasible", "limit"
    T: np.ndarray
    value: float                  # cost (objective cost) or unnormalised F (objective F)
    bound: float                  # lower bound of the cost, or upper bound of F
    iterations: int
    n_cuts: int
    time_s: float
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


class Master:
    """Builds the master MILP from the current cuts."""

    def __init__(self, inst: Instance, C: sparse.csc_matrix, budget: float,
                 cuts_F: Cuts | None, allowed: np.ndarray | None = None):
        self.inst, self.C, self.cuts_F = inst, C, cuts_F
        self.step = cost_step(inst)
        self.allowed = np.ones(inst.n_configs, bool) if allowed is None else allowed
        self.nJ, self.nS = inst.n_configs, inst.n_supports
        self.S_A, self.S_lo, self.S_hi = structural(inst, budget)
        self.paths: list[np.ndarray] = []            # exact cuts in z: Σ_{v∈γ} z_v ≥ 1
        self.cuts_x: list[sparse.csr_matrix] = []    # aggregated cuts: Σ_j max_{v∈γ} C_vj x_j ≥ 1
        self.keys_x: set[tuple] = set()
        self.keys_z: set[tuple] = set()

    def cut_x(self, nodes: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.C[:, nodes].max(axis=1).T)

    def add(self, nodes: np.ndarray, x_hat: np.ndarray, z_ok: bool = True) -> str | None:
        """Cut in x if it is violated by x̂; otherwise exact cut in z. None if already known.

        The cut in x does not see combinations of partial sensors: if x̂ satisfies it
        while the corridor is open, only the cut in z (with links) removes it.
        """
        k = tuple(np.unique(nodes))
        if k not in self.keys_x:
            cx = self.cut_x(np.array(k, np.int64))
            if cx[:, x_hat].sum() < 1.0 - TOL:
                self.keys_x.add(k)
                self.cuts_x.append(cx)
                return "x"
        if z_ok and k not in self.keys_z:
            self.keys_z.add(k)
            self.paths.append(np.array(k, np.int64))
            if k not in self.keys_x:                  # the aggregated version strengthens the relaxation
                self.keys_x.add(k)
                self.cuts_x.append(self.cut_x(self.paths[-1]))
            return "z"
        return None

    def solve(self, time_s: float, gap: float, cap: float = np.inf):
        nJ, nS = self.nJ, self.nS
        Z = np.unique(np.concatenate(self.paths)) if self.paths else np.zeros(0, np.int64)
        nZ = Z.size
        G = self.cuts_F.G if self.cuts_F is not None else 0
        n = nJ + nS + nZ + G
        pos = {v: k for k, v in enumerate(Z)}

        def pad(M, before, after):
            return sparse.hstack([sparse.csr_matrix((M.shape[0], before)), M,
                                  sparse.csr_matrix((M.shape[0], after))]).tocsr()

        blocks = [sparse.hstack([self.S_A, sparse.csr_matrix((self.S_A.shape[0], nZ + G))]).tocsr()]
        lo, hi = [self.S_lo], [self.S_hi]
        if self.cuts_x:
            X = sparse.vstack(self.cuts_x, format="csr")
            blocks.append(sparse.hstack([X, sparse.csr_matrix((X.shape[0], nS + nZ + G))]).tocsr())
            lo.append(np.ones(X.shape[0])); hi.append(np.full(X.shape[0], np.inf))
        if nZ:
            # link: z_v − Σ_j C_vj x_j ≤ 0
            L = sparse.hstack([-self.C[:, Z].T.tocsr(), sparse.csr_matrix((nZ, nS)),
                               sparse.identity(nZ, format="csr"),
                               sparse.csr_matrix((nZ, G))]).tocsr()
            blocks.append(L); lo.append(np.full(nZ, -np.inf)); hi.append(np.zeros(nZ))
            # path cuts: Σ_{v∈γ} z_v ≥ 1
            r = np.concatenate([np.full(c.size, i) for i, c in enumerate(self.paths)])
            cc = np.array([pos[v] for c in self.paths for v in c])
            P = sparse.csr_matrix((np.ones(r.size), (r, cc)), shape=(len(self.paths), nZ))
            blocks.append(pad(P, nJ + nS, G)); lo.append(np.ones(P.shape[0]))
            hi.append(np.full(P.shape[0], np.inf))
        if G:
            M, rhs = self.cuts_F.matrix()                 # columns [x, y, η]
            M = sparse.hstack([M[:, :nJ + nS], sparse.csr_matrix((M.shape[0], nZ)),
                               M[:, nJ + nS:]]).tocsr()
            blocks.append(M); lo.append(np.full(rhs.size, -np.inf)); hi.append(rhs)
            c = np.r_[np.zeros(nJ + nS + nZ), -np.ones(G)]
            ub = np.r_[np.ones(nJ + nS + nZ), self.cuts_F.Wg]
        else:
            # costs multiple of a step q: integer objective (÷q), HiGHS rounds the bound
            k = 1.0 / self.step if self.step else 1.0
            c = np.r_[k * self.inst.cost_sensor, k * self.inst.cost_support, np.zeros(nZ)]
            ub = np.ones(n)
            if np.isfinite(cap):                  # cutoff: strictly cheaper than the incumbent
                blocks.append(sparse.csr_matrix(c[None, :]))
                lo.append(np.array([-np.inf])); hi.append(np.array([k * cap]))
        ub[:nJ] = ub[:nJ] * self.allowed
        res = milp(c, constraints=LinearConstraint(sparse.vstack(blocks, format="csr"),
                                                   np.concatenate(lo), np.concatenate(hi)),
                   bounds=Bounds(np.zeros(n), ub),
                   integrality=np.r_[np.ones(nJ + nS + nZ), np.zeros(G)],
                   options=dict(time_limit=max(1.0, time_s), mip_rel_gap=gap))
        return res, nZ


def adversarial_loop(inst: Instance, dom: Domain, A: sparse.csr_matrix, d: np.ndarray,
                     theta: float, T_R: float, profile: str = "F", objective: str = "cost",
                     budget: float = 1e6, groups: np.ndarray | None = None, seeds=(),
                     n_paths: int = 5, max_time_s: float = 600.0, master_time_s: float = 120.0,
                     gap: float = 5e-3, max_iter: int = 500, fusion: str = "product",
                     pool: list | None = None, incumbent=None,
                     verbose: bool = False) -> LoopResult:
    """Guaranteed timely detection ≥ θ: minimal cost, or maximal F for a given budget.

    pool: list of corridors (start → target) shared between cases. Every corridor gives a
    valid cut for every (θ, T_R, fusion) after intersection with the useful nodes of the
    case; the corridors found here are added to it.
    incumbent: known feasible solution (for example the optimum under "max" fusion, which
    is feasible under "product" since 1 − Π(1 − P) ≥ max P).
    """
    assert objective in ("cost", "F")
    t0 = time.perf_counter()
    C = coefficients(A, theta, fusion)
    useful = d >= T_R
    capable = np.asarray(C.sum(axis=0)).ravel() >= 1.0 - TOL      # ceiling of the catalog
    targetable = useful & capable
    cuts_F = None
    if objective == "F":
        cuts_F = Cuts(inst, groups)
        cuts_F.add([])
    allowed = None
    if objective == "cost":
        allowed = useful_configurations(inst, C, useful, fusion == "max")
    M = Master(inst, C, budget, cuts_F, allowed)

    best_T, best = np.zeros(0, np.int64), (np.inf if objective == "cost" else 0.0)
    n_kind: dict = {}

    def feasible(T) -> bool:
        return not intruder_oracle(dom, level(C, T), useful, profile, 1)

    z_allowed = fusion == "product"           # under "max", the cut in x is exact

    def cut(T, z_ok: bool = True, k: int = n_paths) -> tuple[int, bool]:
        """Adds the cuts violated by T; (number added, infeasible by construction)."""
        n = 0
        T = np.asarray(T, np.int64)
        for ch in intruder_oracle(dom, level(C, T), useful, profile, k):
            nodes = ch[targetable[ch]]
            if nodes.size == 0:
                return n, True                    # no sensor can close this corridor
            if pool is not None:
                pool.append(ch)
            kind = M.add(nodes, T, z_ok and z_allowed)
            n += kind is not None
            n_kind[kind] = n_kind.get(kind, 0) + 1
        return n, False

    for ch in (pool or [])[:]:                    # corridors of the previous cases
        nodes = ch[targetable[ch]]
        if nodes.size == 0:
            return LoopResult("infeasible", np.zeros(0, np.int64), np.nan, np.nan, 0,
                              len(M.cuts_x), time.perf_counter() - t0)
        M.add(nodes, np.zeros(0, np.int64), False)
    if incumbent is not None:
        seeds = list(seeds) + [np.asarray(incumbent, np.int64)]
    for T in seeds:                               # known solutions: cuts and incumbent
        T = np.asarray(T, np.int64)
        _, imp = cut(T, z_ok=False, k=3)          # seeds: a few cuts in x only
        if imp:
            return LoopResult("infeasible", np.zeros(0, np.int64), np.nan, np.nan, 0,
                              len(M.paths), time.perf_counter() - t0)
        if cuts_F is not None:
            cuts_F.add(T)
        if not violations(inst, T, budget) and feasible(T):
            v = cost(inst, T) if objective == "cost" else evaluate(inst, T)
            if (v < best) if objective == "cost" else (v > best):
                best_T, best = T, v
    _, imp = cut(np.zeros(0, np.int64))           # the intruder against the empty site
    if imp:
        return LoopResult("infeasible", np.zeros(0, np.int64), np.nan, np.nan, 0,
                          len(M.paths), time.perf_counter() - t0)

    bound = -np.inf if objective == "cost" else np.inf
    hist, status = [], "limit"
    for it in range(1, max_iter + 1):
        left = max_time_s - (time.perf_counter() - t0)
        if left <= 0:
            break
        if objective == "cost":
            margin = M.step / 2 if M.step else 1e-4 * max(1.0, best)   # > HiGHS tolerance
            res, nZ = M.solve(min(master_time_s, left), 0.05, best - margin)
        else:
            res, nZ = M.solve(min(master_time_s, left), 1e-4)
        db = getattr(res, "mip_dual_bound", None)
        if objective == "cost" and db is not None and np.isfinite(db):
            # with a cutoff, the dual bound only holds under the incumbent: min(db, incumbent)
            bound = max(bound, min(db * (M.step or 1.0), best))
        if res.x is None:
            if res.status == 2:                   # infeasible master
                if objective == "cost" and np.isfinite(best):
                    status, bound = "optimal", best   # nothing cheaper than the incumbent
                else:
                    status = "infeasible"
            break
        if objective == "F":
            bound = min(bound, -res.mip_dual_bound)
        T = np.flatnonzero(res.x[:inst.n_configs] > 0.5)
        n_new, imp = cut(T)
        ok = n_new == 0 and not imp
        if ok:
            v = cost(inst, T) if objective == "cost" else evaluate(inst, T)
            if (v < best) if objective == "cost" else (v > best):
                best_T, best = T, v
        n_F = cuts_F.add(T) if cuts_F is not None else False
        if objective == "cost":
            e = (best - bound) / best if np.isfinite(best) and best > 0 else np.inf
        else:
            e = (bound - best) / bound if bound > EPS else 0.0
        hist.append(dict(iteration=it, time_s=time.perf_counter() - t0, bound=bound,
                         best=best, gap=e, n_cuts_x=len(M.cuts_x),
                         n_cuts_z=len(M.paths), n_z=nZ,
                         feasible=ok))
        if verbose:
            print(f"    it {it:3d} | {hist[-1]['time_s']:6.1f} s | bound {bound:.4f} | "
                  f"best {best:.4f} | cuts x {len(M.cuts_x)}, z {len(M.paths)} "
                  f"({nZ} z)", flush=True)
        if imp:
            status = "infeasible"
            break
        if objective == "cost":
            # exact stop: bound within half a step of the incumbent (or infeasible master)
            if np.isfinite(best) and best - bound <= (M.step / 2 if M.step else 1e-4 * best):
                status, bound = "optimal", best
                break
            continue                              # otherwise: new cutoff, new master
        if e <= gap:
            status = "optimal"
            break
        if n_new == 0 and not n_F:
            status = "optimal" if res.status == 0 else "limit"
            break
    if status == "optimal" and objective == "cost":
        bound = best
        hist.append(dict(iteration=len(hist) + 1, time_s=time.perf_counter() - t0, bound=bound,
                         best=best, gap=0.0, n_cuts_x=len(M.cuts_x),
                         n_cuts_z=len(M.paths), n_z=np.nan, feasible=True))
    return LoopResult(status, best_T, best, bound, len(hist),
                      len(M.cuts_x) + len(M.paths), time.perf_counter() - t0,
                      pd.DataFrame(hist))


# =============================================================================
# Step 3: independent check of a solution
# =============================================================================
def p_tilde(dom: Domain, A, T, d: np.ndarray, T_R: float, profile: str) -> float:
    """p̃* recomputed by src.intruder, without going through the coefficients C."""
    P, _ = detection(A, T)
    return corridor_threshold(dom, np.where(d >= T_R, P, 0.0), profile)[0]


# =============================================================================
# Step 4: figures
# =============================================================================
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
C_TR = ("#2a78d6", "#1f9e6e", "#eb6834", "#b8456a", "#4a3aa7")


def plot_cost(df, T_min, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150, layout="constrained")
    for col, (th, g) in zip(C_TR, df.groupby("theta")):
        g = g.sort_values("T_R")
        o = g[g["status_max"] == "optimal"]
        ax.plot(o["T_R"], o["cost_max"], color=col, lw=2, marker="o", ms=5,
                label=f"θ = {th:g}: certified optimum")
        f = g.dropna(subset=["cost_front"])
        ax.plot(f["T_R"], f["cost_front"], color=col, lw=1, ls=":", marker="x", ms=6,
                label=f"θ = {th:g}: best point of the NSGA-II front")
        if "bound_product" in g:
            q = g[np.isfinite(g["bound_product"].astype(float))]
            ax.vlines(q["T_R"] + 0.5, q["bound_product"], q["cost_product"], color=col, lw=5,
                      alpha=0.3)
    ax.axvline(T_min, color=INK_2, lw=0.8, ls="--")
    ax.text(T_min - 0.4, ax.get_ylim()[0] + 0.3, f"T_min = {T_min:.1f} s\nbeyond: impossible ",
            color=INK_2, fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel("Response time T_R (s)", color=INK_2)
    ax.set_ylabel("Minimal cost (units)", color=INK_2)
    ax.set_xticks(sorted(df["T_R"].unique()))
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(frameon=False, fontsize=7, loc="outside right upper",
               title="light bar: bracket\nunder independence", title_fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=150, layout="constrained")
    for col, (k, h) in zip(C_TR * 4, hist.groupby("case", sort=False)):
        h = h.sort_values("iteration")
        ax.step(h["iteration"], h["bound"].clip(lower=np.nanmin(hist["bound"].replace(-np.inf, np.nan))),
                where="post", color=col, lw=1.8, label=k)
        m = h[np.isfinite(h["best"])]
        ax.step(m["iteration"], m["best"], where="post", color=col, lw=1, ls="--")
        ax.plot(h["iteration"].iloc[-1], h["bound"].iloc[-1], color=col, marker="*", ms=10)
    ax.set_xlabel("Iteration (master → intruder)", color=INK_2)
    ax.set_ylabel("Cost (units)", color=INK_2)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(frameon=False, fontsize=7, loc="outside right upper",
               title="solid: lower bound\ndashed: best solution\n"
                     "★: certified optimum", title_fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figures(root, cfg, T_min):
    """Regenerates the figures from the CSVs (python -m src.adversarial_loop figures)."""
    out = root / cfg["loop"]["out_dir"]
    fig_dir = root / cfg["optimization"]["figures_dir"]
    df = pd.read_csv(out / "robust_cost.csv")
    plot_cost(df, T_min, fig_dir / "robust_cost.png")
    H = pd.read_csv(out / "convergence.csv")
    H = H[(H["fusion"] == "max") & (H["theta"] == 0.9)].copy()
    last = []                                  # final certified point (older outputs)
    for (th, T_R), g in H.groupby(["theta", "T_R"]):
        r = df[(df["theta"] == th) & (df["T_R"] == T_R)].iloc[0]
        if r["status_max"] == "optimal" and not np.isclose(g["bound"].iloc[-1], r["cost_max"]):
            last.append(dict(iteration=g["iteration"].max() + 1, bound=r["cost_max"],
                             best=r["cost_max"], case=g["case"].iloc[0], theta=th, T_R=T_R,
                             fusion="max"))
    if last:
        H = pd.concat([H, pd.DataFrame(last)])
    plot_convergence(H, fig_dir / "loop_convergence.png")


# =============================================================================
# Step 5: main program
# =============================================================================
def load(cfg, root):
    from src.instance import from_configs
    from src.intruder import load_domain
    from src.timely import time_remaining

    dom, A, conf = load_domain(cfg, root)
    cells = np.load(root / cfg["coverage"]["out_dir"] / "cells.npz")
    n_int = int(dom.inside.sum())
    assert dom.inside[:n_int].all()
    inst = from_configs(A[:, :n_int], cells["w"], conf, cfg)
    d = {pr: time_remaining(dom, cfg["intruder"], pr) for pr in PROFILES}
    return dom, A, conf, inst, cells, d


def front_nsga(root, cfg) -> list[np.ndarray]:
    f = pd.read_csv(root / cfg["nsga"]["out_dir"] / "front_global.csv")
    return [np.array(str(s).split(), np.int64) for s in f["configs"].drop_duplicates()]


def main() -> None:
    from src.greedy import describe
    from src.milp import spatial_blocks
    from src.site import ROOT, load_config

    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    lc = cfg["loop"]
    out = ROOT / lc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    dom, A, conf, inst, cells, d = load(cfg, ROOT)
    front = front_nsga(ROOT, cfg)
    pr = lc["profile"]
    print(f"{inst.n_configs:,} configurations, {dom.n:,} nodes, {len(front)} front solutions\n")

    # ---- 1. Minimal robust cost c(θ, T_R) ---------------------------------------------
    # "max" fusion: exact set cover, certified optimum; its solution is feasible under
    # "product". "product" fusion: relaxation → lower bound, with the "max" optimum as the
    # starting solution. Hence c_product ∈ [bound, c_max].
    pool: list = []
    if what in ("all", "cost"):
        rows, hist, sols = [], [], []
        cf = np.array([cost(inst, T) for T in front])
        cases = sorted((float(th), float(t)) for th in lc["theta"] for t in lc["t_response_s"])
        for th, T_R in cases:
            pt = np.array([p_tilde(dom, A, T, d[pr], T_R, pr) for T in front])
            ok = pt >= th - 1e-12
            c_front = float(cf[ok].min()) if ok.any() else np.nan
            print(f"θ = {th:g}, T_R = {T_R:g} s  (front: {c_front:.2f})", flush=True)
            r = adversarial_loop(inst, dom, A, d[pr], th, T_R, pr, "cost", seeds=front,
                                 n_paths=lc["paths_per_iteration"], max_time_s=lc["max_time_s"],
                                 master_time_s=lc["master_time_s"], fusion="max",
                                 pool=pool, verbose=True)
            chk = p_tilde(dom, A, r.T, d[pr], T_R, pr) if r.T.size else np.nan
            if r.status == "optimal":
                assert chk >= th - 1e-6, f"check: p̃* = {chk} < θ = {th}"
                assert not violations(inst, r.T, 1e6)
            row = dict(theta=th, T_R=T_R, status_max=r.status, cost_max=r.value,
                       bound_max=r.bound, p_tilde_check=chk,
                       F=evaluate(inst, r.T) / inst.W if r.T.size else np.nan,
                       cost_front=c_front, iterations=r.iterations, time_max_s=r.time_s,
                       **(describe(inst, conf, r.T) if r.T.size else {}))
            hist.append(r.history.assign(case=f"θ = {th:g}, T_R = {T_R:g} s", theta=th,
                                         T_R=T_R, fusion="max"))
            sols += [dict(theta=th, T_R=T_R, fusion="max", config=int(j)) for j in r.T]
            print(f"  -> max: {r.status}, cost {r.value:.2f}, p̃* checked {chk:.3f}, "
                  f"{r.iterations} it., {r.time_s:.0f} s", flush=True)
            if th in lc["theta_product"] and r.status == "optimal":
                q = adversarial_loop(inst, dom, A, d[pr], th, T_R, pr, "cost", seeds=front,
                                     n_paths=lc["paths_per_iteration"],
                                     max_time_s=lc["product_time_s"],
                                     master_time_s=lc["master_time_s"],
                                     fusion="product", pool=pool, incumbent=r.T, verbose=True)
                chq = p_tilde(dom, A, q.T, d[pr], T_R, pr)
                assert chq >= th - 1e-6
                row.update(status_product=q.status, cost_product=q.value,
                           bound_product=q.bound, time_product_s=q.time_s)
                hist.append(q.history.assign(case=f"θ = {th:g}, T_R = {T_R:g} s", theta=th,
                                             T_R=T_R, fusion="product"))
                sols += [dict(theta=th, T_R=T_R, fusion="product", config=int(j)) for j in q.T]
                print(f"  -> product: {q.status}, cost ∈ [{q.bound:.2f}, {q.value:.2f}]",
                      flush=True)
            rows.append(row)
            print(flush=True)
            pd.DataFrame(rows).to_csv(out / "robust_cost.csv", index=False)
        df = pd.DataFrame(rows)
        pd.DataFrame(sols).to_csv(out / "solutions.csv", index=False)
        H = pd.concat(hist)
        H.to_csv(out / "convergence.csv", index=False)
        pd.set_option("display.width", 220)
        print(df.round(3).to_string(index=False))
        figures(ROOT, cfg, float(d[pr][dom.source].min()))

    if what == "figures":
        figures(ROOT, cfg, float(d[pr][dom.source].min()))

    # ---- 2. Certified price of robustness ---------------------------------------------
    if what in ("all", "price"):
        from src.milp import solve
        mc = cfg["optimization"]["milp"]
        block_px = int(round(mc["block_m"] / cfg["site"]["resolution_m"]))
        groups = spatial_blocks(cells["rows"], cells["cols"], block_px)
        W = inst.W
        rows = []
        chosen = [int(k) for k in sys.argv[2:]] if len(sys.argv) > 2 else range(len(lc["price"]))
        for case in [lc["price"][k] for k in chosen]:
            B, th, T_R = float(case["budget"]), float(case["theta"]), float(case["t_response_s"])
            seeds = [T for T in front if cost(inst, T) <= B + EPS]
            f_sol = out / "solutions.csv"          # certified optimum of the minimal cost, if it fits
            if f_sol.exists():
                sol = pd.read_csv(f_sol)
                sol = sol[(sol["fusion"] == "max") & np.isclose(sol["theta"], th)
                          & np.isclose(sol["T_R"], T_R)]
                if len(sol) and cost(inst, sol["config"].to_numpy()) <= B + EPS:
                    seeds.append(sol["config"].to_numpy(np.int64))
            print(f"B = {B:g}, θ = {th:g}, T_R = {T_R:g} s", flush=True)
            free = solve(inst, B, groups, seeds, mc["target_gap"], mc["master_time_s"],
                         lc["max_time_s"])
            rob = adversarial_loop(inst, dom, A, d[pr], th, T_R, pr, "F", budget=B, groups=groups,
                                   seeds=seeds, n_paths=lc["paths_per_iteration"],
                                   fusion="max", pool=pool,
                                   max_time_s=lc["max_time_s"], master_time_s=lc["master_time_s"],
                                   gap=mc["target_gap"], verbose=True)
            chk = p_tilde(dom, A, rob.T, d[pr], T_R, pr) if rob.T.size else np.nan
            assert rob.T.size == 0 or chk >= th - 1e-6
            rows.append(dict(budget=B, theta=th, T_R=T_R,
                             F_free=free.value / W, F_free_ub=free.ub / W,
                             F_robust=rob.value / W, F_robust_ub=rob.bound / W,
                             price_min=max(0.0, (free.value - rob.bound) / W),
                             price_max=(free.ub - rob.value) / W,
                             status=rob.status, p_tilde_check=chk, time_s=rob.time_s,
                             **(describe(inst, conf, rob.T) if rob.T.size else {})))
            r = rows[-1]
            print(f"  -> free F ∈ [{r['F_free']:.4f}, {r['F_free_ub']:.4f}] | robust F ∈ "
                  f"[{r['F_robust']:.4f}, {r['F_robust_ub']:.4f}] | price ∈ "
                  f"[{100 * r['price_min']:.2f}, {100 * r['price_max']:.2f}] pt\n", flush=True)
        df = pd.DataFrame(rows)
        f_price = out / "price_of_robustness.csv"
        if len(sys.argv) > 2 and f_price.exists():    # update of the recomputed cases only
            old = pd.read_csv(f_price)
            key = ["budget", "theta", "T_R"]
            old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
            df = pd.concat([old, df]).sort_values(key)
        df.to_csv(f_price, index=False)
        print(df.round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

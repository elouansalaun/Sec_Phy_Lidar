"""
src/boucle.py — Boucle adversariale exacte : optimiseur ↔ intrus par génération de chemins
(docs/formulation.md §9.2).

Linéarisation. Avec s_vj = −ln(1 − P_vj) et s_θ = −ln(1 − θ) :
    P_v ≥ θ  ⇔  Σ_j s_vj x_j ≥ s_θ  ⇔  Σ_j C_vj x_j ≥ 1,   C_vj = min(s_vj / s_θ, 1)
(la troncature est exacte : si un terme est tronqué, les deux membres sont vrais ; elle
resserre la relaxation et élimine s = ∞).

Détection opportune garantie ≥ θ (profil donné, temps de réponse T_R) :
    tout chemin départ → cible contient un nœud v avec d(v) ≥ T_R et z_v = 1,
    z_v ≤ Σ_j C_vj x_j,  z_v ∈ {0, 1}.

Maître   : (C1)–(C4) + liaisons z–x + coupes Σ_{v ∈ γ ∩ V_z} z_v ≥ 1 pour les chemins γ connus.
           Objectif « cout » : min Σ a_j x_j + Σ f_s y_s (coût minimal robuste c(θ, T_R)).
           Objectif « F »    : max Σ_g η_g avec les coupes sous-modulaires de src.milp, coût ≤ B.
Intrus   : union-find sur u_v = min(Σ_j C_vj x̂_j, 1), masqué par d(v) ≥ T_R ; s'il existe un
           couloir où u < 1, il devient une coupe (plusieurs couloirs disjoints par itération).
Le maître est une relaxation : sa borne est valide à chaque itération. Les chemins sont en
nombre fini : la boucle termine. Quand l'intrus ne trouve plus de couloir, la solution du
maître est réalisable, donc optimale (à l'écart du maître près).

Seuls les nœuds déjà traversés par une coupe portent une variable z : fixer z_v = 0 ailleurs
ne change rien, puisque z n'intervient que dans les coupes.

Sorties : boucle/cout_robuste.csv, boucle/prix_robustesse.csv, boucle/convergence.csv,
          boucle/solutions.csv, figures/cout_robuste.png, figures/boucle_convergence.png
Usage   : python -m src.boucle            (tout)
          python -m src.boucle cout       (coût minimal robuste seulement)
          python -m src.boucle prix       (prix de la robustesse certifié seulement)
          python -m src.boucle prix 2 3   (seulement les cas 2 et 3 de boucle.prix)
          python -m src.boucle figures    (figures depuis les CSV)
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
from src.intrus import PROFILS, Domaine, detection, seuil_couloir
from src.milp import Cuts, structural

TOL = 1e-6          # tolérance de faisabilité du maître sur z_v ≤ Σ C_vj x_j


# =============================================================================
# Étape 1 — Coefficients de liaison et oracle de l'intrus
# =============================================================================
def coefficients(A: sparse.csr_matrix, theta: float, fusion: str = "produit") -> sparse.csc_matrix:
    """Coefficients de liaison (configurations × nœuds), stockés par colonnes.

    fusion = "produit" : P_v = 1 − Π(1 − P_vj) (indépendance), C_vj = min(s_vj / s_θ, 1).
    fusion = "max"     : P_v = max_j P_vj (corrélation totale), C_vj = 1[P_vj ≥ θ].
    Dans les deux cas : P_v ≥ θ ⇔ min(Σ_{j∈T} C_vj, 1) = 1.
    """
    C = A.astype(np.float64).tocsr(copy=True)
    if fusion == "max":
        C.data = (C.data >= theta - 1e-12).astype(np.float64)
    elif fusion == "produit":
        s_theta = -np.log1p(-theta)
        with np.errstate(divide="ignore"):
            C.data = np.minimum(-np.log1p(-C.data) / s_theta, 1.0)
    else:
        raise ValueError(fusion)
    C.eliminate_zeros()
    return C.tocsc()


def niveau(C: sparse.csc_matrix, T) -> np.ndarray:
    """u_v = min(Σ_{j∈T} C_vj, 1) = min(σ_v / s_θ, 1) : u_v = 1 ⇔ P_v ≥ θ."""
    T = np.asarray(T, np.int64)
    if T.size == 0:
        return np.zeros(C.shape[1])
    return np.minimum(np.asarray(C.tocsr()[T].sum(axis=0)).ravel(), 1.0)


def intrus_oracle(dom: Domaine, u: np.ndarray, utile: np.ndarray, profil: str,
                  n_chemins: int) -> list[np.ndarray]:
    """Couloirs où aucun nœud utile n'atteint θ (u < 1). Chacun est une coupe violée par x̂.

    Après chaque couloir, ses nœuds utiles sont bloqués (u = 1) : le suivant évite les
    précédents, ce qui donne des coupes plus variées pour la même solution x̂.
    """
    w = np.where(utile, u, 0.0)
    out = []
    for _ in range(n_chemins):
        p, ch = seuil_couloir(dom, w, profil)
        if p >= 1.0 - TOL:
            break
        out.append(ch)
        w = w.copy()
        w[ch[utile[ch]]] = 1.0
    return out


def pas_cout(inst: Instance) -> float:
    """Pas q tel que tous les coûts soient des multiples de q (0,1 ou 0,01), sinon 0."""
    c = np.r_[inst.cost_sensor, inst.cost_support]
    for q in (0.1, 0.01):
        if np.allclose(c / q, np.round(c / q), atol=1e-9):
            return q
    return 0.0


def configurations_utiles(inst: Instance, C: sparse.csc_matrix, utile: np.ndarray,
                          dominance: bool) -> np.ndarray:
    """Configurations à garder pour le coût minimal (même optimum).

    Vide : aucun coefficient sur un nœud utile ; elle ne sert à aucune coupe et coûte > 0.
    Dominée (fusion « max » seulement) : même support et même type qu'une configuration j'
    au coût capteur ≤, dont l'ensemble couvert contient le sien. La remplacer par j' (ou la
    retirer si j' est déjà posée) garde (C3), (C4) et la réalisabilité, sans coûter plus.
    En fusion « produit », les contributions s'additionnent : l'argument ne tient pas.
    """
    Cu = C.tocsr()[:, np.flatnonzero(utile)].tocsr()
    Cu.data = (Cu.data > 0).astype(np.float64)
    taille = np.diff(Cu.indptr)
    ok = taille > 0
    if not dominance:
        return ok
    for s_ in np.unique(inst.support):
        J = np.flatnonzero((inst.support == s_) & ok)
        if J.size < 2:
            continue
        S = Cu[J]
        inter = (S @ S.T).toarray()                        # |S_a ∩ S_b|
        t = taille[J]
        for a_, ja in enumerate(J):
            sub = inter[a_] == t[a_]                       # S_a ⊆ S_b
            sub &= inst.kind[J] == inst.kind[ja]
            sub &= inst.cost_sensor[J] <= inst.cost_sensor[ja] + 1e-12
            egal = (t == t[a_]) & sub
            strict = sub & ~egal
            if strict.any() or (egal & (J < ja)).any():    # à égalité, on garde le plus petit indice
                ok[ja] = False
    return ok


# =============================================================================
# Étape 2 — Maître et boucle
# =============================================================================
@dataclass
class Resultat:
    statut: str                   # "optimal", "irrealisable", "limite"
    T: np.ndarray
    valeur: float                 # coût (objectif cout) ou F non normalisé (objectif F)
    borne: float                  # borne inférieure du coût, ou supérieure de F
    iterations: int
    n_coupes: int
    temps_s: float
    historique: pd.DataFrame = field(default_factory=pd.DataFrame)


class Maitre:
    """Assemble le MILP maître à partir des coupes courantes."""

    def __init__(self, inst: Instance, C: sparse.csc_matrix, budget: float,
                 cuts_F: Cuts | None, permis: np.ndarray | None = None):
        self.inst, self.C, self.cuts_F = inst, C, cuts_F
        self.pas = pas_cout(inst)
        self.permis = np.ones(inst.n_configs, bool) if permis is None else permis
        self.nJ, self.nS = inst.n_configs, inst.n_supports
        self.S_A, self.S_lo, self.S_hi = structural(inst, budget)
        self.chemins: list[np.ndarray] = []           # coupes exactes en z : Σ_{v∈γ} z_v ≥ 1
        self.coupes_x: list[sparse.csr_matrix] = []   # coupes agrégées : Σ_j max_{v∈γ} C_vj x_j ≥ 1
        self.cles_x: set[tuple] = set()
        self.cles_z: set[tuple] = set()

    def coupe_x(self, noeuds: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.C[:, noeuds].max(axis=1).T)

    def ajouter(self, noeuds: np.ndarray, x_hat: np.ndarray, z_ok: bool = True) -> str | None:
        """Coupe en x si elle est violée par x̂ ; sinon coupe exacte en z. None si déjà connue.

        La coupe en x ne voit pas les combinaisons de capteurs partiels : si x̂ la satisfait
        alors que le couloir est ouvert, seule la coupe en z (avec liaisons) l'élimine.
        """
        k = tuple(np.unique(noeuds))
        if k not in self.cles_x:
            cx = self.coupe_x(np.array(k, np.int64))
            if cx[:, x_hat].sum() < 1.0 - TOL:
                self.cles_x.add(k)
                self.coupes_x.append(cx)
                return "x"
        if z_ok and k not in self.cles_z:
            self.cles_z.add(k)
            self.chemins.append(np.array(k, np.int64))
            if k not in self.cles_x:                  # la version agrégée renforce la relaxation
                self.cles_x.add(k)
                self.coupes_x.append(self.coupe_x(self.chemins[-1]))
            return "z"
        return None

    def resoudre(self, temps_s: float, gap: float, plafond: float = np.inf):
        nJ, nS = self.nJ, self.nS
        Z = np.unique(np.concatenate(self.chemins)) if self.chemins else np.zeros(0, np.int64)
        nZ = Z.size
        G = self.cuts_F.G if self.cuts_F is not None else 0
        n = nJ + nS + nZ + G
        pos = {v: k for k, v in enumerate(Z)}

        def pad(M, avant, apres):
            return sparse.hstack([sparse.csr_matrix((M.shape[0], avant)), M,
                                  sparse.csr_matrix((M.shape[0], apres))]).tocsr()

        blocs = [sparse.hstack([self.S_A, sparse.csr_matrix((self.S_A.shape[0], nZ + G))]).tocsr()]
        lo, hi = [self.S_lo], [self.S_hi]
        if self.coupes_x:
            X = sparse.vstack(self.coupes_x, format="csr")
            blocs.append(sparse.hstack([X, sparse.csr_matrix((X.shape[0], nS + nZ + G))]).tocsr())
            lo.append(np.ones(X.shape[0])); hi.append(np.full(X.shape[0], np.inf))
        if nZ:
            # liaison : z_v − Σ_j C_vj x_j ≤ 0
            L = sparse.hstack([-self.C[:, Z].T.tocsr(), sparse.csr_matrix((nZ, nS)),
                               sparse.identity(nZ, format="csr"),
                               sparse.csr_matrix((nZ, G))]).tocsr()
            blocs.append(L); lo.append(np.full(nZ, -np.inf)); hi.append(np.zeros(nZ))
            # coupes de chemins : Σ_{v∈γ} z_v ≥ 1
            r = np.concatenate([np.full(c.size, i) for i, c in enumerate(self.chemins)])
            cc = np.array([pos[v] for c in self.chemins for v in c])
            P = sparse.csr_matrix((np.ones(r.size), (r, cc)), shape=(len(self.chemins), nZ))
            blocs.append(pad(P, nJ + nS, G)); lo.append(np.ones(P.shape[0]))
            hi.append(np.full(P.shape[0], np.inf))
        if G:
            M, rhs = self.cuts_F.matrix()                 # colonnes [x, y, η]
            M = sparse.hstack([M[:, :nJ + nS], sparse.csr_matrix((M.shape[0], nZ)),
                               M[:, nJ + nS:]]).tocsr()
            blocs.append(M); lo.append(np.full(rhs.size, -np.inf)); hi.append(rhs)
            c = np.r_[np.zeros(nJ + nS + nZ), -np.ones(G)]
            ub = np.r_[np.ones(nJ + nS + nZ), self.cuts_F.Wg]
        else:
            # coûts multiples d'un pas q : objectif entier (÷q), HiGHS arrondit la borne
            k = 1.0 / self.pas if self.pas else 1.0
            c = np.r_[k * self.inst.cost_sensor, k * self.inst.cost_support, np.zeros(nZ)]
            ub = np.ones(n)
            if np.isfinite(plafond):              # coupure : strictement moins cher que l'incumbent
                blocs.append(sparse.csr_matrix(c[None, :]))
                lo.append(np.array([-np.inf])); hi.append(np.array([k * plafond]))
        ub[:nJ] = ub[:nJ] * self.permis
        res = milp(c, constraints=LinearConstraint(sparse.vstack(blocs, format="csr"),
                                                   np.concatenate(lo), np.concatenate(hi)),
                   bounds=Bounds(np.zeros(n), ub),
                   integrality=np.r_[np.ones(nJ + nS + nZ), np.zeros(G)],
                   options=dict(time_limit=max(1.0, temps_s), mip_rel_gap=gap))
        return res, nZ


def boucle(inst: Instance, dom: Domaine, A: sparse.csr_matrix, d: np.ndarray, theta: float,
           T_R: float, profil: str = "F", objectif: str = "cout", budget: float = 1e6,
           groups: np.ndarray | None = None, graines=(), n_chemins: int = 5,
           temps_max_s: float = 600.0, temps_maitre_s: float = 120.0, ecart: float = 5e-3,
           max_iter: int = 500, fusion: str = "produit", reservoir: list | None = None,
           incumbent=None, verbose: bool = False) -> Resultat:
    """Détection opportune garantie ≥ θ : coût minimal, ou F maximal à budget donné.

    reservoir : liste de couloirs (départ → cible) partagée entre les cas. Tout couloir donne
    une coupe valide pour tout (θ, T_R, fusion) après intersection avec les nœuds utiles du
    cas ; les couloirs trouvés ici y sont ajoutés.
    incumbent : solution réalisable connue (par exemple l'optimum sous fusion « max », qui
    est réalisable sous « produit » puisque 1 − Π(1 − P) ≥ max P).
    """
    assert objectif in ("cout", "F")
    t0 = time.perf_counter()
    C = coefficients(A, theta, fusion)
    utile = d >= T_R
    capable = np.asarray(C.sum(axis=0)).ravel() >= 1.0 - TOL      # plafond du catalogue
    cible = utile & capable
    cuts_F = None
    if objectif == "F":
        cuts_F = Cuts(inst, groups)
        cuts_F.add([])
    permis = None
    if objectif == "cout":
        permis = configurations_utiles(inst, C, utile, fusion == "max")
    M = Maitre(inst, C, budget, cuts_F, permis)

    meilleur_T, meilleur = np.zeros(0, np.int64), (np.inf if objectif == "cout" else 0.0)
    n_type: dict = {}

    def realisable(T) -> bool:
        return not intrus_oracle(dom, niveau(C, T), utile, profil, 1)

    z_permis = fusion == "produit"            # en « max », la coupe en x est exacte

    def couper(T, z_ok: bool = True, k: int = n_chemins) -> tuple[int, bool]:
        """Ajoute les coupes violées par T ; (nombre ajouté, irréalisable par construction)."""
        n = 0
        T = np.asarray(T, np.int64)
        for ch in intrus_oracle(dom, niveau(C, T), utile, profil, k):
            noeuds = ch[cible[ch]]
            if noeuds.size == 0:
                return n, True                    # aucun capteur ne peut fermer ce couloir
            if reservoir is not None:
                reservoir.append(ch)
            genre = M.ajouter(noeuds, T, z_ok and z_permis)
            n += genre is not None
            n_type[genre] = n_type.get(genre, 0) + 1
        return n, False

    for ch in (reservoir or [])[:]:               # couloirs des cas précédents
        noeuds = ch[cible[ch]]
        if noeuds.size == 0:
            return Resultat("irrealisable", np.zeros(0, np.int64), np.nan, np.nan, 0,
                            len(M.coupes_x), time.perf_counter() - t0)
        M.ajouter(noeuds, np.zeros(0, np.int64), False)
    if incumbent is not None:
        graines = list(graines) + [np.asarray(incumbent, np.int64)]
    for T in graines:                             # solutions connues : coupes et incumbent
        T = np.asarray(T, np.int64)
        _, imp = couper(T, z_ok=False, k=3)  # graines : quelques coupes en x seulement
        if imp:
            return Resultat("irrealisable", np.zeros(0, np.int64), np.nan, np.nan, 0,
                            len(M.chemins), time.perf_counter() - t0)
        if cuts_F is not None:
            cuts_F.add(T)
        if not violations(inst, T, budget) and realisable(T):
            v = cost(inst, T) if objectif == "cout" else evaluate(inst, T)
            if (v < meilleur) if objectif == "cout" else (v > meilleur):
                meilleur_T, meilleur = T, v
    _, imp = couper(np.zeros(0, np.int64))        # l'intrus face au site vide
    if imp:
        return Resultat("irrealisable", np.zeros(0, np.int64), np.nan, np.nan, 0,
                        len(M.chemins), time.perf_counter() - t0)

    borne = -np.inf if objectif == "cout" else np.inf
    hist, statut = [], "limite"
    for it in range(1, max_iter + 1):
        reste = temps_max_s - (time.perf_counter() - t0)
        if reste <= 0:
            break
        if objectif == "cout":
            marge = M.pas / 2 if M.pas else 1e-4 * max(1.0, meilleur)   # > tolérance HiGHS
            res, nZ = M.resoudre(min(temps_maitre_s, reste), 0.05, meilleur - marge)
        else:
            res, nZ = M.resoudre(min(temps_maitre_s, reste), 1e-4)
        db = getattr(res, "mip_dual_bound", None)
        if objectif == "cout" and db is not None and np.isfinite(db):
            # avec coupure, la borne duale ne vaut que sous l'incumbent : min(db, incumbent)
            borne = max(borne, min(db * (M.pas or 1.0), meilleur))
        if res.x is None:
            if res.status == 2:                   # maître irréalisable
                if objectif == "cout" and np.isfinite(meilleur):
                    statut, borne = "optimal", meilleur   # rien de moins cher que l'incumbent
                else:
                    statut = "irrealisable"
            break
        if objectif == "cout":
            pass                                  # borne déjà relevée ci-dessus
        else:
            borne = min(borne, -res.mip_dual_bound)
        T = np.flatnonzero(res.x[:inst.n_configs] > 0.5)
        n_new, imp = couper(T)
        ok = n_new == 0 and not imp
        if ok:
            v = cost(inst, T) if objectif == "cout" else evaluate(inst, T)
            if (v < meilleur) if objectif == "cout" else (v > meilleur):
                meilleur_T, meilleur = T, v
        n_F = cuts_F.add(T) if cuts_F is not None else False
        if objectif == "cout":
            e = (meilleur - borne) / meilleur if np.isfinite(meilleur) and meilleur > 0 else np.inf
        else:
            e = (borne - meilleur) / borne if borne > EPS else 0.0
        hist.append(dict(iteration=it, temps_s=time.perf_counter() - t0, borne=borne,
                         meilleur=meilleur, ecart=e, n_coupes_x=len(M.coupes_x),
                         n_coupes_z=len(M.chemins), n_z=nZ,
                         realisable=ok))
        if verbose:
            print(f"    it {it:3d} | {hist[-1]['temps_s']:6.1f} s | borne {borne:.4f} | "
                  f"meilleur {meilleur:.4f} | coupes x {len(M.coupes_x)}, z {len(M.chemins)} "
                  f"({nZ} z)", flush=True)
        if imp:
            statut = "irrealisable"
            break
        if objectif == "cout":
            # arrêt exact : borne à moins d'un demi-pas de l'incumbent (ou maître irréalisable)
            if np.isfinite(meilleur) and meilleur - borne <= (M.pas / 2 if M.pas else 1e-4 * meilleur):
                statut, borne = "optimal", meilleur
                break
            continue                              # sinon : nouvelle coupure, nouveau maître
        if e <= ecart:
            statut = "optimal"
            break
        if n_new == 0 and not n_F:
            statut = "optimal" if res.status == 0 else "limite"
            break
    if statut == "optimal" and objectif == "cout":
        borne = meilleur
        hist.append(dict(iteration=len(hist) + 1, temps_s=time.perf_counter() - t0, borne=borne,
                         meilleur=meilleur, ecart=0.0, n_coupes_x=len(M.coupes_x),
                         n_coupes_z=len(M.chemins), n_z=np.nan, realisable=True))
    return Resultat(statut, meilleur_T, meilleur, borne, len(hist),
                    len(M.coupes_x) + len(M.chemins), time.perf_counter() - t0,
                    pd.DataFrame(hist))


# =============================================================================
# Étape 3 — Vérification indépendante d'une solution
# =============================================================================
def p_tilde(dom: Domaine, A, T, d: np.ndarray, T_R: float, profil: str) -> float:
    """p̃* recalculé par src.intrus, sans passer par les coefficients C."""
    P, _ = detection(A, T)
    return seuil_couloir(dom, np.where(d >= T_R, P, 0.0), profil)[0]


# =============================================================================
# Étape 4 — Figures
# =============================================================================
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
C_TR = ("#2a78d6", "#1f9e6e", "#eb6834", "#b8456a", "#4a3aa7")


def plot_cout(df, T_min, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150, layout="constrained")
    for col, (th, g) in zip(C_TR, df.groupby("theta")):
        g = g.sort_values("T_R")
        o = g[g["statut_max"] == "optimal"]
        ax.plot(o["T_R"], o["cout_max"], color=col, lw=2, marker="o", ms=5,
                label=f"θ = {th:g} : optimum certifié")
        f = g.dropna(subset=["cout_front"])
        ax.plot(f["T_R"], f["cout_front"], color=col, lw=1, ls=":", marker="x", ms=6,
                label=f"θ = {th:g} : meilleur point du front NSGA-II")
        if "borne_produit" in g:
            q = g[np.isfinite(g["borne_produit"].astype(float))]
            ax.vlines(q["T_R"] + 0.5, q["borne_produit"], q["cout_produit"], color=col, lw=5,
                      alpha=0.3)
    ax.axvline(T_min, color=INK_2, lw=0.8, ls="--")
    ax.text(T_min - 0.4, ax.get_ylim()[0] + 0.3, f"T_min = {T_min:.1f} s\nau-delà : impossible ",
            color=INK_2, fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel("Temps de réponse T_R (s)", color=INK_2)
    ax.set_ylabel("Coût minimal (unités)", color=INK_2)
    ax.set_xticks(sorted(df["T_R"].unique()))
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(frameon=False, fontsize=7, loc="outside right upper",
              title="barre claire : encadrement\nsous indépendance", title_fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=150, layout="constrained")
    for col, (k, h) in zip(C_TR * 4, hist.groupby("cas", sort=False)):
        h = h.sort_values("iteration")
        ax.step(h["iteration"], h["borne"].clip(lower=np.nanmin(hist["borne"].replace(-np.inf, np.nan))),
                where="post", color=col, lw=1.8, label=k)
        m = h[np.isfinite(h["meilleur"])]
        ax.step(m["iteration"], m["meilleur"], where="post", color=col, lw=1, ls="--")
        ax.plot(h["iteration"].iloc[-1], h["borne"].iloc[-1], color=col, marker="*", ms=10)
    ax.set_xlabel("Itération (maître → intrus)", color=INK_2)
    ax.set_ylabel("Coût (unités)", color=INK_2)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(frameon=False, fontsize=7, loc="outside right upper",
              title="trait plein : borne inférieure\ntirets : meilleure solution\n"
                    "★ : optimum certifié", title_fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figures(root, cfg, T_min):
    """Régénère les figures depuis les CSV (python -m src.boucle figures)."""
    out = root / cfg["boucle"]["out_dir"]
    fig_dir = root / cfg["optimisation"]["figures_dir"]
    df = pd.read_csv(out / "cout_robuste.csv")
    plot_cout(df, T_min, fig_dir / "cout_robuste.png")
    H = pd.read_csv(out / "convergence.csv")
    H = H[(H["fusion"] == "max") & (H["theta"] == 0.9)].copy()
    fin = []                                   # point final certifié (anciennes sorties)
    for (th, T_R), g in H.groupby(["theta", "T_R"]):
        r = df[(df["theta"] == th) & (df["T_R"] == T_R)].iloc[0]
        if r["statut_max"] == "optimal" and not np.isclose(g["borne"].iloc[-1], r["cout_max"]):
            fin.append(dict(iteration=g["iteration"].max() + 1, borne=r["cout_max"],
                            meilleur=r["cout_max"], cas=g["cas"].iloc[0], theta=th, T_R=T_R,
                            fusion="max"))
    if fin:
        H = pd.concat([H, pd.DataFrame(fin)])
    plot_convergence(H, fig_dir / "boucle_convergence.png")


# =============================================================================
# Étape 5 — Programme principal
# =============================================================================
def charger(cfg, root):
    from src.instance import from_configs
    from src.intrus import load_domaine
    from src.opportune import temps_restant

    dom, A, conf = load_domaine(cfg, root)
    cells = np.load(root / cfg["coverage"]["out_dir"] / "cellules.npz")
    n_int = int(dom.interieur.sum())
    assert dom.interieur[:n_int].all()
    inst = from_configs(A[:, :n_int], cells["w"], conf, cfg)
    d = {pr: temps_restant(dom, cfg["intrus"], pr) for pr in PROFILS}
    return dom, A, conf, inst, cells, d


def front_nsga(root, cfg) -> list[np.ndarray]:
    f = pd.read_csv(root / cfg["nsga"]["out_dir"] / "front_global.csv")
    return [np.array(str(s).split(), np.int64) for s in f["configs"].drop_duplicates()]


def main() -> None:
    from src.greedy import describe
    from src.milp import spatial_blocks
    from src.site import ROOT, load_config

    quoi = sys.argv[1] if len(sys.argv) > 1 else "tout"
    cfg = load_config()
    bc = cfg["boucle"]
    out = ROOT / bc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / cfg["optimisation"]["figures_dir"]
    dom, A, conf, inst, cells, d = charger(cfg, ROOT)
    front = front_nsga(ROOT, cfg)
    pr = bc["profil"]
    print(f"{inst.n_configs:,} configurations, {dom.n:,} nœuds, {len(front)} solutions du front\n")

    # ---- 1. Coût minimal robuste c(θ, T_R) ----------------------------------------------
    # Fusion « max » : couverture d'ensembles exacte, optimum certifié ; sa solution est
    # réalisable sous « produit ». Fusion « produit » : relaxation → borne inférieure, avec
    # l'optimum « max » comme solution de départ. D'où c_produit ∈ [borne, c_max].
    reservoir: list = []
    if quoi in ("tout", "cout"):
        rows, hist, sols, ref = [], [], [], []
        cf = np.array([cost(inst, T) for T in front])
        cas = sorted((float(th), float(t)) for th in bc["theta"] for t in bc["t_reponse_s"])
        for th, T_R in cas:
            pt = np.array([p_tilde(dom, A, T, d[pr], T_R, pr) for T in front])
            ok = pt >= th - 1e-12
            c_front = float(cf[ok].min()) if ok.any() else np.nan
            print(f"θ = {th:g}, T_R = {T_R:g} s  (front : {c_front:.2f})", flush=True)
            r = boucle(inst, dom, A, d[pr], th, T_R, pr, "cout", graines=front,
                       n_chemins=bc["chemins_par_iteration"], temps_max_s=bc["temps_max_s"],
                       temps_maitre_s=bc["temps_maitre_s"], fusion="max",
                       reservoir=reservoir, verbose=True)
            chk = p_tilde(dom, A, r.T, d[pr], T_R, pr) if r.T.size else np.nan
            if r.statut == "optimal":
                assert chk >= th - 1e-6, f"vérification : p̃* = {chk} < θ = {th}"
                assert not violations(inst, r.T, 1e6)
            row = dict(theta=th, T_R=T_R, statut_max=r.statut, cout_max=r.valeur,
                       borne_max=r.borne, p_tilde_verif=chk,
                       F=evaluate(inst, r.T) / inst.W if r.T.size else np.nan,
                       cout_front=c_front, iterations=r.iterations, temps_max_s=r.temps_s,
                       **(describe(inst, conf, r.T) if r.T.size else {}))
            hist.append(r.historique.assign(cas=f"θ = {th:g}, T_R = {T_R:g} s", theta=th,
                                            T_R=T_R, fusion="max"))
            sols += [dict(theta=th, T_R=T_R, fusion="max", config=int(j)) for j in r.T]
            print(f"  -> max : {r.statut}, coût {r.valeur:.2f}, p̃* vérifié {chk:.3f}, "
                  f"{r.iterations} it., {r.temps_s:.0f} s", flush=True)
            if th in bc["theta_produit"] and r.statut == "optimal":
                q = boucle(inst, dom, A, d[pr], th, T_R, pr, "cout", graines=front,
                           n_chemins=bc["chemins_par_iteration"],
                           temps_max_s=bc["temps_produit_s"], temps_maitre_s=bc["temps_maitre_s"],
                           fusion="produit", reservoir=reservoir, incumbent=r.T, verbose=True)
                chq = p_tilde(dom, A, q.T, d[pr], T_R, pr)
                assert chq >= th - 1e-6
                row.update(statut_produit=q.statut, cout_produit=q.valeur,
                           borne_produit=q.borne, temps_produit_s=q.temps_s)
                hist.append(q.historique.assign(cas=f"θ = {th:g}, T_R = {T_R:g} s", theta=th,
                                                T_R=T_R, fusion="produit"))
                sols += [dict(theta=th, T_R=T_R, fusion="produit", config=int(j)) for j in q.T]
                print(f"  -> produit : {q.statut}, coût ∈ [{q.borne:.2f}, {q.valeur:.2f}]",
                      flush=True)
            rows.append(row)
            print(flush=True)
            pd.DataFrame(rows).to_csv(out / "cout_robuste.csv", index=False)
        df = pd.DataFrame(rows)
        pd.DataFrame(sols).to_csv(out / "solutions.csv", index=False)
        H = pd.concat(hist)
        H.to_csv(out / "convergence.csv", index=False)
        pd.set_option("display.width", 220)
        print(df.round(3).to_string(index=False))
        figures(ROOT, cfg, float(d[pr][dom.source].min()))

    if quoi == "figures":
        figures(ROOT, cfg, float(d[pr][dom.source].min()))

    # ---- 2. Prix de la robustesse certifié --------------------------------------------
    if quoi in ("tout", "prix"):
        from src.milp import solve
        mc = cfg["optimisation"]["milp"]
        block_px = int(round(mc["bloc_m"] / cfg["site"]["resolution_m"]))
        groups = spatial_blocks(cells["rows"], cells["cols"], block_px)
        W = inst.W
        rows = []
        choix = [int(k) for k in sys.argv[2:]] if len(sys.argv) > 2 else range(len(bc["prix"]))
        for cas in [bc["prix"][k] for k in choix]:
            B, th, T_R = float(cas["budget"]), float(cas["theta"]), float(cas["t_reponse_s"])
            graines = [T for T in front if cost(inst, T) <= B + EPS]
            f_sol = out / "solutions.csv"          # optimum certifié du coût minimal, s'il tient
            if f_sol.exists():
                sol = pd.read_csv(f_sol)
                sol = sol[(sol["fusion"] == "max") & np.isclose(sol["theta"], th)
                          & np.isclose(sol["T_R"], T_R)]
                if len(sol) and cost(inst, sol["config"].to_numpy()) <= B + EPS:
                    graines.append(sol["config"].to_numpy(np.int64))
            print(f"B = {B:g}, θ = {th:g}, T_R = {T_R:g} s", flush=True)
            libre = solve(inst, B, groups, graines, mc["ecart_cible"], mc["temps_maitre_s"],
                          bc["temps_max_s"])
            rob = boucle(inst, dom, A, d[pr], th, T_R, pr, "F", budget=B, groups=groups,
                         graines=graines, n_chemins=bc["chemins_par_iteration"],
                         fusion="max", reservoir=reservoir,
                         temps_max_s=bc["temps_max_s"], temps_maitre_s=bc["temps_maitre_s"],
                         ecart=mc["ecart_cible"], verbose=True)
            chk = p_tilde(dom, A, rob.T, d[pr], T_R, pr) if rob.T.size else np.nan
            assert rob.T.size == 0 or chk >= th - 1e-6
            rows.append(dict(budget=B, theta=th, T_R=T_R,
                             F_libre=libre.value / W, F_libre_ub=libre.ub / W,
                             F_robuste=rob.valeur / W, F_robuste_ub=rob.borne / W,
                             prix_min=max(0.0, (libre.value - rob.borne) / W),
                             prix_max=(libre.ub - rob.valeur) / W,
                             statut=rob.statut, p_tilde_verif=chk, temps_s=rob.temps_s,
                             **(describe(inst, conf, rob.T) if rob.T.size else {})))
            r = rows[-1]
            print(f"  -> F libre ∈ [{r['F_libre']:.4f}, {r['F_libre_ub']:.4f}] | F robuste ∈ "
                  f"[{r['F_robuste']:.4f}, {r['F_robuste_ub']:.4f}] | prix ∈ "
                  f"[{100 * r['prix_min']:.2f}, {100 * r['prix_max']:.2f}] pt\n", flush=True)
        df = pd.DataFrame(rows)
        f_prix = out / "prix_robustesse.csv"
        if len(sys.argv) > 2 and f_prix.exists():     # mise à jour des seuls cas recalculés
            old = pd.read_csv(f_prix)
            key = ["budget", "theta", "T_R"]
            old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
            df = pd.concat([old, df]).sort_values(key)
        df.to_csv(f_prix, index=False)
        print(df.round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

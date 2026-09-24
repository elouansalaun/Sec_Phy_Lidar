"""Tests des métriques de l'intrus (src/intrus.py, formulation §7.8) sur des domaines synthétiques."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from src.intrus import Domaine, adjacency, exposition, seuil_couloir
from src.intrusion import build_edges
from src.site import Grid

IC = dict(vitesse_m_s=1.5, t_cloture_s=10.0, t_portail_s=3.0)


def domaine(mask, source, cible, inside=None, res=1.0, portail_rows=None):
    """Domaine sur une grille : `source`/`cible`/`inside` sont des masques (H, W)."""
    H, W = mask.shape
    g = Grid(crs="EPSG:2950", xmin=0.0, ymin=0.0, xmax=W * res, ymax=H * res, res=res)
    rows, cols = np.nonzero(mask)
    ins = np.zeros(rows.size, bool) if inside is None else inside[rows, cols]
    u, v, l, fr = build_edges(g, rows, cols, ins)
    portail = np.full(u.size, -1, np.int16)
    if portail_rows is not None:          # franchissements dont la cellule extérieure est sur ces lignes
        ext = np.where(~ins[u], u, v)
        portail[fr & np.isin(rows[ext], portail_rows)] = 0
    dom = Domaine(rows=rows, cols=cols, interieur=ins, source=source[rows, cols],
                  cible=cible[rows, cols], u=u, v=v, longueur=l, franchit=fr, portail=portail)
    return dom, rows, cols


def cols_mask(H, W, c):
    m = np.zeros((H, W), bool)
    m[:, c] = True
    return m


def p_star_dichotomie(dom, P, profil):
    """Référence brute : plus petit t tel que {P ≤ t} relie source et cible."""
    mask = dom.aretes(profil)
    for t in np.unique(P):
        ok = P <= t
        keep = mask & ok[dom.u] & ok[dom.v]
        _, lab = connected_components(adjacency(dom, keep), directed=False)
        if np.isin(lab[dom.source & ok], lab[dom.cible & ok]).any():
            return t
    return np.inf


# ---------------------------------------------------------------------------- p*
def test_couloir_entre_deux_murs():
    """P = 0,9 partout sauf un couloir à 0,3 : p* = 0,3 et le couloir reste dans la ligne."""
    H, W = 7, 12
    dom, r, c = domaine(np.ones((H, W), bool), cols_mask(H, W, 0), cols_mask(H, W, W - 1))
    P = np.where(r == 3, 0.3, 0.9)
    p, path = seuil_couloir(dom, P, "F")
    assert p == pytest.approx(0.3)
    assert (r[path] == 3).all() and P[path].max() == pytest.approx(0.3)


def test_union_find_egale_dichotomie():
    rng = np.random.default_rng(0)
    for _ in range(20):
        H, W = 9, 11
        mask = rng.random((H, W)) > 0.15
        mask[:, 0] = mask[:, -1] = True
        dom, *_ = domaine(mask, cols_mask(H, W, 0), cols_mask(H, W, W - 1))
        P = rng.random(dom.n).round(2)
        ref = p_star_dichotomie(dom, P, "F")
        if not np.isfinite(ref):
            continue
        assert seuil_couloir(dom, P, "F")[0] == pytest.approx(ref)


def test_source_egale_cible():
    """Un nœud à la fois départ et cible : p* = sa propre détection, Λ = 0."""
    m = np.ones((3, 3), bool)
    s = np.zeros((3, 3), bool); s[1, 1] = True
    dom, *_ = domaine(m, s, s)
    P = np.full(dom.n, 0.4)
    assert seuil_couloir(dom, P, "F")[0] == pytest.approx(0.4)
    assert exposition(dom, -np.log1p(-P), "F", IC, 1.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_goulot():
    """Deux salles reliées par une seule cellule : le couloir passe par le goulot."""
    m = np.zeros((5, 9), bool)
    m[:, :4] = m[:, 5:] = True
    m[2, 4] = True
    dom, r, c = domaine(m, cols_mask(5, 9, 0), cols_mask(5, 9, 8))
    P = np.zeros(dom.n)
    goulot = np.flatnonzero((r == 2) & (c == 4))[0]
    P[goulot] = 0.7
    p, path = seuil_couloir(dom, P, "F")
    assert p == pytest.approx(0.7) and goulot in path
    assert goulot in exposition(dom, -np.log1p(-P), "F", IC, 1.0)[1]


def test_aucun_chemin():
    m = np.ones((3, 5), bool)
    m[:, 2] = False
    dom, *_ = domaine(m, cols_mask(3, 5, 0), cols_mask(3, 5, 4))
    with pytest.raises(ValueError):
        seuil_couloir(dom, np.zeros(dom.n), "F")


# ---------------------------------------------------------------------------- Λ
def test_exposition_ligne_droite():
    """σ uniforme sur une ligne de W cellules : Λ = σ·(W − 1)·res / (v·τ)."""
    W, sig, tau = 20, 0.05, 2.0
    m = np.ones((1, W), bool)
    dom, *_ = domaine(m, cols_mask(1, W, 0), cols_mask(1, W, W - 1))
    lam, _ = exposition(dom, np.full(dom.n, sig), "F", IC, tau)
    assert lam == pytest.approx(sig * (W - 1) / (IC["vitesse_m_s"] * tau))


def test_invariance_resolution():
    """σ(x) linéaire : Λ identique à 1 m et 0,5 m (trapèzes exacts) ; le produit par cellule, non."""
    L = 20.0

    def run(res):
        n = int(L / res) + 1
        m = np.ones((1, n), bool)
        dom, r, c = domaine(m, cols_mask(1, n, 0), cols_mask(1, n, n - 1), res=res)
        x = c * res
        sigma = 0.01 + 0.004 * x
        lam, path = exposition(dom, sigma, "F", IC, 1.0)
        prod = 1 - np.prod(np.exp(-sigma[path]))          # une observation par cellule
        return lam, prod

    (l1, p1), (l2, p2) = run(1.0), run(0.5)
    assert l1 == pytest.approx(l2, rel=1e-9)
    assert abs(p2 - p1) > 0.1                              # le produit dépend de la grille


def test_temps_de_franchissement():
    """Deux cellules de part et d'autre de la clôture : Λ = σ·(ℓ/v + t_c)/τ, ou t_p au portail."""
    m = np.ones((1, 2), bool)
    ins = np.array([[False, True]])
    s = np.array([[True, False]]); t = ~s
    sig = 0.2
    dom, *_ = domaine(m, s, t, inside=ins)
    lam, _ = exposition(dom, np.full(2, sig), "F", IC, 1.0)
    assert lam == pytest.approx(sig * (1 / IC["vitesse_m_s"] + IC["t_cloture_s"]))
    dom_g, *_ = domaine(m, s, t, inside=ins, portail_rows=[0])
    lam_g, _ = exposition(dom_g, np.full(2, sig), "G", IC, 1.0)
    assert lam_g == pytest.approx(sig * (1 / IC["vitesse_m_s"] + IC["t_portail_s"]))


# ---------------------------------------------------------------------------- profils
def cloture(H=9, W=10, murs=False):
    """Clôture verticale entre les colonnes 4 et 5, portail sur la ligne 4."""
    m = np.ones((H, W), bool)
    ins = np.zeros((H, W), bool); ins[:, 5:] = True
    if murs:                                   # la clôture n'est franchissable qu'au portail
        m[:, 4] = False; m[4, 4] = True
    return m, ins


def test_portail_seul_passage():
    m, ins = cloture(murs=True)
    H, W = m.shape
    rng = np.random.default_rng(1)
    dom, r, c = domaine(m, cols_mask(H, W, 0), cols_mask(H, W, W - 1), inside=ins,
                        portail_rows=[4])
    # Avec des murs, tout franchissement part de la cellule du portail : F et G coïncident
    assert ((dom.portail >= 0) == dom.franchit).all()
    for _ in range(10):
        P = rng.random(dom.n)
        sig = -np.log1p(-P)
        assert seuil_couloir(dom, P, "F")[0] == seuil_couloir(dom, P, "G")[0]
        assert exposition(dom, sig, "F", IC, 1.0)[0] == pytest.approx(
            exposition(dom, sig, "G", IC, 1.0)[0])


def test_F_inferieur_a_G():
    rng = np.random.default_rng(2)
    m, ins = cloture()
    H, W = m.shape
    dom, *_ = domaine(m, cols_mask(H, W, 0), cols_mask(H, W, W - 1), inside=ins, portail_rows=[4])
    for _ in range(20):
        P = rng.random(dom.n) * 0.9
        sig = -np.log1p(-P)
        assert seuil_couloir(dom, P, "F")[0] <= seuil_couloir(dom, P, "G")[0]
        assert exposition(dom, sig, "F", IC, 1.0)[0] <= exposition(dom, sig, "G", IC, 1.0)[0] + 1e-9


def test_profil_G_passe_par_le_portail():
    m, ins = cloture()
    H, W = m.shape
    dom, r, c = domaine(m, cols_mask(H, W, 0), cols_mask(H, W, W - 1), inside=ins, portail_rows=[4])
    P = np.full(dom.n, 0.2)
    _, path = exposition(dom, -np.log1p(-P), "G", IC, 1.0)
    side = dom.interieur[path]
    k = np.flatnonzero(side[1:] != side[:-1])
    assert k.size == 1 and r[path[k[0]]] == 4 == r[path[k[0] + 1]]


def test_detection_certaine_bloque():
    """σ = ∞ sur toute une colonne : Λ = ∞ (tout chemin est détecté à coup sûr)."""
    H, W = 4, 6
    dom, r, c = domaine(np.ones((H, W), bool), cols_mask(H, W, 0), cols_mask(H, W, W - 1))
    sig = np.where(c == 3, np.inf, 0.1)
    lam, path = exposition(dom, sig, "F", IC, 1.0)
    assert lam == np.inf and path.size == 0

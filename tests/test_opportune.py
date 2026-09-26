"""Tests de la détection opportune (src/opportune.py, formulation §9.1) sur des domaines synthétiques."""
import numpy as np
import pytest
from scipy import sparse

from src.intrus import detection, seuil_couloir
from src.opportune import (chemin_rapide, evaluer_opportune, masquer, plafond, reste_chemin,
                           temps_restant, valeur_chemin)
from test_intrus import cloture, cols_mask, domaine

IC = dict(vitesse_m_s=1.5, t_cloture_s=10.0, t_portail_s=3.0, tau_obs_s=1.0)
V = IC["vitesse_m_s"]


def ligne(W=12, fence_col=None, portail=False):
    m = np.ones((1, W), bool)
    ins = None
    if fence_col is not None:
        ins = np.zeros((1, W), bool); ins[:, fence_col:] = True
    return domaine(m, cols_mask(1, W, 0), cols_mask(1, W, W - 1), inside=ins,
                   portail_rows=[0] if portail else None)


# ---------------------------------------------------------------------------- d(v)
def test_temps_restant_ligne():
    W = 12
    dom, r, c = ligne(W)
    d = temps_restant(dom, IC, "F")
    assert d == pytest.approx((W - 1 - c) / V)
    assert d[dom.cible] == pytest.approx(0.0)


def test_temps_restant_cloture_et_portail():
    W = 12
    dom, _, c = ligne(W, fence_col=5)
    assert temps_restant(dom, IC, "F")[c == 0][0] == pytest.approx((W - 1) / V + IC["t_cloture_s"])
    dom, _, c = ligne(W, fence_col=5, portail=True)
    for pr in ("F", "G"):
        assert temps_restant(dom, IC, pr)[c == 0][0] == pytest.approx((W - 1) / V + IC["t_portail_s"])


def test_G_inatteignable_sans_portail():
    """Clôture sans portail : l'opportuniste n'atteint pas la cible, d = inf à l'extérieur."""
    dom, r, c = domaine(*cloture()[:1], cols_mask(9, 10, 0), cols_mask(9, 10, 9),
                        inside=cloture()[1])
    d = temps_restant(dom, IC, "G")
    assert np.isinf(d[~dom.interieur]).all() and np.isfinite(d[dom.interieur]).all()


def test_reste_chemin_egal_d_sur_le_plus_rapide():
    """Le long du chemin le plus rapide, le temps restant réel vaut exactement d."""
    m, ins = cloture()
    dom, *_ = domaine(m, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins, portail_rows=[4])
    for pr in ("F", "G"):
        d = temps_restant(dom, IC, pr)
        ch = chemin_rapide(dom, IC, pr)
        assert reste_chemin(dom, IC, pr, ch) == pytest.approx(d[ch])


def test_d_F_inferieur_a_d_G():
    m, ins = cloture()
    dom, *_ = domaine(m, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins, portail_rows=[4])
    assert (temps_restant(dom, IC, "F") <= temps_restant(dom, IC, "G") + 1e-12).all()


# ---------------------------------------------------------------------------- p̃*
def test_detection_tardive_inutile():
    """Ligne : une détection à 1 s de la cible ne compte plus dès que T_R > 1 s."""
    W = 12
    dom, _, c = ligne(W)
    d = temps_restant(dom, IC, "F")
    P = np.where(c == W - 2, 0.9, 0.0)                  # capteur juste avant la cible
    sigma = -np.log1p(-P)
    assert evaluer_opportune(dom, P, sigma, IC, 0.5, {"F": d, "G": d})["p_tilde_F"] == pytest.approx(0.9)
    assert evaluer_opportune(dom, P, sigma, IC, 1.0, {"F": d, "G": d})["p_tilde_F"] == 0.0
    P = np.where(c == 1, 0.9, 0.0)                      # détection précoce : utile jusqu'à d(1)
    r = evaluer_opportune(dom, P, -np.log1p(-P), IC, d[c == 1][0], {"F": d, "G": d})
    assert r["p_tilde_F"] == pytest.approx(0.9)


def instances(n=20, seed=0):
    rng = np.random.default_rng(seed)
    m, ins = cloture()
    for _ in range(n):
        mm = m & (rng.random(m.shape) > 0.1)
        mm[:, 0] = mm[:, -1] = mm[4, :] = True
        dom, *_ = domaine(mm, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins,
                          portail_rows=[4])
        P = rng.random(dom.n) * (rng.random(dom.n) < 0.6)
        d = {pr: temps_restant(dom, IC, pr) for pr in ("F", "G")}
        yield dom, P, -np.log1p(-P), d


def test_T_R_nul_redonne_p_star():
    for dom, P, sigma, d in instances():
        r = evaluer_opportune(dom, P, sigma, IC, 0.0, d)
        for pr in ("F", "G"):
            assert r[f"p_tilde_{pr}"] == pytest.approx(seuil_couloir(dom, P, pr)[0])


def test_proposition_T_R_superieur_a_T_min():
    """T_R > T_min ⇒ p̃* = 0 quel que soit le dispositif (formulation §9.1)."""
    for dom, P, sigma, d in instances():
        for pr in ("F", "G"):
            T_min = d[pr][dom.source].min()
            r = evaluer_opportune(dom, np.maximum(P, 0.99), sigma + 5, IC, T_min + 1e-6, d)
            assert r[f"p_tilde_{pr}"] == 0.0


def test_monotonie_et_encadrement():
    for dom, P, sigma, d in instances():
        for pr in ("F", "G"):
            prev = np.inf
            for T_R in np.linspace(0, d[pr][dom.source].min() + 2, 15):
                r = evaluer_opportune(dom, P, sigma, IC, T_R, d)
                assert r[f"p_tilde_{pr}"] <= prev + 1e-12
                assert r[f"p_tilde_{pr}"] <= r[f"p_tilde_ub_{pr}"] + 1e-12
                prev = r[f"p_tilde_{pr}"]


def test_masquer_et_valeur_chemin():
    P, s, d = np.array([0.5, 0.8, 0.9]), np.array([1.0, 2.0, 3.0]), np.array([5.0, 3.0, 1.0])
    Pm, sm = masquer(P, s, d, 3.0)
    assert Pm.tolist() == [0.5, 0.8, 0.0] and sm.tolist() == [1.0, 2.0, 0.0]
    ch, reste = np.array([0, 1, 2]), np.array([5.0, 3.0, 1.0])
    assert valeur_chemin(P, ch, reste, 3.0) == 0.8
    assert valeur_chemin(P, ch, reste, 6.0) == 0.0


def test_plafond_egal_detection_complete():
    rng = np.random.default_rng(1)
    D = rng.random((6, 15)) * (rng.random((6, 15)) < 0.5)
    D[2, 3] = 1.0                                        # détection certaine : σ = inf, P = 1
    A = sparse.csr_matrix(D)
    P, s = plafond(A)
    P_ref, s_ref = detection(A, np.arange(6))
    assert P == pytest.approx(P_ref) and s == pytest.approx(s_ref)

"""Tests de l'analyse de sensibilité (src/sensibilite.py, formulation §9.3)."""
import numpy as np
import pandas as pd
import pytest

from src.opportune import temps_restant
from src.sensibilite import cles, jaccard, traduire
from src.sensors import transmittance
from src.viewshed import ray_tau
from test_intrus import cloture, cols_mask, domaine

IC = dict(vitesse_m_s=1.5, t_cloture_s=10.0, t_portail_s=3.0)


def test_feuillage_equivaut_a_alpha():
    """τ est linéaire en k : k × f donne τ × f, et T(τ·f, α) = T(τ, α·f)."""
    rng = np.random.default_rng(0)
    H = W = 40
    z_op = np.zeros((H, W))
    z_veg = rng.uniform(0, 8, (H, W))
    k = rng.uniform(0, 0.4, (H, W))
    for f in (2.0, 4.0):
        for (r, c) in ((35, 30), (5, 38), (20, 2)):
            t1 = ray_tau(z_op, z_veg, k, 20, 20, 12.0, r, c, 1.0, 1.0)
            tf = ray_tau(z_op, z_veg, f * k, 20, 20, 12.0, r, c, 1.0, 1.0)
            assert tf == pytest.approx(f * t1, rel=1e-6)
            for a in (0.3, 1.0):
                assert transmittance(np.array([tf]), a) == pytest.approx(
                    transmittance(np.array([t1]), a * f))


def test_retard_plafonne_par_le_portail():
    """T_min = marche + min(t_c, t_p) : renforcer la clôture seule ne sert qu'à hauteur du portail."""
    m, ins = cloture()                       # clôture entre les colonnes 4 et 5, portail ligne 4
    dom, *_ = domaine(m, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins, portail_rows=[4])
    marche = 9 / IC["vitesse_m_s"]
    for tc, tp in ((10, 3), (30, 3), (30, 30), (60, 60)):
        d = temps_restant(dom, dict(IC, t_cloture_s=tc, t_portail_s=tp), "F")
        assert d[dom.source].min() == pytest.approx(marche + min(tc, tp))


def test_traduire_et_cles():
    conf = pd.DataFrame(dict(cand=["C1", "C1", "C2"], h=[4.0, 4.0, 6.0],
                             capteur=["cam_standard", "radar_perimetre", "cam_tele"],
                             azimut=[90.0, np.nan, 180.0], inclinaison=[12.3, 5.0, 8.0]))
    k = cles(conf)
    assert len(set(k)) == 3
    dst = {kk: i for i, kk in enumerate(k[[2, 0]])}          # la configuration 1 a disparu
    T, manq = traduire([0, 1, 2], k, dst)
    assert T.tolist() == [0, 1] and manq == 1
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3) and jaccard(set(), set()) == 1.0

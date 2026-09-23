"""Tests des modèles de capteurs : chaque test vérifie une propriété connue analytiquement."""
import numpy as np
import pytest

from src.sensors import Camera, Geometry, Radar, p_camera, p_radar, swerling1, ttpf
from src.site import load_config

CAM = Camera(name="t", n_h=1920, n_v=1080, hfov_deg=60, ppm_50=25.0, alpha=1.0)
RAD = Radar(name="r", r90_m=150.0, p_fa=1e-6, elev_max_deg=15.0, alpha=0.3)
Z = np.zeros(1)


def point(dx, dy, dz, h_rel=5.0) -> Geometry:
    a = lambda v: np.array([float(v)])
    return Geometry(dx=a(dx), dy=a(dy), dz=a(dz), h_rel=h_rel)


def axe(h_rel, az):
    a, t = np.radians(az), np.radians(CAM.tilt_deg(h_rel))
    return np.array([np.sin(a) * np.cos(t), np.cos(a) * np.cos(t), -np.sin(t)])


def test_ttpf():
    assert ttpf(1.0) == pytest.approx(0.5)
    assert np.all(np.diff(ttpf(np.linspace(0.1, 5, 50))) > 0)

def test_ttpf_stable_pour_x_grand():
    p = ttpf(np.array([10.0, 1e3, 1e6]))
    assert np.all(np.isfinite(p)) and np.allclose(p, 1.0)


def test_camera_seuil_dori_sur_axe():
    # cible sur l'axe optique à la profondeur r_design -> 25 px/m -> P = 0,5
    v = axe(5.0, 30.0) * CAM.r_design
    assert p_camera(Z, point(v[0], v[1], -v[2]), CAM, 30.0)[0] == pytest.approx(0.5, rel=1e-6)


def test_camera_champ_horizontal_et_arriere():
    f, r, D = axe(5.0, 0.0), np.array([1.0, 0, 0]), 30.0
    th = np.tan(np.radians(CAM.hfov_deg / 2))
    for frac, dedans in ((0.95, True), (1.05, False)):
        v = D * f + frac * th * D * r
        assert (p_camera(Z, point(v[0], v[1], -v[2]), CAM, 0.0)[0] > 0) == dedans
    v = -D * f                                                   # derrière la caméra
    assert p_camera(Z, point(v[0], v[1], -v[2]), CAM, 0.0)[0] == 0.0


def test_camera_vegetation_aller_simple():
    v = axe(5.0, 0.0) * CAM.r_design
    assert p_camera(np.array([1.0]), point(v[0], v[1], -v[2]), CAM, 0.0)[0] == \
        pytest.approx(0.5 * np.exp(-1.0))


def test_radar_portee_reference():
    assert p_radar(Z, point(RAD.r90_m, 0, 0), RAD)[0] == pytest.approx(0.9)


def test_radar_vegetation_aller_retour():
    attendu = swerling1(RAD.snr_90 * np.exp(-2 * RAD.alpha * 1.0), RAD.p_fa)
    assert p_radar(np.array([1.0]), point(RAD.r90_m, 0, 0), RAD)[0] == pytest.approx(attendu)


def test_radar_masque_elevation():
    assert p_radar(Z, point(3, 0, 10), RAD)[0] == 0.0           # 73° sous l'horizon

def test_radar_inclinaison():
    g_bas, g_haut = point(10, 0, 10 * np.tan(np.radians(25))), point(100, 0, -9)   # 25° sous, 5° au-dessus
    assert p_radar(Z, g_bas, RAD, tilt_deg=0.0)[0] == 0.0       # hors faisceau horizontal
    assert p_radar(Z, g_bas, RAD, tilt_deg=15.0)[0] > 0.5       # dans le faisceau incliné
    assert p_radar(Z, g_haut, RAD, tilt_deg=15.0)[0] == 0.0     # 20° hors du faisceau : masqué

def test_opaque_reste_opaque():
    tau = np.array([np.inf])
    v = axe(5.0, 0.0) * 20
    assert p_camera(tau, point(v[0], v[1], -v[2]), CAM, 0.0)[0] == 0.0
    assert p_radar(tau, point(50, 0, 0), RAD)[0] == 0.0


def test_troncature_du_disque_negligeable():
    """P_det idéale au bord du disque < 5 % : couper à radius_m ne biaise pas la couverture."""
    cfg = load_config()
    R, sc = cfg["viewshed"]["radius_m"], cfg["sensors"]
    for c in sc["cameras"]:
        cam = Camera(**c)
        assert ttpf(cam.r_design / R) < 0.05, cam.name
    for r in sc["radars"]:
        rad = Radar(**r)
        assert swerling1(rad.snr_90 * (rad.r90_m / R) ** 4, rad.p_fa) < 0.05, rad.name
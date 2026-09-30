"""Tests of the sensor models: each test checks an analytically known property."""
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


def axis(h_rel, az):
    a, t = np.radians(az), np.radians(CAM.tilt_deg(h_rel))
    return np.array([np.sin(a) * np.cos(t), np.cos(a) * np.cos(t), -np.sin(t)])


def test_ttpf():
    assert ttpf(1.0) == pytest.approx(0.5)
    assert np.all(np.diff(ttpf(np.linspace(0.1, 5, 50))) > 0)

def test_ttpf_stable_for_large_x():
    p = ttpf(np.array([10.0, 1e3, 1e6]))
    assert np.all(np.isfinite(p)) and np.allclose(p, 1.0)


def test_camera_dori_threshold_on_axis():
    # target on the optical axis at depth r_design -> 25 px/m -> P = 0.5
    v = axis(5.0, 30.0) * CAM.r_design
    assert p_camera(Z, point(v[0], v[1], -v[2]), CAM, 30.0)[0] == pytest.approx(0.5, rel=1e-6)


def test_camera_horizontal_field_and_behind():
    f, r, D = axis(5.0, 0.0), np.array([1.0, 0, 0]), 30.0
    th = np.tan(np.radians(CAM.hfov_deg / 2))
    for frac, inside in ((0.95, True), (1.05, False)):
        v = D * f + frac * th * D * r
        assert (p_camera(Z, point(v[0], v[1], -v[2]), CAM, 0.0)[0] > 0) == inside
    v = -D * f                                                   # behind the camera
    assert p_camera(Z, point(v[0], v[1], -v[2]), CAM, 0.0)[0] == 0.0


def test_camera_vegetation_one_way():
    v = axis(5.0, 0.0) * CAM.r_design
    assert p_camera(np.array([1.0]), point(v[0], v[1], -v[2]), CAM, 0.0)[0] == \
        pytest.approx(0.5 * np.exp(-1.0))


def test_radar_reference_range():
    assert p_radar(Z, point(RAD.r90_m, 0, 0), RAD)[0] == pytest.approx(0.9)


def test_radar_vegetation_two_way():
    expected = swerling1(RAD.snr_90 * np.exp(-2 * RAD.alpha * 1.0), RAD.p_fa)
    assert p_radar(np.array([1.0]), point(RAD.r90_m, 0, 0), RAD)[0] == pytest.approx(expected)


def test_radar_elevation_mask():
    assert p_radar(Z, point(3, 0, 10), RAD)[0] == 0.0           # 73° below the horizon

def test_radar_tilt():
    g_low, g_high = point(10, 0, 10 * np.tan(np.radians(25))), point(100, 0, -9)   # 25° below, 5° above
    assert p_radar(Z, g_low, RAD, tilt_deg=0.0)[0] == 0.0       # outside the horizontal beam
    assert p_radar(Z, g_low, RAD, tilt_deg=15.0)[0] > 0.5       # inside the tilted beam
    assert p_radar(Z, g_high, RAD, tilt_deg=15.0)[0] == 0.0     # 20° outside the beam: masked

def test_opaque_stays_opaque():
    tau = np.array([np.inf])
    v = axis(5.0, 0.0) * 20
    assert p_camera(tau, point(v[0], v[1], -v[2]), CAM, 0.0)[0] == 0.0
    assert p_radar(tau, point(50, 0, 0), RAD)[0] == 0.0


def test_disc_truncation_negligible():
    """Ideal P_det at the edge of the disc < 5 %: cutting at radius_m does not bias the coverage."""
    cfg = load_config()
    R, sc = cfg["viewshed"]["radius_m"], cfg["sensors"]
    for c in sc["cameras"]:
        cam = Camera(**c)
        assert ttpf(cam.r_design / R) < 0.05, cam.name
    for r in sc["radars"]:
        rad = Radar(**r)
        assert swerling1(rad.snr_90 * (rad.r90_m / R) ** 4, rad.p_fa) < 0.05, rad.name


def test_cameras_only(tmp_path):
    """sensors.cameras_only: no radar reaches the chain, the cameras are unchanged."""
    import yaml

    from src.sensors import load_sensors
    cfg = yaml.safe_load(open(load_config.__defaults__[0], encoding="utf-8"))
    cfg["sensors"]["cameras_only"] = True
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    c = load_config(tmp_path / "config.yaml")
    cams, radars, _, _ = load_sensors(c)
    assert radars == [] and len(cams) == len(cfg["sensors"]["cameras"])

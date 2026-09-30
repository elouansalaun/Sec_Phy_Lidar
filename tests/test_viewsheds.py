"""
Analytical tests of the line-of-sight kernel.
Each test isolates one term of the model on a synthetic terrain with a known solution.

Usage : python -m pytest tests -v
"""
import numpy as np
import pytest

from src.viewshed import optical_depth

N, R0, C0, RES, STEP, RADIUS = 121, 60, 60, 1.0, 0.5, 60


def flat(v: float = 0.0) -> np.ndarray:
    return np.full((N, N), v, np.float64)


def los(z_opaque, z_veg, k, z_tgt, h):
    return optical_depth(z_opaque, z_veg, k, z_tgt, R0, C0, h,
                         RADIUS, RES).astype(np.float64)


# -----------------------------------------------------------------------------
# 1. Geometry of the disc and flat terrain: everything visible, nothing attenuates
# -----------------------------------------------------------------------------
def test_flat_terrain():
    tau = los(flat(), flat(), flat(), flat(1.0), h=5.0)
    rr, cc = np.indices(tau.shape)
    inside = np.hypot(rr - R0, cc - C0) <= RADIUS
    assert np.all(np.isnan(tau[~inside]))        # out of range = NaN
    assert np.all(tau[inside] == 0.0)            # no obstacle, no vegetation


# -----------------------------------------------------------------------------
# 2. Occlusion: shadow of a wall, length given by the intercept theorem
#    Ray observer (h) -> target (t) at distance D passes above the wall (Hw, at Dw)
#    iff h + (t - h)·Dw/D ≥ Hw  <=>  D ≥ D* = Dw·(h - t)/(h - Hw)
# -----------------------------------------------------------------------------
def test_wall_shadow():
    h, t, Hw, Dw = 10.0, 1.0, 5.0, 20
    z = flat()
    z[:, C0 + Dw] = Hw
    row = los(z, z, flat(), flat(t), h)[R0, C0:]           # row[D] = target at D m
    d_star = Dw * (h - t) / (h - Hw)                         # 36 m
    assert np.all(row[1:Dw] == 0.0)                          # in front of the wall
    assert np.all(np.isinf(row[Dw + 1:int(d_star) - 1]))     # in the shadow
    assert np.all(row[int(d_star) + 1:RADIUS + 1] == 0.0)    # beyond the shadow


# -----------------------------------------------------------------------------
# 3. Beer-Lambert in an infinite slab: τ = k × 3D length
#    Observer and target pixels excluded -> horizontal length D - 1
# -----------------------------------------------------------------------------
def test_vegetation_slab():
    h, t, k = 3.0, 1.0, 0.1
    tau = los(flat(), flat(50.0), flat(k), flat(t), h)
    for D in (5, 20, 50):                                    # axis: D - 1 full cells
        expected = k * (D - 1) * np.sqrt(1 + ((t - h) / D) ** 2)
        assert tau[R0, C0 + D] == pytest.approx(expected, rel=1e-5)
    n = 30                                                   # diagonal: n - 1 cells
    expected = k * (n - 1) / n * np.sqrt(2 * n**2 + (t - h) ** 2)
    assert tau[R0 + n, C0 + n] == pytest.approx(expected, rel=1e-5)


# -----------------------------------------------------------------------------
# 4. Partial crossing: observer above the canopy
#    The ray is inside the layer (z < Hv) only over the fraction (Hv - t)/(h - t)
#    of the path, on the target side
# -----------------------------------------------------------------------------
def test_partial_slab():
    h, t, k, Hv, D = 21.0, 1.0, 0.1, 5.0, 40
    tau = los(flat(), flat(Hv), flat(k), flat(t), h)
    t_entry = (h - Hv) / (h - t)                             # the ray goes below Hv
    t_target = (D - 0.5) / D                                 # edge of the target cell
    expected = k * (t_target - t_entry) * np.sqrt(D**2 + (h - t) ** 2)
    assert tau[R0, C0 + D] == pytest.approx(expected, rel=1e-5)


# -----------------------------------------------------------------------------
# 5. Convention: the mast pixel itself does not occlude
# -----------------------------------------------------------------------------
def test_mast_pixel_ignored():
    z = flat()
    z[R0, C0] = 100.0
    tau = los(z, flat(), flat(), flat(1.0), h=5.0)
    assert np.nanmax(tau) == 0.0

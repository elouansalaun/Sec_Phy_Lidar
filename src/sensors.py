"""
src/sensors.py: sensor models: from the optical depth τ to the detection probability.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import geopandas as gpd
import numpy as np

from src.rasters import write_raster
from src.site import ROOT, load_config
from src.viewshed import compute_tau, prepare_surfaces, to_pixel


# =============================================================================
# Step 1: sensor geometry -> cells
# =============================================================================
@dataclass
class Geometry:
    dx: np.ndarray        # sensor -> target vector, east component [m]
    dy: np.ndarray        # north component [m]
    dz: np.ndarray        # z_sensor - z_target [m] (> 0: target lower)
    h_rel: float          # sensor height above the target at the foot of the mast [m]

    @property
    def slant(self) -> np.ndarray:
        return np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

    @property
    def depression(self) -> np.ndarray:           # angle below the horizontal [°]
        return np.degrees(np.arctan2(self.dz, np.hypot(self.dx, self.dy)))


def geometry(s, r0: int, c0: int, z_obs: float) -> Geometry:
    res = s.grid.res
    rr, cc = np.indices(s.z_tgt.shape)
    dz = z_obs - s.z_tgt
    return Geometry(dx=(cc - c0) * res, dy=(r0 - rr) * res, dz=dz, h_rel=float(dz[r0, c0]))


def transmittance(tau, alpha: float, passes: int = 1) -> np.ndarray:
    """exp(-passes·α·τ). Opaque (τ = inf) and out of range (NaN) -> 0, whatever α."""
    t = np.asarray(tau, np.float64)
    blocked = ~np.isfinite(t)
    return np.where(blocked, 0.0, np.exp(-passes * alpha * np.where(blocked, 0.0, t)))


# =============================================================================
# Step 2: detection laws
# =============================================================================
def ttpf(x) -> np.ndarray:
    """Target Transfer Probability Function (Johnson criteria): P(x), x = N / N_50.
    Logistic form in ln x: identical to x^E/(1+x^E), without overflow for large x."""
    x = np.clip(np.asarray(x, np.float64), 1e-9, None)
    e = 2.7 + 0.7 * x
    return 1.0 / (1.0 + np.exp(-e * np.log(x)))


def swerling1(snr, p_fa: float) -> np.ndarray:
    """P_d of a Swerling I target, single pulse, square-law detection."""
    return p_fa ** (1.0 / (1.0 + np.asarray(snr, np.float64)))


# =============================================================================
# Step 3: camera (pinhole, EN 62676-4)
# =============================================================================
@dataclass
class Camera:
    name: str
    n_h: int
    n_v: int
    hfov_deg: float
    ppm_50: float
    alpha: float
    cost: float = 1.0

    @property
    def vfov_deg(self) -> float:
        return 2 * np.degrees(np.arctan(np.tan(np.radians(self.hfov_deg / 2)) * self.n_v / self.n_h))

    @property
    def r_design(self) -> float:
        """Depth at which the pixel density equals ppm_50 (detection limit)."""
        return self.n_h / (2 * self.ppm_50 * np.tan(np.radians(self.hfov_deg / 2)))

    def tilt_deg(self, h_rel: float) -> float:
        """Tilt of the axis: top edge of the image on the ground at r_design."""
        return np.degrees(np.arctan2(h_rel, self.r_design)) + self.vfov_deg / 2


def p_camera(tau, g: Geometry, cam: Camera, az0_deg: float) -> np.ndarray:
    a, t = np.radians(az0_deg), np.radians(cam.tilt_deg(g.h_rel))
    f = (np.sin(a) * np.cos(t), np.cos(a) * np.cos(t), -np.sin(t))   # optical axis
    r = (np.cos(a), -np.sin(a), 0.0)                                   # image right
    u = (np.sin(a) * np.sin(t), np.cos(a) * np.sin(t), np.cos(t))     # image up
    vx, vy, vz = g.dx, g.dy, -g.dz
    th = np.tan(np.radians(cam.hfov_deg / 2))
    tv = np.tan(np.radians(cam.vfov_deg / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = vx * f[0] + vy * f[1] + vz * f[2]
        x = (vx * r[0] + vy * r[1]) / depth
        y = (vx * u[0] + vy * u[1] + vz * u[2]) / depth
        in_fov = (depth > 0) & (np.abs(x) <= th) & (np.abs(y) <= tv)
        p = ttpf(cam.n_h / (2 * depth * th) / cam.ppm_50)
    return np.where(in_fov, transmittance(tau, cam.alpha) * p, 0.0)


# =============================================================================
# Step 4: radar (radar equation, Swerling I, elevation mask)
# =============================================================================
@dataclass
class Radar:
    name: str
    r90_m: float
    p_fa: float
    elev_max_deg: float
    alpha: float
    cost: float = 1.0

    @property
    def snr_90(self) -> float:
        """SNR giving P_d = 0.9: P_fa^(1/(1+SNR)) = 0.9."""
        return np.log(self.p_fa) / np.log(0.9) - 1.0


def p_radar(tau, g: Geometry, rad: Radar, tilt_deg: float = 0.0) -> np.ndarray:
    t2 = transmittance(tau, rad.alpha, passes=2)              # two way
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = rad.snr_90 * (rad.r90_m / g.slant) ** 4 * t2
        ok = (np.abs(g.depression - tilt_deg) <= rad.elev_max_deg) & (t2 > 0)
        p = swerling1(snr, rad.p_fa)
    return np.where(ok, p, 0.0)


# =============================================================================
# Step 5: demo: effective detection area of a candidate
# =============================================================================
def load_sensors(cfg: dict):
    sc = cfg["sensors"]
    return ([Camera(**c) for c in sc["cameras"]],
            [Radar(**r) for r in sc["radars"]],
            sc["camera_azimuths_deg"],
            sc["radar_tilts_deg"])


def main() -> None:
    cand_id, h = sys.argv[1], float(sys.argv[2])
    cfg = load_config()
    s = prepare_surfaces(cfg)
    c = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id").loc[cand_id]
    z_obs = c["z_support"] + h
    r0, c0 = to_pixel(s.grid, c.geometry.x, c.geometry.y)
    tau = compute_tau(s, c.geometry.x, c.geometry.y, z_obs, cfg)
    tau_free = np.where(np.isnan(tau), np.nan, 0.0)           # same disc, no obstacle
    g = geometry(s, r0, c0, z_obs)
    ground = s.z_opaque < s.z_tgt
    a_cell = s.grid.res ** 2
    out = ROOT / cfg["viewshed"]["out_dir"]
    cams, radars, azs, tilts = load_sensors(cfg)

    A = lambda p: float(np.sum(p[ground]) * a_cell)
    print(f"{cand_id} ({c['support']}) at {h:.0f} m: A_eff = Σ P·a on the ground [m²]; "
          f"yield = A_eff / A_eff without obstacles\n")
    for cam in cams:
        res = [(az, A(p_camera(tau, g, cam, az)), A(p_camera(tau_free, g, cam, az))) for az in azs]
        az, a_eff, a_free = max(res, key=lambda x: x[1])
        print(f"  {cam.name:16s} r_design {cam.r_design:4.0f} m, tilt {cam.tilt_deg(g.h_rel):4.1f}° | "
              + " ".join(f"{a:5.0f}" for _, a, _ in res)
              + f" | max {a_eff:5.0f} m² at {az:3d}°, yield {a_eff / a_free:.2f}")
        write_raster(out / f"P_{cand_id}_{h:.0f}m_{cam.name}_{az}.tif",
                     p_camera(tau, g, cam, az), s.grid)
    for rad in radars:
        p, p_free = p_radar(tau, g, rad), p_radar(tau_free, g, rad)
        print(f"  {rad.name:16s} R90 {rad.r90_m:4.0f} m, mask ±{rad.elev_max_deg:.0f}°"
              f"{'':>58s}| A_eff {A(p):5.0f} m², yield {A(p) / A(p_free):.2f}")
        write_raster(out / f"P_{cand_id}_{h:.0f}m_{rad.name}.tif", p, s.grid)


if __name__ == "__main__":
    main()

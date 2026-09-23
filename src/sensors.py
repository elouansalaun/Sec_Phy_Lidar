"""
src/sensors.py — Modèles de capteurs : de l'épaisseur optique τ à la probabilité de détection.

P_det(x) = T(x) · P_capteur(géométrie)
  caméra : T = exp(-α_c τ) (aller simple) ; sténopé exact ; P = TTPF(ppm / ppm_50), EN 62676-4
  radar  : SNR = SNR_90 (R_90/R)^4 exp(-2 α_r τ) (aller-retour) ; P = P_fa^(1/(1+SNR)), Swerling I

Usage  : python -m src.sensors C052 4
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
# Étape 1 — Géométrie capteur -> cellules
# =============================================================================
@dataclass
class Geometry:
    dx: np.ndarray        # vecteur capteur -> cible, composante est [m]
    dy: np.ndarray        # composante nord [m]
    dz: np.ndarray        # z_capteur - z_cible [m] (> 0 : cible plus basse)
    h_rel: float          # hauteur du capteur au-dessus de la cible au pied du mât [m]

    @property
    def slant(self) -> np.ndarray:
        return np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

    @property
    def depression(self) -> np.ndarray:           # angle sous l'horizontale [°]
        return np.degrees(np.arctan2(self.dz, np.hypot(self.dx, self.dy)))


def geometry(s, r0: int, c0: int, z_obs: float) -> Geometry:
    res = s.grid.res
    rr, cc = np.indices(s.z_tgt.shape)
    dz = z_obs - s.z_tgt
    return Geometry(dx=(cc - c0) * res, dy=(r0 - rr) * res, dz=dz, h_rel=float(dz[r0, c0]))


def transmittance(tau, alpha: float, passes: int = 1) -> np.ndarray:
    """exp(-passes·α·τ). Opaque (τ = inf) et hors portée (NaN) -> 0, quel que soit α."""
    t = np.asarray(tau, np.float64)
    blocked = ~np.isfinite(t)
    return np.where(blocked, 0.0, np.exp(-passes * alpha * np.where(blocked, 0.0, t)))


# =============================================================================
# Étape 2 — Lois de détection
# =============================================================================
def ttpf(x) -> np.ndarray:
    """Target Transfer Probability Function (critères de Johnson) : P(x), x = N / N_50.
    Forme logistique en ln x : identique à x^E/(1+x^E), sans débordement pour x grand."""
    x = np.clip(np.asarray(x, np.float64), 1e-9, None)
    e = 2.7 + 0.7 * x
    return 1.0 / (1.0 + np.exp(-e * np.log(x)))


def swerling1(snr, p_fa: float) -> np.ndarray:
    """P_d d'une cible Swerling I, impulsion unique, détection quadratique."""
    return p_fa ** (1.0 / (1.0 + np.asarray(snr, np.float64)))


# =============================================================================
# Étape 3 — Caméra (sténopé, EN 62676-4)
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
        """Profondeur à laquelle la densité vaut ppm_50 (limite de détection)."""
        return self.n_h / (2 * self.ppm_50 * np.tan(np.radians(self.hfov_deg / 2)))

    def tilt_deg(self, h_rel: float) -> float:
        """Inclinaison de l'axe : bord haut de l'image au sol à r_design."""
        return np.degrees(np.arctan2(h_rel, self.r_design)) + self.vfov_deg / 2


def p_camera(tau, g: Geometry, cam: Camera, az0_deg: float) -> np.ndarray:
    a, t = np.radians(az0_deg), np.radians(cam.tilt_deg(g.h_rel))
    f = (np.sin(a) * np.cos(t), np.cos(a) * np.cos(t), -np.sin(t))   # axe optique
    r = (np.cos(a), -np.sin(a), 0.0)                                   # droite de l'image
    u = (np.sin(a) * np.sin(t), np.cos(a) * np.sin(t), np.cos(t))     # haut de l'image
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
# Étape 4 — Radar (équation radar, Swerling I, masque d'élévation)
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
        """SNR donnant P_d = 0,9 : P_fa^(1/(1+SNR)) = 0,9."""
        return np.log(self.p_fa) / np.log(0.9) - 1.0


def p_radar(tau, g: Geometry, rad: Radar, tilt_deg: float = 0.0) -> np.ndarray:
    t2 = transmittance(tau, rad.alpha, passes=2)              # aller-retour
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = rad.snr_90 * (rad.r90_m / g.slant) ** 4 * t2
        ok = (np.abs(g.depression - tilt_deg) <= rad.elev_max_deg) & (t2 > 0)
        p = swerling1(snr, rad.p_fa)
    return np.where(ok, p, 0.0)


# =============================================================================
# Étape 5 — Démo : surface efficace de détection d'un candidat
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
    tau_libre = np.where(np.isnan(tau), np.nan, 0.0)          # même disque, sans obstacle
    g = geometry(s, r0, c0, z_obs)
    sol = s.z_opaque < s.z_tgt
    a_cell = s.grid.res ** 2
    out = ROOT / cfg["viewshed"]["out_dir"]
    cams, radars, azs, tilts = load_sensors(cfg)

    A = lambda p: float(np.sum(p[sol]) * a_cell)
    print(f"{cand_id} ({c['support']}) à {h:.0f} m — A_eff = Σ P·a sur le sol [m²] ; "
          f"rendement = A_eff / A_eff sans obstacle\n")
    for cam in cams:
        res = [(az, A(p_camera(tau, g, cam, az)), A(p_camera(tau_libre, g, cam, az))) for az in azs]
        az, a_eff, a_lib = max(res, key=lambda x: x[1])
        print(f"  {cam.name:16s} r_design {cam.r_design:4.0f} m, tilt {cam.tilt_deg(g.h_rel):4.1f}° | "
              + " ".join(f"{a:5.0f}" for _, a, _ in res)
              + f" | max {a_eff:5.0f} m² à {az:3d}°, rendement {a_eff / a_lib:.2f}")
        write_raster(out / f"P_{cand_id}_{h:.0f}m_{cam.name}_{az}.tif",
                     p_camera(tau, g, cam, az), s.grid)
    for rad in radars:
        p, p_lib = p_radar(tau, g, rad), p_radar(tau_libre, g, rad)
        print(f"  {rad.name:16s} R90 {rad.r90_m:4.0f} m, masque ±{rad.elev_max_deg:.0f}°"
              f"{'':>58s}| A_eff {A(p):5.0f} m², rendement {A(p) / A(p_lib):.2f}")
        write_raster(out / f"P_{cand_id}_{h:.0f}m_{rad.name}.tif", p, s.grid)


if __name__ == "__main__":
    main()
"""
Tests analytiques du noyau de ligne de visée.
Chaque test isole un terme du modèle sur un terrain synthétique à solution connue.

Usage : python -m pytest tests -v
"""
import numpy as np
import pytest

from src.viewshield import optical_depth      

N, R0, C0, RES, STEP, RADIUS = 121, 60, 60, 1.0, 0.5, 60


def flat(v: float = 0.0) -> np.ndarray:
    return np.full((N, N), v, np.float64)


def los(z_opaque, z_veg, k, z_tgt, h):
    return optical_depth(z_opaque, z_veg, k, z_tgt, R0, C0, h,
                         RADIUS, RES, STEP).astype(np.float64)


# -----------------------------------------------------------------------------
# 1. Géométrie du disque et terrain plat : tout est visible, rien n'atténue
# -----------------------------------------------------------------------------
def test_terrain_plat():
    tau = los(flat(), flat(), flat(), flat(1.0), h=5.0)
    rr, cc = np.indices(tau.shape)
    dans = np.hypot(rr - R0, cc - C0) <= RADIUS
    assert np.all(np.isnan(tau[~dans]))          # hors portée = NaN
    assert np.all(tau[dans] == 0.0)              # aucun obstacle, aucune végétation


# -----------------------------------------------------------------------------
# 2. Occultation : ombre d'un mur, longueur donnée par Thalès
#    Rayon observateur (h) -> cible (t) à distance D passe au-dessus du mur (Hw, à Dw)
#    ssi h + (t - h)·Dw/D ≥ Hw  <=>  D ≥ D* = Dw·(h - t)/(h - Hw)
# -----------------------------------------------------------------------------
def test_ombre_mur():
    h, t, Hw, Dw = 10.0, 1.0, 5.0, 20
    z = flat()
    z[:, C0 + Dw] = Hw
    row = los(z, z, flat(), flat(t), h)[R0, C0:]           # row[D] = cible à D m
    d_star = Dw * (h - t) / (h - Hw)                         # 36 m
    assert np.all(row[1:Dw] == 0.0)                          # devant le mur
    assert np.all(np.isinf(row[Dw + 1:int(d_star) - 1]))     # dans l'ombre
    assert np.all(row[int(d_star) + 1:RADIUS + 1] == 0.0)    # au-delà de l'ombre


# -----------------------------------------------------------------------------
# 3. Beer-Lambert dans une dalle infinie : τ = k × longueur 3D
#    Pixels observateur et cible exclus -> longueur horizontale D - 1
# -----------------------------------------------------------------------------
def test_dalle_vegetation():
    h, t, k = 3.0, 1.0, 0.1
    tau = los(flat(), flat(50.0), flat(k), flat(t), h)
    for D in (5, 20, 50):                                    # axe des colonnes : exact
        attendu = k * (D - 1) * np.sqrt(1 + ((t - h) / D) ** 2)
        assert tau[R0, C0 + D] == pytest.approx(attendu, rel=1e-5)
    n = 30                                                   # diagonale : isotropie
    d = n * np.sqrt(2)
    attendu = k * (d - 1) * np.sqrt(1 + ((t - h) / d) ** 2)
    assert tau[R0 + n, C0 + n] == pytest.approx(attendu, abs=1.5 * k)


# -----------------------------------------------------------------------------
# 4. Traversée partielle : observateur au-dessus de la canopée
#    Le rayon n'est dans la couche (z < Hv) que sur la fraction (Hv - t)/(h - t)
#    du trajet, côté cible
# -----------------------------------------------------------------------------
def test_dalle_partielle():
    h, t, k, Hv, D = 21.0, 1.0, 0.1, 5.0, 40
    tau = los(flat(), flat(Hv), flat(k), flat(t), h)
    horiz = D * (Hv - t) / (h - t) - 1                       # 8 m dans la couche - pixel cible
    attendu = k * horiz * np.sqrt(1 + ((t - h) / D) ** 2)
    assert tau[R0, C0 + D] == pytest.approx(attendu, abs=0.5 * k)


# -----------------------------------------------------------------------------
# 5. Convention : le pixel du mât lui-même n'occulte pas
# -----------------------------------------------------------------------------
def test_pixel_mat_ignore():
    z = flat()
    z[R0, C0] = 100.0
    tau = los(z, flat(), flat(), flat(1.0), h=5.0)
    assert np.nanmax(tau) == 0.0
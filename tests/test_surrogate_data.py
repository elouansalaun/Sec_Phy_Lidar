"""
Tests du jeu de données du modèle de substitution (src/dataset_surrogate.py).

1. Cohérence : le noyau `potentiel` redonne exactement l'agrégation de `optical_depth`.
2. Terrain plat, sans végétation : tout le sol est visible, V = 1 dans chaque bin.
3. Symétries du groupe D4 : tourner ou retourner le terrain permute les secteurs.
   C'est ce qui autorise l'augmentation de données par rotations en 5.3.
4. Monotonie : V ne décroît pas quand l'observateur monte.
5. Découpage : le site est dans le test, le tampon est respecté, les disques sont complets.

Usage : python -m pytest tests/test_surrogate_data.py -v
"""
import numpy as np
import pytest

from src.dataset_surrogate import SPLITS, EXCLU, bins_disque, decoupage, potentiel
from src.viewshed import optical_depth

N, RES, R = 81, 1.0, 30
ANNEAUX, NS = [0, 10, 30], 8
H = np.array([2.0, 5.0, 12.0])
DR, DC, B, NBIN = bins_disque(R, ANNEAUX, NS, RES)


def terrain(seed=0, veg=True):
    """Relief doux, blocs opaques, taches de végétation : un cas où tout intervient."""
    rng = np.random.default_rng(seed)
    rr, cc = np.indices((N, N))
    z = 2.0 * np.sin(rr / 9.0) + 1.5 * np.cos(cc / 7.0)
    zo = z.copy()
    for _ in range(12):
        r, c, a, b = rng.integers(0, N - 6, 4) % (N - 6)
        zo[r:r + 3 + a % 4, c:c + 3 + b % 4] += rng.uniform(3, 9)
    zv = zo.copy()
    k = np.zeros_like(z)
    if veg:
        m = (rng.random((N, N)) < 0.15) & (zo == z)
        zv[m] += rng.uniform(1, 6, m.sum())
        k[m] = rng.uniform(0.05, 0.8, m.sum())
    return zo, zv, k, z + 1.0


def run(zo, zv, k, zt, obs):
    sol = zo < zt
    r = np.array([o[0] for o in obs], np.int64)
    c = np.array([o[1] for o in obs], np.int64)
    return potentiel(zo, zv, k, zt, sol, r, c, H, DR, DC, B, NBIN, RES).astype(np.float64)


def agreger(zo, zv, k, zt, r0, c0, h):
    """Référence : disque de optical_depth, puis somme de T par bin, en numpy."""
    tau = optical_depth(zo, zv, k, zt, r0, c0, zo[r0, c0] + h, R, RES).astype(np.float64)
    sol = zo < zt
    acc = np.zeros(NBIN.size)
    for dr, dc, b in zip(DR, DC, B):
        t = tau[r0 + dr, c0 + dc]
        if sol[r0 + dr, c0 + dc] and np.isfinite(t):
            acc[b] += np.exp(-t)
    return acc / NBIN


# -----------------------------------------------------------------------------
# 1. Cohérence avec le moteur de visibilité
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coherence_optical_depth(seed):
    zo, zv, k, zt = terrain(seed)
    obs = [(40, 40), (35, 47), (45, 33)]
    V = run(zo, zv, k, zt, obs)
    for i, (r0, c0) in enumerate(obs):
        for ih, h in enumerate(H):
            np.testing.assert_allclose(V[i, ih], agreger(zo, zv, k, zt, r0, c0, h), atol=1e-6)


# -----------------------------------------------------------------------------
# 2. Terrain plat : tout est visible
# -----------------------------------------------------------------------------
def test_plat_tout_visible():
    z = np.zeros((N, N))
    V = run(z, z, z, z + 1.0, [(40, 40)])
    np.testing.assert_allclose(V, 1.0, atol=1e-6)
    assert NBIN.sum() == DR.size                      # chaque cellule du disque dans un bin


def test_bins_geometrie():
    # Répartition quasi uniforme entre secteurs, et les deux anneaux sont remplis
    n = NBIN.reshape(len(ANNEAUX) - 1, NS)
    assert (n > 0).all()
    assert n.max(axis=1).min() / n.min(axis=1).max() < 1.1
    # Le nord (−r) est le secteur 0, l'est (+c) le secteur 2
    i_n = np.flatnonzero((DR == -20) & (DC == 0))[0]
    i_e = np.flatnonzero((DR == 0) & (DC == 20))[0]
    assert B[i_n] % NS == 0 and B[i_e] % NS == 2


# -----------------------------------------------------------------------------
# 3. Symétries D4 : rotation de 90° et miroir
# -----------------------------------------------------------------------------
def permute(V, perm):
    """V[..., anneau·NS + s] -> secteur perm[s]."""
    Vr = V.reshape(*V.shape[:-1], len(ANNEAUX) - 1, NS)
    out = np.empty_like(Vr)
    out[..., perm] = Vr
    return out.reshape(V.shape)


def test_rotation_90():
    zo, zv, k, zt = terrain(3)
    r0, c0 = 38, 44
    V = run(zo, zv, k, zt, [(r0, c0)])
    rot = lambda a: np.rot90(a, k=-1).copy()          # sens horaire : nord -> est
    # (r, c) -> (c, N-1-r) pour une rotation horaire
    Vr = run(rot(zo), rot(zv), rot(k), rot(zt), [(c0, N - 1 - r0)])
    np.testing.assert_allclose(Vr, permute(V, (np.arange(NS) + 2) % NS), atol=1e-5)


def test_miroir():
    zo, zv, k, zt = terrain(4)
    r0, c0 = 42, 37
    V = run(zo, zv, k, zt, [(r0, c0)])
    fl = lambda a: a[:, ::-1].copy()                  # est <-> ouest : azimut -> −azimut
    Vm = run(fl(zo), fl(zv), fl(k), fl(zt), [(r0, N - 1 - c0)])
    np.testing.assert_allclose(Vm, permute(V, (NS - np.arange(NS)) % NS), atol=1e-5)


# -----------------------------------------------------------------------------
# 4. Monotonie en hauteur
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [5, 6])
def test_monotone_hauteur_opaque(seed):
    """Sans végétation, le rayon monte en tout point quand z_obs monte : V est non décroissant."""
    zo, zv, k, zt = terrain(seed, veg=False)
    V = run(zo, zv, k, zt, [(40, 40), (33, 50)])
    assert np.all(np.diff(V, axis=1) >= -1e-6)


def test_vegetation_brise_la_monotonie():
    """Avec végétation, la monotonie n'est PAS garantie : un rayon entièrement sous la canopée
    y parcourt √(1 + pente²) m par mètre horizontal, et la pente croît avec h (§8.2).
    Le réseau ne doit donc pas être contraint à être monotone en h."""
    z = np.zeros((N, N))
    zv, k = z.copy(), z.copy()
    zv[40, 45:48] = 30.0                              # colonne de canopée haute, traversée de part en part
    k[40, 45:48] = 0.5
    zt = z + 1.0
    V = run(z, zv, k, zt, [(40, 40)])
    b = B[np.flatnonzero((DR == 0) & (DC == 10))[0]]
    assert V[0, 1, b] < V[0, 0, b]                     # h = 5 m voit moins bien que h = 2 m


# -----------------------------------------------------------------------------
# 5. Découpage spatial
# -----------------------------------------------------------------------------
class _G:
    xmin, ymax, res = 0.0, 400.0, 1.0


class _S:
    z_tgt = np.zeros((400, 400))
    grid = _G()


def test_decoupage():
    cfg = {"project": {"random_seed": 0},
           "surrogate": {"rayon_m": 60, "test": {"marge_m": 20, "tampon_m": 15},
                         "validation": {"tuile_m": 50, "part": 0.2}}}
    split = decoupage(_S(), cfg, (180.0, 180.0, 220.0, 220.0))   # site de 40 m au centre
    r, c = np.nonzero(split != EXCLU)
    assert r.min() >= 60 and r.max() < 340 and c.min() >= 60 and c.max() < 340   # disque complet
    # le site est entièrement en test
    assert (split[180:220, 180:220] == SPLITS["test"]).all()
    # aucune cellule d'entraînement ou de validation à moins de 15 m du bloc test
    tr, tc = np.nonzero(split == SPLITS["test"])
    a, b = np.nonzero(split <= SPLITS["validation"])
    dr = np.maximum(np.maximum(tr.min() - a, a - tr.max()), 0)
    dc = np.maximum(np.maximum(tc.min() - b, b - tc.max()), 0)
    assert np.maximum(dr, dc).min() > 15
    assert (split == SPLITS["validation"]).any()

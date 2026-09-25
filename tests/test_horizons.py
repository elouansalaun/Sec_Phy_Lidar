"""
Tests des horizons denses (src/horizons.py) et de l'augmentation du réseau physique.

1. Blocs de 1 px : la pyramide redonne EXACTEMENT les horizons de B1, à chaque hauteur.
2. Pavage : exact dans chaque bande ; ≥ 90 % du disque couvert une fois (frontières de bandes).
3. Équivariance D4 : tourner le terrain tourne les cartes d'horizons et permute les secteurs,
   exactement comme les étiquettes (`transformer` = `surrogate.d4`).

Usage : python -m pytest tests/test_horizons.py -v
"""
import numpy as np
import pytest

from src.baselines_surrogate import _descripteurs
from src.horizons import Horizons, reseau_blocs
from src.surrogate import d4
from src.surrogate_physique import transformer

N, NS = 101, 8
BANDES = [0, 5, 12, 25, 40]
H = [2.0, 7.0]


class _G:
    res = 1.0


class _S:
    def __init__(self, zo, zv, k, zt):
        self.z_opaque, self.z_veg, self.k_ext, self.z_tgt, self.grid = zo, zv, k, zt, _G()


def cfg(pas):
    return {"surrogate": {"horizons": {"bandes_m": BANDES, "pas_px": pas},
                          "n_secteurs": NS, "hauteurs_m": H}}


def terrain(seed=0):
    rng = np.random.default_rng(seed)
    zo = rng.uniform(0, 1.5, (N, N))
    zo[rng.random((N, N)) < 0.04] += rng.uniform(3, 10)
    zv = zo + np.where(rng.random((N, N)) < 0.1, rng.uniform(1, 5, (N, N)), 0.0)
    k = np.where(zv > zo, 0.3, 0.0)
    return _S(zo, zv, k, zo + 1.0)


def exact_b1(s, r, c):
    """Horizons exacts (tangentes) par la même logique que B1, sur les mêmes bandes."""
    R = BANDES[-1]
    dr, dc = np.mgrid[-R:R + 1, -R:R + 1]
    dr, dc = dr.ravel(), dc.ravel()
    d = np.hypot(dr, dc)
    keep = (dr * dr + dc * dc <= R * R) & (d > 0)
    dr, dc, d = dr[keep], dc[keep], d[keep]
    az = np.degrees(np.arctan2(dc, -dr)) % 360
    sect = (np.floor((az + 22.5) / 45).astype(np.int64)) % NS
    band = np.searchsorted(np.asarray(BANDES[1:-1], float), d, side="left")
    E = _descripteurs(s.z_opaque, s.z_veg, s.k_ext, s.z_opaque < s.z_tgt, r, c,
                      np.asarray(H), dr.astype(np.int64), dc.astype(np.int64), sect, band, d,
                      NS, len(BANDES) - 1)
    return E.reshape(r.size, len(H), NS, len(BANDES) - 1, 4)   # (n, h, s, b, desc)


def test_blocs_unitaires_egalent_b1():
    s = terrain(0)
    r = np.array([50, 45, 60], np.int64)
    c = np.array([50, 58, 47], np.int64)
    hor, moy = Horizons(s, cfg([1, 1, 1, 1]))(r, c)             # (n, h, 2, b, s)
    E = exact_b1(s, r, c)
    np.testing.assert_allclose(hor[:, :, 0], np.arctan(E[..., 0]).transpose(0, 1, 3, 2), atol=1e-5)
    np.testing.assert_allclose(hor[:, :, 1], np.arctan(E[..., 1]).transpose(0, 1, 3, 2), atol=1e-5)
    np.testing.assert_allclose(moy[:, 1], E[:, 0, :, :, 3].transpose(0, 2, 1), atol=1e-5)  # sol


def couverture(bandes, pas, M):
    dr, dc, lvl, *_ = reseau_blocs(bandes, pas, NS, 1.0)
    tot = np.zeros((2 * M + 1,) * 2, int)
    par_bande = [np.zeros_like(tot) for _ in pas]
    for r, c, L in zip(dr, dc, lvl):
        p = pas[L] // 2
        sl = (slice(M + r - p, M + r + p + 1), slice(M + c - p, M + c + p + 1))
        tot[sl] += 1
        par_bande[L][sl] += 1
    return tot, par_bande


@pytest.mark.parametrize("pas", [[1, 3, 3, 5], [1, 3, 5, 9]])
def test_pavage_dans_chaque_bande(pas):
    """Dans une bande, les blocs ne se recouvrent jamais (réseau de pas p, blocs de côté p)."""
    _, par_bande = couverture(BANDES, pas, 60)
    assert all(pb.max() == 1 for pb in par_bande)


def test_couverture_globale():
    """Configuration par défaut : ≥ 90 % du disque couvert exactement une fois (le reste, aux
    frontières de bandes, est chevauché ou non couvert : approximation documentée)."""
    tot, _ = couverture([0, 10, 25, 50, 100, 150], [1, 3, 5, 9, 15], 170)
    rr, cc = np.mgrid[-170:171, -170:171]
    d = np.hypot(rr, cc)
    disque = (d > 0) & (d <= 150)
    assert (tot[disque] == 1).mean() > 0.9
    assert (tot[(d > 0) & (d <= 9)] == 1).all()                  # bande 0 (pas de 1) : exacte


def test_equivariance_d4():
    s = terrain(1)
    r = np.array([50], np.int64)
    c = np.array([44], np.int64)
    base, _ = Horizons(s, cfg([1, 3, 5, 5]))(r, c)
    rot = lambda a: np.rot90(a, k=-1).copy()
    sr = _S(rot(s.z_opaque), rot(s.z_veg), rot(s.k_ext), rot(s.z_tgt))
    hr, _ = Horizons(sr, cfg([1, 3, 5, 5]))(np.array([c[0]]), np.array([N - 1 - r[0]]))
    perm = (np.arange(NS) + 2) % NS
    attendu = np.empty_like(base)
    attendu[..., perm] = base
    np.testing.assert_allclose(hr, attendu, atol=1e-5)


def test_transformer_egale_d4():
    rng = np.random.default_rng(3)
    x = rng.random((5, 16, 16)).astype(np.float32)
    y = rng.random((5 * 2 * 8, 16, 16)).astype(np.float32)
    for k in range(4):
        for m in (False, True):
            _, yd = d4(x, y, k, m, 8, (5, 2, 8))
            yt = transformer(y.reshape(5, 2, 8, 16, 16), k, m, 8, secteur=True)
            np.testing.assert_array_equal(yd, yt.reshape(y.shape))

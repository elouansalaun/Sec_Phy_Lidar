"""
Tests du réseau et de l'augmentation (src/surrogate.py).

1. Champ récepteur ≥ 2R + 1 = 301 px, et localité : un pixel loin du champ n'a aucune influence.
2. Équivariance par translation en évaluation : décaler l'entrée décale la sortie.
3. L'augmentation D4 de `surrogate.d4` est EXACTEMENT la transformation des étiquettes calculées
   par le noyau sur le terrain transformé (lien entre 5.2 et 5.3).
4. Masque de marge : aucune étiquette dans la bande où le champ récepteur est incomplet.

Usage : python -m pytest tests/test_surrogate_model.py -v
"""
import numpy as np
import torch

from src.dataset_surrogate import bins_disque, potentiel
from src.surrogate import Tuiles, UNetPotentiel, champ_recepteur, d4
from src.site import load_config


def reseau():
    nc = load_config()["surrogate"]["reseau"]
    torch.manual_seed(0)
    return UNetPotentiel(base=nc["base"], niveaux=nc["niveaux"],
                         dilatations=tuple(nc["dilatations"])).eval()


def test_champ_recepteur():
    cfg = load_config()
    rf = champ_recepteur(reseau(), 512)
    assert rf >= 2 * cfg["surrogate"]["rayon_m"] / cfg["site"]["resolution_m"] + 1
    assert rf < 512                                   # local : pas de dépendance à toute la tuile


def test_equivariance_translation():
    m = reseau()
    torch.manual_seed(1)
    x = torch.randn(1, 5, 256, 256)
    with torch.no_grad():
        y = m(x)
        ys = m(torch.roll(x, shifts=(16, 32), dims=(2, 3)))  # multiple du facteur 16
    # loin des bords (le rembourrage et le roulement y diffèrent)
    np.testing.assert_allclose(ys[..., 96:160, 112:176], y[..., 80:144, 80:144], atol=1e-5)


# -----------------------------------------------------------------------------
# D4 : augmentation = étiquettes du terrain transformé
# -----------------------------------------------------------------------------
N, R, NS = 71, 25, 8
H = np.array([2.0, 6.0])
DR, DC, B, NBIN = bins_disque(R, [0, 10, 25], NS, 1.0)


def etiquettes(zo, zv, k, zt):
    """Carte d'étiquettes (2·2·8, N, N) aux positions à disque complet, NaN ailleurs."""
    r, c = np.mgrid[R:N - R, R:N - R]
    r, c = r.ravel().astype(np.int64), c.ravel().astype(np.int64)
    V = potentiel(zo, zv, k, zt, zo < zt, r, c, H, DR, DC, B, NBIN, 1.0)
    Y = np.full((V[0].size, N, N), np.nan, np.float32)
    Y[:, r, c] = V.reshape(r.size, -1).T
    return Y


def test_d4_egale_etiquettes_du_terrain_transforme():
    rng = np.random.default_rng(0)
    zo = rng.uniform(0, 1, (N, N))
    zo[rng.random((N, N)) < 0.05] += 6
    zv = zo + np.where(rng.random((N, N)) < 0.1, 4.0, 0.0)
    k = np.where(zv > zo, 0.3, 0.0)
    zt = zo + 1.0
    zt[zo > 3] = zo[zo > 3] - 1.0                     # bâti : pas de sol
    Y = etiquettes(zo, zv, k, zt)
    X = np.stack([zo, zv, k, zt])
    for k_rot in range(4):
        for miroir in (False, True):
            Xa, Ya = d4(X, Y, k_rot, miroir, NS, (H.size, 2, NS))
            Yref = etiquettes(*Xa)
            np.testing.assert_allclose(Ya, Yref, atol=1e-5, equal_nan=True)


def test_marge_des_tuiles():
    X = np.zeros((5, 300, 300), np.float32)
    Y = np.full((80, 300, 300), np.nan, np.float32)
    Y[:, 100:200, 100:200] = 0.5
    t = Tuiles(X, Y, taille=128, marge=40, ns=8, forme_y=(5, 2, 8),
               rng=np.random.default_rng(0))
    _, yb = t.lot(16)
    lab = ~torch.isnan(yb[:, 0])
    assert lab.any()
    assert not lab[:, :40].any() and not lab[:, -40:].any()
    assert not lab[:, :, :40].any() and not lab[:, :, -40:].any()

"""
Tests of the network and of the augmentation (src/surrogate.py).

1. Receptive field ≥ 2R + 1 = 301 px, and locality: a pixel far from the field has no influence.
2. Translation equivariance in evaluation: shifting the input shifts the output.
3. The D4 augmentation of `surrogate.d4` is EXACTLY the transformation of the labels computed
   by the kernel on the transformed terrain (link between 5.2 and 5.3).
4. Margin mask: no label in the band where the receptive field is incomplete.

Usage : python -m pytest tests/test_surrogate_model.py -v
"""
import numpy as np
import torch

from src.site import load_config
from src.surrogate import PotentialUNet, Tiles, d4, receptive_field
from src.surrogate_dataset import disc_bins, potential


def network():
    nc = load_config()["surrogate"]["network"]
    torch.manual_seed(0)
    return PotentialUNet(base=nc["base"], levels=nc["levels"],
                         dilations=tuple(nc["dilations"])).eval()


def test_receptive_field():
    cfg = load_config()
    rf = receptive_field(network(), 512)
    assert rf >= 2 * cfg["surrogate"]["radius_m"] / cfg["site"]["resolution_m"] + 1
    assert rf < 512                                   # local: no dependency on the whole tile


def test_translation_equivariance():
    m = network()
    torch.manual_seed(1)
    x = torch.randn(1, 5, 256, 256)
    with torch.no_grad():
        y = m(x)
        ys = m(torch.roll(x, shifts=(16, 32), dims=(2, 3)))  # multiple of the factor 16
    # far from the edges (the padding and the roll differ there)
    np.testing.assert_allclose(ys[..., 96:160, 112:176], y[..., 80:144, 80:144], atol=1e-5)


# -----------------------------------------------------------------------------
# D4: augmentation = labels of the transformed terrain
# -----------------------------------------------------------------------------
N, R, NS = 71, 25, 8
H = np.array([2.0, 6.0])
DR, DC, B, NBIN = disc_bins(R, [0, 10, 25], NS, 1.0)


def labels(zo, zv, k, zt):
    """Label map (2·2·8, N, N) at the positions with a complete disc, NaN elsewhere."""
    r, c = np.mgrid[R:N - R, R:N - R]
    r, c = r.ravel().astype(np.int64), c.ravel().astype(np.int64)
    V = potential(zo, zv, k, zt, zo < zt, r, c, H, DR, DC, B, NBIN, 1.0)
    Y = np.full((V[0].size, N, N), np.nan, np.float32)
    Y[:, r, c] = V.reshape(r.size, -1).T
    return Y


def test_d4_equals_labels_of_the_transformed_terrain():
    rng = np.random.default_rng(0)
    zo = rng.uniform(0, 1, (N, N))
    zo[rng.random((N, N)) < 0.05] += 6
    zv = zo + np.where(rng.random((N, N)) < 0.1, 4.0, 0.0)
    k = np.where(zv > zo, 0.3, 0.0)
    zt = zo + 1.0
    zt[zo > 3] = zo[zo > 3] - 1.0                     # buildings: no ground
    Y = labels(zo, zv, k, zt)
    X = np.stack([zo, zv, k, zt])
    for k_rot in range(4):
        for mirror in (False, True):
            Xa, Ya = d4(X, Y, k_rot, mirror, NS, (H.size, 2, NS))
            Yref = labels(*Xa)
            np.testing.assert_allclose(Ya, Yref, atol=1e-5, equal_nan=True)


def test_tile_margin():
    X = np.zeros((5, 300, 300), np.float32)
    Y = np.full((80, 300, 300), np.nan, np.float32)
    Y[:, 100:200, 100:200] = 0.5
    t = Tiles(X, Y, size=128, margin=40, ns=8, shape_y=(5, 2, 8),
              rng=np.random.default_rng(0))
    _, yb = t.batch(16)
    lab = ~torch.isnan(yb[:, 0])
    assert lab.any()
    assert not lab[:, :40].any() and not lab[:, -40:].any()
    assert not lab[:, :, :40].any() and not lab[:, :, -40:].any()

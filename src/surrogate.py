"""
src/surrogate.py: fully convolutional network for the potential map (§8, step 5.3).

Input (5 channels, 1 m, relative elevations -> vertical invariance, §8.2):
  0  relief     (DTM − local mean of the DTM over 301 m) / 10
  1  buildings  (z_opaque − DTM) / 10
  2  canopy     (z_veg − z_opaque) / 10
  3  opacity    log(1 + k · (z_veg − z_opaque))   vertical optical depth of the column
  4  ground     1 if an intruder can stand there
Output: 80 channels = (height, ring, sector), sigmoid, V ∈ [0, 1] at each pixel.

Architecture: U-Net (Ronneberger et al., 2015) with `levels` downsamplings and a bottleneck of
dilated convolutions, for a receptive field ≥ 2R + 1 = 301 m (checked by a test).
Training: tiles drawn at random, exact D4 augmentation (the sectors are permuted with the
terrain, §8.2), squared loss on the labelled pixels only whose receptive field is complete
(more than `margin_px` from the tile edge).

Outputs (surrogate.out_dir): model.pt, history.csv, map.npz (prediction on the whole grid),
                             surrogate.csv (test metrics, same columns as baselines.csv)
Usage : python -m src.surrogate train
        python -m src.surrogate evaluate
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

from src.rasters import load_grid
from src.site import ROOT, load_config
from src.surrogate_metrics import bin_areas, report, site_interior
from src.viewshed import prepare_surfaces, read


# =============================================================================
# Step 1: inputs and labels as rasters
# =============================================================================
def inputs(cfg) -> np.ndarray:
    s = prepare_surfaces(cfg)
    dtm = read(ROOT / cfg["rasters"]["out_dir"] / "dtm.tif")
    win = int(2 * cfg["surrogate"]["radius_m"] / cfg["site"]["resolution_m"]) + 1
    relief = dtm - uniform_filter(dtm, size=win, mode="nearest")
    can = np.maximum(s.z_veg - s.z_opaque, 0.0)            # zero on buildings
    X = np.stack([relief / 10.0,
                  (s.z_opaque - dtm) / 10.0,
                  can / 10.0,
                  np.log1p(s.k_ext * can),
                  (s.z_opaque < s.z_tgt).astype(np.float64)])
    return np.nan_to_num(X).astype(np.float32)


def label_raster(d, shape, wanted_split) -> np.ndarray:
    """(80, H, W) with NaN outside the labelled positions of the requested split."""
    Y = np.full((80, *shape), np.nan, np.float32)
    m = np.isin(d["split"], wanted_split) if "split" in d else np.ones(d["rows"].size, bool)
    Y[:, d["rows"][m], d["cols"][m]] = d["Y"][m].reshape(m.sum(), -1).T
    return Y


# =============================================================================
# Step 2: exact D4 augmentation
# =============================================================================
def d4(x: np.ndarray, y: np.ndarray, k_rot: int, mirror: bool, ns: int, shape_y):
    """x (C, T, T), y (80, T, T). Clockwise rotation of k_rot × 90°, then east-west mirror.
    Sectors: rotation -> s + 2·k_rot; mirror -> −s (tests/test_surrogate_data.py)."""
    x = np.rot90(x, k=-k_rot, axes=(1, 2))
    y = np.rot90(y, k=-k_rot, axes=(1, 2))
    perm = (np.arange(ns) + 2 * k_rot) % ns
    if mirror:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
        perm = (ns - perm) % ns
    nh, na, _ = shape_y
    y4 = y.reshape(nh, na, ns, *y.shape[1:])
    out = np.empty_like(y4)
    out[:, :, perm] = y4                                      # sector s -> perm[s]
    return np.ascontiguousarray(x), np.ascontiguousarray(out.reshape(y.shape))


class Tiles:
    def __init__(self, X, Y, size, margin, ns, shape_y, rng):
        self.X, self.Y, self.T, self.m = X, Y, size, margin
        self.ns, self.shape_y, self.rng = ns, shape_y, rng
        H, W = X.shape[1:]
        lab = ~np.isnan(Y[0])
        # origins whose interior area contains at least one label
        r, c = np.nonzero(lab)
        self.cand = (r, c)
        self.H, self.W = H, W

    def batch(self, n):
        xs, ys = [], []
        T, m = self.T, self.m
        for _ in range(n):
            i = self.rng.integers(self.cand[0].size)            # centred on a label
            r = int(np.clip(self.cand[0][i] - self.rng.integers(m, T - m), 0, self.H - T))
            c = int(np.clip(self.cand[1][i] - self.rng.integers(m, T - m), 0, self.W - T))
            x = self.X[:, r:r + T, c:c + T]
            y = self.Y[:, r:r + T, c:c + T].copy()
            y[:, :m] = y[:, T - m:] = np.nan                    # incomplete receptive field
            y[:, :, :m] = y[:, :, T - m:] = np.nan
            x, y = d4(x, y, int(self.rng.integers(4)), bool(self.rng.integers(2)),
                      self.ns, self.shape_y)
            xs.append(x)
            ys.append(y)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# =============================================================================
# Step 3: network
# =============================================================================
def block(i, o, dil=1):
    # BatchNorm and not GroupNorm: in evaluation, GroupNorm normalises over the whole image and
    # would make each pixel depend on the whole tile (no more translation equivariance).
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=dil, dilation=dil), nn.BatchNorm2d(o),
                         nn.SiLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=dil, dilation=dil), nn.BatchNorm2d(o),
                         nn.SiLU(inplace=True))


class PotentialUNet(nn.Module):
    def __init__(self, c_in=5, c_out=80, base=16, levels=5, dilations=(2, 4)):
        super().__init__()
        ch = [min(base * 2 ** i, 8 * base) for i in range(levels + 1)]
        self.encoder = nn.ModuleList([block(c_in, ch[0])] +
                                     [block(ch[i], ch[i + 1]) for i in range(levels)])
        self.bottleneck = nn.Sequential(*[block(ch[-1], ch[-1], d) for d in dilations])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2)
                                 for i in reversed(range(levels))])
        self.decoder = nn.ModuleList([block(2 * ch[i], ch[i]) for i in reversed(range(levels))])
        self.head = nn.Conv2d(ch[0], c_out, 1)
        self.factor = 2 ** levels

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.encoder):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.bottleneck(skips.pop())
        for u, b in zip(self.up, self.decoder):
            x = b(torch.cat([u(x), skips.pop()], 1))
        return torch.sigmoid(self.head(x))


def receptive_field(model, size=512) -> int:
    """Width (px) of the input area that influences the central pixel: non-zero gradient."""
    model = model.eval().float()
    x = torch.zeros(1, model.encoder[0][0].in_channels, size, size, requires_grad=True)
    y = model(x)
    y[0, :, size // 2, size // 2].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1))
    cols = torch.nonzero(g.sum(0) > 0).ravel()
    return int(cols.max() - cols.min() + 1)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict(model, X, dev) -> np.ndarray:
    """Full map in one pass: (80, H, W)."""
    f = model.factor
    H, W = X.shape[1:]
    ph, pw = (-H) % f, (-W) % f
    x = torch.from_numpy(np.pad(X, ((0, 0), (0, ph), (0, pw)), mode="reflect"))[None].to(dev)
    model.eval()
    with torch.no_grad():
        y = model(x)[0, :, :H, :W]
    return y.float().cpu().numpy()


# =============================================================================
# Step 4: training
# =============================================================================
def train(cfg) -> None:
    sc, nc = cfg["surrogate"], cfg["surrogate"]["network"]
    out = ROOT / sc["out_dir"]
    torch.manual_seed(cfg["project"]["random_seed"])
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    dev = device()

    X = inputs(cfg)
    d = np.load(out / "labels.npz")
    shape_y = d["Y"].shape[1:]
    Ytr = label_raster(d, X.shape[1:], [0])
    Yva = label_raster(d, X.shape[1:], [1])
    va = ~np.isnan(Yva[0])

    model = PotentialUNet(base=nc["base"], levels=nc["levels"], dilations=tuple(nc["dilations"]))
    rf = receptive_field(model)
    margin = (rf + 1) // 2
    assert rf >= 2 * sc["radius_m"] + 1, f"receptive field {rf} px < 301 px"
    assert nc["tile_px"] > 2 * margin, "tile too small for the receptive field"
    model = model.to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{dev}; {n_par / 1e6:.2f} M parameters; receptive field {rf} px; "
          f"margin {margin} px; tile {nc['tile_px']} px")

    tiles = Tiles(X, Ytr, nc["tile_px"], margin, shape_y[-1], shape_y, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=nc["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=nc["lr"], total_steps=nc["iterations"],
                                                pct_start=0.05)
    hist, best = [], np.inf
    t0 = time.time()
    for it in range(1, nc["iterations"] + 1):
        model.train()
        xb, yb = tiles.batch(nc["batch"])
        xb, yb = xb.to(dev), yb.to(dev)
        m = ~torch.isnan(yb)
        loss = F.mse_loss(model(xb)[m], yb[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % nc["eval_every"] == 0 or it == nc["iterations"]:
            P = predict(model, X, dev)
            mae = float(np.nanmean(np.abs(P[:, va] - Yva[:, va])))
            hist.append(dict(iteration=it, loss=float(loss.detach()), mae_validation=mae,
                             time_s=time.time() - t0))
            flag = ""
            if mae < best:
                best = mae
                torch.save({"state": model.state_dict(), "config": nc, "rf": rf}, out / "model.pt")
                flag = "  *"
            print(f"  it {it:5d}  loss {float(loss.detach()):.5f}  validation MAE {mae:.4f}  "
                  f"{time.time() - t0:5.0f} s{flag}", flush=True)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print(f"Best validation MAE: {best:.4f}; {time.time() - t0:.0f} s -> {out}")


# =============================================================================
# Step 5: evaluation on the test block and timing
# =============================================================================
def time_prediction(model, X, dev, n=5) -> float:
    predict(model, X, dev)
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        predict(model, X, dev)
    return (time.perf_counter() - t0) / n


def evaluate(cfg) -> None:
    sc = cfg["surrogate"]
    out = ROOT / sc["out_dir"]
    ck = torch.load(out / "model.pt", map_location="cpu")
    nc = ck["config"]
    model = PotentialUNet(base=nc["base"], levels=nc["levels"], dilations=tuple(nc["dilations"]))
    model.load_state_dict(ck["state"])
    X = inputs(cfg)
    te = np.load(out / "test.npz")
    Yt = te["Y"]

    times = {}
    for dev in {device(), torch.device("cpu")}:
        times[dev.type] = time_prediction(model.to(dev), X, dev)
    dev = device()
    P = predict(model.to(dev), X, dev)
    n_pos = int((np.asarray(__import__("rasterio").open(out / "split.tif").read(1)) != 255).sum())
    Pt = P[:, te["rows"], te["cols"]].T.reshape(Yt.shape)
    inside, xy = site_interior(cfg, ROOT, te["rows"], te["cols"])
    res = report(Pt, Yt, te["stratum"].astype(int), te["heights"], bin_areas(cfg),
                 inside, xy, name="U-Net (test)")
    res["cost_ms_position"] = 1e3 * times[dev.type] / n_pos
    res.to_csv(out / "surrogate.csv", index=False)
    np.savez_compressed(out / "map.npz", V=P.astype(np.float16))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    for k, t in times.items():
        print(f"Full map ({X.shape[1]}×{X.shape[2]}, {n_pos:,} valid positions) on {k}: "
              f"{t:.2f} s, i.e. {1e6 * t / n_pos:.1f} µs per position")


def main() -> None:
    cfg = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "evaluate": evaluate}[cmd](cfg)


if __name__ == "__main__":
    main()

"""
src/surrogate_physics.py: physics-informed U-Net, conditional on h (§8.8, step 6.1).

Idea: the raw U-Net (src/surrogate.py) must rebuild the occlusion, a maximum along rays
divided by the distance, with 3 × 3 convolutions: it underfits. We give it as input the dense
horizons of the log-polar pyramid (src/horizons.py), computed for the requested height. The far
context is then carried by the inputs: a receptive field of ≈ 85 px is enough, and the network
corrects what the block horizons do not say (exact curvature of the rays, crossed vegetation,
fine shape of the visible ground).

Input for a height h (166 channels):
  5 base channels (src.surrogate.inputs), h / 10 (constant),
  horizons at h: 2 (opaque, vegetation) × 5 bands × 8 sectors,
  means        : 2 (opacity, ground share) × 5 bands × 8 sectors.
Output: 16 channels = (ring, sector) at this height. One pass per height.
D4 augmentation: the channels indexed by sector (inputs AND outputs) are permuted with the terrain.

Outputs (surrogate.out_dir): physics_model.pt, physics_history.csv, surrogate_physics.csv,
                             physics_map.npz
Usage : python -m src.surrogate_physics train
        python -m src.surrogate_physics evaluate
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from src.horizons import Horizons
from src.site import ROOT, load_config
from src.surrogate import PotentialUNet, device, inputs, label_raster, receptive_field
from src.surrogate_metrics import bin_areas, report, site_interior
from src.viewshed import prepare_surfaces


# =============================================================================
# Step 1: inputs: base + pyramid horizons, for the whole grid
# =============================================================================
class Inputs:
    def __init__(self, cfg):
        self.base = inputs(cfg)                                    # (5, H, W) float32
        s = prepare_surfaces(cfg)
        hz = Horizons(s, cfg)
        self.hor, self.mean, self.t_horizons = hz.maps()           # float16
        self.t_horizons += hz.t_prep
        self.h = hz.h
        self.nh, self.ns = self.h.size, hz.ns
        self.c_in = (self.base.shape[0] + 1 + int(np.prod(self.hor.shape[1:-2]))
                     + int(np.prod(self.mean.shape[:-2])))             # 5 + 1 + 80 + 80 = 166

    @property
    def shape(self):
        return self.base.shape[1:]

    def tile(self, ih, r, c, T):
        """Pieces of a tile, each with its sector axis just before the spatial axes."""
        sl = (slice(r, r + T), slice(c, c + T))
        return [self.base[(slice(None), *sl)],                           # no sector
                np.full((1, T, T), self.h[ih] / 10.0, np.float32),         # no sector
                self.hor[ih][(Ellipsis, *sl)].astype(np.float32),          # (2, b, s, T, T)
                self.mean[(Ellipsis, *sl)].astype(np.float32)]             # (2, b, s, T, T)

    def window(self, ih, r, c, T) -> np.ndarray:
        """Input (166, T, T) of a window that may overflow the grid (replicated edges)."""
        H, W = self.shape
        rr = np.clip(np.arange(r, r + T), 0, H - 1)
        cc = np.clip(np.arange(c, c + T), 0, W - 1)
        r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
        x = concat(
            [self.base[:, r0:r1, c0:c1],
             np.full((1, r1 - r0, c1 - c0), self.h[ih] / 10.0, np.float32),
             self.hor[ih][..., r0:r1, c0:c1].astype(np.float32),
             self.mean[..., r0:r1, c0:c1].astype(np.float32)])
        return x[:, rr - r0][:, :, cc - c0]


def concat(pieces) -> np.ndarray:
    return np.concatenate([m.reshape(-1, *m.shape[-2:]) for m in pieces], axis=0)


# =============================================================================
# Step 2: D4 augmentation on arrays whose sector axis precedes the spatial axes
# =============================================================================
def transform(a, k_rot: int, mirror: bool, ns: int, sector: bool):
    a = np.rot90(a, k=-k_rot, axes=(-2, -1))
    perm = (np.arange(ns) + 2 * k_rot) % ns
    if mirror:
        a = a[..., ::-1]
        perm = (ns - perm) % ns
    if sector:
        out = np.empty_like(a)
        out[..., perm, :, :] = a                                  # sector s -> perm[s]
        a = out
    return np.ascontiguousarray(a)


class TilesH:
    def __init__(self, E: Inputs, Y5, size, margin, rng):
        """Y5: (h, ring, sector, H, W) with NaN outside the labels."""
        self.E, self.Y5, self.T, self.m, self.rng = E, Y5, size, margin, rng
        self.cand = np.nonzero(~np.isnan(Y5[0, 0, 0]))
        self.H, self.W = E.shape

    def batch(self, n):
        xs, ys = [], []
        T, m, ns = self.T, self.m, self.E.ns
        for _ in range(n):
            i = self.rng.integers(self.cand[0].size)
            r = int(np.clip(self.cand[0][i] - self.rng.integers(m, T - m), 0, self.H - T))
            c = int(np.clip(self.cand[1][i] - self.rng.integers(m, T - m), 0, self.W - T))
            ih = int(self.rng.integers(self.E.nh))
            k, mi = int(self.rng.integers(4)), bool(self.rng.integers(2))
            x = [transform(p, k, mi, ns, sector=j >= 2)
                 for j, p in enumerate(self.E.tile(ih, r, c, T))]
            y = self.Y5[ih, :, :, r:r + T, c:c + T].copy()
            y[..., :m, :] = y[..., T - m:, :] = np.nan
            y[..., :m] = y[..., T - m:] = np.nan
            y = transform(y, k, mi, ns, sector=True)
            xs.append(concat(x))
            ys.append(y.reshape(-1, T, T))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# =============================================================================
# Step 3: prediction by overlapping tiles (bounded memory on MPS)
# =============================================================================
def predict(model, E: Inputs, dev, margin: int, T: int = 512) -> np.ndarray:
    """-> (h, ring·sector, H, W). Tiles of T px built on demand, overlap of `margin` px:
    only the interior of each tile, where the receptive field is complete, is kept."""
    assert T % model.factor == 0
    H, W = E.shape
    step = T - 2 * margin
    out = np.zeros((E.nh, 2 * E.ns, H, W), np.float32)
    model.eval()
    with torch.no_grad():
        for ih in range(E.nh):
            for r in range(0, H, step):
                for c in range(0, W, step):
                    x = torch.from_numpy(E.window(ih, r - margin, c - margin, T))[None]
                    y = model(x.to(dev))[0].float().cpu().numpy()
                    h_, w_ = min(step, H - r), min(step, W - c)
                    out[ih, :, r:r + h_, c:c + w_] = y[:, margin:margin + h_, margin:margin + w_]
    return out


# =============================================================================
# Step 4: training
# =============================================================================
def build_model(nc, c_in):
    return PotentialUNet(c_in=c_in, c_out=16, base=nc["base"], levels=nc["levels"],
                         dilations=tuple(nc["dilations"]))


def train(cfg) -> None:
    sc, nc = cfg["surrogate"], cfg["surrogate"]["physics_network"]
    out = ROOT / sc["out_dir"]
    torch.manual_seed(cfg["project"]["random_seed"])
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    dev = device()

    E = Inputs(cfg)
    print(f"Horizons of the whole grid: {E.t_horizons:.1f} s; {E.c_in} input channels")
    d = np.load(out / "labels.npz")
    fy = d["Y"].shape[1:]
    Ytr = label_raster(d, E.shape, [0]).reshape(*fy, *E.shape)
    Yva = label_raster(d, E.shape, [1]).reshape(*fy, *E.shape)
    va = ~np.isnan(Yva[0, 0, 0])

    model = build_model(nc, E.c_in)
    rf = receptive_field(model, 256)
    margin = (rf + 1) // 2
    assert nc["tile_px"] > 2 * margin
    model = model.to(dev)
    print(f"{dev}; {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters; "
          f"receptive field {rf} px; margin {margin} px; tile {nc['tile_px']} px")

    tiles = TilesH(E, Ytr, nc["tile_px"], margin, rng)
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
            P = predict(model, E, dev, margin).reshape(*fy, *E.shape)
            mae = float(np.mean(np.abs(P[..., va] - Yva[..., va])))
            hist.append(dict(iteration=it, loss=float(loss.detach()), mae_validation=mae,
                             time_s=time.time() - t0))
            flag = ""
            if mae < best:
                best = mae
                torch.save({"state": model.state_dict(), "config": nc, "rf": rf, "c_in": E.c_in},
                           out / "physics_model.pt")
                flag = "  *"
            print(f"  it {it:5d}  loss {float(loss.detach()):.5f}  validation MAE {mae:.4f}  "
                  f"{time.time() - t0:5.0f} s{flag}", flush=True)
    pd.DataFrame(hist).to_csv(out / "physics_history.csv", index=False)
    print(f"Best validation MAE: {best:.4f}; {time.time() - t0:.0f} s -> {out}")


# =============================================================================
# Step 5: evaluation and timing (horizons + network)
# =============================================================================
def evaluate(cfg) -> None:
    sc = cfg["surrogate"]
    out = ROOT / sc["out_dir"]
    ck = torch.load(out / "physics_model.pt", map_location="cpu")
    model = build_model(ck["config"], ck["c_in"])
    model.load_state_dict(ck["state"])
    margin = (ck["rf"] + 1) // 2
    E = Inputs(cfg)
    te = np.load(out / "test.npz")
    Yt = te["Y"]
    n_pos = int((rasterio.open(out / "split.tif").read(1) != 255).sum())

    times = {}
    for dev in {device(), torch.device("cpu")}:
        m = model.to(dev)
        predict(m, E, dev, margin)                                 # warm-up
        t0 = time.perf_counter()
        P = predict(m, E, dev, margin)
        times[dev.type] = time.perf_counter() - t0
    Pt = P.reshape(E.nh, 2, E.ns, *E.shape)[..., te["rows"], te["cols"]]
    Pt = np.moveaxis(Pt, -1, 0)
    inside, xy = site_interior(cfg, ROOT, te["rows"], te["cols"])
    res = report(Pt, Yt, te["stratum"].astype(int), te["heights"], bin_areas(cfg),
                 inside, xy, name="Physics U-Net (test)")
    dev = device().type
    res["cost_us_position"] = 1e6 * (times[dev] + E.t_horizons) / n_pos
    res.to_csv(out / "surrogate_physics.csv", index=False)
    np.savez_compressed(out / "physics_map.npz", V=P.astype(np.float16))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"Horizons (whole grid, CPU): {E.t_horizons:.2f} s")
    for k, t in times.items():
        print(f"Network, 5 heights, on {k}: {t:.2f} s; total with horizons "
              f"{1e6 * (t + E.t_horizons) / n_pos:.1f} µs per position ({n_pos:,} positions)")


def main() -> None:
    cfg = load_config()
    {"train": train, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1
                                           else "train"](cfg)


if __name__ == "__main__":
    main()

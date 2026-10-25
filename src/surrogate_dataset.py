"""
src/surrogate_dataset.py: exact dataset of the surrogate model.


"""
from __future__ import annotations

import sys
import time

import geopandas as gpd
import numpy as np
from numba import njit, prange

from src.rasters import write_raster
from src.site import ROOT, load_config
from src.viewshed import prepare_surfaces, ray_tau, to_pixel

STRATA = ("roof", "open_ground", "vegetated_ground")
SPLITS = {"train": 0, "validation": 1, "test": 2}
EXCLUDED = 255


# =============================================================================
# Step 1: geometry of the bins: disc offsets, ring and sector of each one
# =============================================================================
def disc_bins(radius_m: float, rings_m, n_sectors: int, res: float):
    """Offsets (dr, dc) of the disc (centre excluded), bin index = ring·n_sectors + sector,
    and number of cells per bin (denominator of V, independent of the terrain).

    Azimuth counted from north, clockwise (camera convention of src.sensors);
    sector k is centred on k·360/n_sectors."""
    R = int(radius_m / res)
    dr, dc = np.mgrid[-R:R + 1, -R:R + 1]
    dr, dc = dr.ravel(), dc.ravel()
    d = np.hypot(dr, dc) * res
    keep = (dr * dr + dc * dc <= R * R) & (d > 0)              # same disc as optical_depth
    dr, dc, d = dr[keep], dc[keep], d[keep]
    az = np.degrees(np.arctan2(dc, -dr)) % 360.0               # east = +c, north = −r
    w = 360.0 / n_sectors
    sect = np.floor((az + w / 2) / w).astype(np.int64) % n_sectors
    edges = np.asarray(rings_m, float)
    ring = np.searchsorted(edges[1:-1], d, side="left")         # d ≤ 50 -> 0; 50 < d ≤ 150 -> 1
    b = ring * n_sectors + sect
    n_bin = np.bincount(b, minlength=(len(edges) - 1) * n_sectors).astype(np.float64)
    return dr.astype(np.int64), dc.astype(np.int64), b, n_bin


# =============================================================================
# Step 2: kernel: exact potential of a list of observers
# =============================================================================
@njit(parallel=True, cache=True)
def potential(z_opaque, z_veg, k_ext, z_tgt, ground, obs_r, obs_c, heights,
              dr, dc, b, n_bin, res):
    """V[i, h, bin] for each observer i. Parallel over the observers."""
    n, nh, nb = obs_r.size, heights.size, n_bin.size
    out = np.zeros((n, nh, nb), np.float32)
    for i in prange(n):
        r0, c0 = obs_r[i], obs_c[i]
        acc = np.zeros(nb)
        for ih in range(nh):
            z_obs = z_opaque[r0, c0] + heights[ih]
            acc[:] = 0.0
            for m in range(dr.size):
                r, c = r0 + dr[m], c0 + dc[m]
                if not ground[r, c]:
                    continue
                tau = ray_tau(z_opaque, z_veg, k_ext, r0, c0, z_obs, r, c, z_tgt[r, c], res)
                if np.isfinite(tau):
                    acc[b[m]] += np.exp(-tau)
            for q in range(nb):
                out[i, ih, q] = acc[q] / n_bin[q]
    return out


# =============================================================================
# Step 3: strata and spatial split
# =============================================================================
def strata(s, cfg) -> np.ndarray:
    """0 roof (opaque surface above the target), 1 open ground, 2 ground under vegetation."""
    ground = s.z_opaque < s.z_tgt
    veg = (s.z_veg - s.z_opaque) > cfg["viewshed"]["veg_min_h_m"]
    st = np.zeros(ground.shape, np.uint8)
    st[ground & ~veg] = 1
    st[ground & veg] = 2
    return st


def spatial_split(s, cfg, site_bounds) -> np.ndarray:
    sc, g = cfg["surrogate"], s.grid
    H, W = s.z_tgt.shape
    R = int(sc["radius_m"] / g.res)
    split = np.full((H, W), EXCLUDED, np.uint8)
    valid = np.zeros((H, W), bool)
    valid[R:H - R, R:W - R] = True                      # full disc inside the grid

    # Test: site extent + margin (in pixels)
    xmin, ymin, xmax, ymax = site_bounds
    m = sc["test"]["margin_m"]
    r_a, c_a = to_pixel(g, xmin - m, ymax + m)
    r_b, c_b = to_pixel(g, xmax + m, ymin - m)
    test = np.zeros((H, W), bool)
    test[max(r_a, 0):r_b + 1, max(c_a, 0):c_b + 1] = True
    t = int(sc["test"]["buffer_m"] / g.res)
    buffer = np.zeros((H, W), bool)
    buffer[max(r_a - t, 0):r_b + t + 1, max(c_a - t, 0):c_b + t + 1] = True

    # Validation: square tiles drawn at random outside the buffer
    tu = int(sc["validation"]["tile_m"] / g.res)
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    tr, tc = np.divmod(np.arange(H * W), W)
    tile = (tr // tu) * (W // tu + 1) + (tc // tu)
    tile = tile.reshape(H, W)
    free = np.unique(tile[valid & ~buffer])
    n_val = int(round(sc["validation"]["share"] * free.size))
    val_t = rng.choice(free, n_val, replace=False)

    split[valid & ~buffer] = SPLITS["train"]
    split[valid & ~buffer & np.isin(tile, val_t)] = SPLITS["validation"]
    split[valid & test] = SPLITS["test"]
    return split


# =============================================================================
# Step 4: sampling of the observers
# =============================================================================
def sample_observers(split, st, n: int, props: dict, rng):
    """Training + validation observers, stratified by support type."""
    rows, cols = [], []
    for k, name in enumerate(STRATA):
        cand = np.flatnonzero((split.ravel() <= SPLITS["validation"]) & (st.ravel() == k))
        m = min(int(round(n * props[name])), cand.size)
        pick = rng.choice(cand, m, replace=False)
        r, c = np.divmod(pick, split.shape[1])
        rows.append(r)
        cols.append(c)
    return np.concatenate(rows), np.concatenate(cols)


def test_observers(split, step: int):
    r, c = np.nonzero(split == SPLITS["test"])
    keep = (r % step == 0) & (c % step == 0)
    return r[keep], c[keep]


# =============================================================================
# Step 5: batched computation, with time tracking
# =============================================================================
class Context:
    def __init__(self, cfg):
        self.cfg = cfg
        sc = cfg["surrogate"]
        self.s = prepare_surfaces(cfg)
        self.ground = self.s.z_opaque < self.s.z_tgt
        self.h = np.asarray(sc["heights_m"], np.float64)
        self.dr, self.dc, self.b, self.n_bin = disc_bins(
            sc["radius_m"], sc["rings_m"], sc["n_sectors"], self.s.grid.res)
        self.shape = (self.h.size, len(sc["rings_m"]) - 1, sc["n_sectors"])

    def compute(self, rows, cols, batch: int = 2000, label: str = "") -> tuple[np.ndarray, float]:
        s = self.s
        out = np.empty((rows.size, self.h.size, self.n_bin.size), np.float32)
        t0 = time.time()
        for a in range(0, rows.size, batch):
            sl = slice(a, a + batch)
            out[sl] = potential(s.z_opaque, s.z_veg, s.k_ext, s.z_tgt, self.ground,
                                rows[sl].astype(np.int64), cols[sl].astype(np.int64), self.h,
                                self.dr, self.dc, self.b, self.n_bin, s.grid.res)
            done = min(a + batch, rows.size)
            el = time.time() - t0
            print(f"  {label} {done:>7,}/{rows.size:,}  {el:6.0f} s  "
                  f"(≈ {el / done * (rows.size - done) / 60:.0f} min left)", flush=True)
        return out.reshape(rows.size, *self.shape), time.time() - t0


def main() -> None:
    cfg = load_config()
    sc = cfg["surrogate"]
    pilot = int(sys.argv[sys.argv.index("--pilot") + 1]) if "--pilot" in sys.argv else 0
    ctx = Context(cfg)
    s = ctx.s
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    split = spatial_split(s, cfg, per.bounds)
    st = strata(s, cfg)
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    n_valid = int((split != EXCLUDED).sum())
    cnt = {k: int((split == v).sum()) for k, v in SPLITS.items()}
    print(f"{ctx.dr.size:,} cells per disc (R = {sc['radius_m']} m), "
          f"{ctx.n_bin.size} bins × {ctx.h.size} heights = {ctx.n_bin.size * ctx.h.size} outputs")
    print(f"Valid positions: {n_valid:,} ({cnt})")
    if not (cnt["train"] and cnt["validation"]):
        sys.exit(f"No training or validation position: the grid ({s.grid.width} x {s.grid.height} px) "
                 f"leaves no full {sc['radius_m']} m disc outside the test block (site + "
                 f"{sc['test']['margin_m']} m + {sc['test']['buffer_m']} m). The surrogate model needs "
                 f"a larger grid: a site farther from the edge of the tiles, or neighbouring tiles.")

    # Compilation outside the timing
    potential(s.z_opaque, s.z_veg, s.k_ext, s.z_tgt, ctx.ground, np.array([500]), np.array([500]),
              ctx.h, ctx.dr, ctx.dc, ctx.b, ctx.n_bin, s.grid.res)

    if pilot:
        r, c = sample_observers(split, st, pilot, sc["sampling"]["proportions"], rng)
        _, dt = ctx.compute(r, c, batch=max(pilot // 4, 1), label="pilot")
        per_obs = dt / r.size
        n_tr = sc["sampling"]["n_observers"]
        n_te = test_observers(split, sc["test_step_px"])[0].size
        print(f"\n{1e3 * per_obs:.1f} ms per observer ({ctx.h.size} heights) on this machine")
        print(f"  training set ({n_tr:,}): ≈ {per_obs * n_tr / 60:.0f} min")
        print(f"  test block ({n_te:,}, step {sc['test_step_px']} px): ≈ {per_obs * n_te / 60:.0f} min")
        print(f"  full exact map ({n_valid:,} positions): ≈ {per_obs * n_valid / 3600:.1f} h")
        return

    out = ROOT / sc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_raster(out / "split.tif", split, s.grid)
    write_raster(out / "strata.tif", st, s.grid)

    r, c = sample_observers(split, st, sc["sampling"]["n_observers"],
                            sc["sampling"]["proportions"], rng)
    Y, dt = ctx.compute(r, c, label="train+validation")
    np.savez_compressed(out / "labels.npz", rows=r, cols=c, split=split[r, c], stratum=st[r, c],
                        Y=Y, heights=ctx.h, rings=np.asarray(sc["rings_m"]),
                        n_sectors=sc["n_sectors"], time_s=dt)
    per_obs = dt / r.size

    rt, ct = test_observers(split, sc["test_step_px"])
    Yt, dtt = ctx.compute(rt, ct, label="test")
    np.savez_compressed(out / "test.npz", rows=rt, cols=ct, stratum=st[rt, ct], Y=Yt,
                        heights=ctx.h, rings=np.asarray(sc["rings_m"]),
                        n_sectors=sc["n_sectors"], time_s=dtt)

    print(f"\n{r.size:,} observers sampled in {dt / 60:.1f} min, "
          f"{rt.size:,} test observers in {dtt / 60:.1f} min")
    print(f"Exact cost: {1e3 * per_obs:.1f} ms per observer; full map "
          f"({n_valid:,} positions) ≈ {per_obs * n_valid / 3600:.1f} h on this machine")
    for k, name in enumerate(STRATA):
        m = st[r, c] == k
        print(f"  {name:17s} n = {m.sum():6,}  mean V (h = {ctx.h[-1]:g} m, ring 50-150 m) = "
              f"{Y[m, -1, -1].mean():.3f}")
    figure(split, st, r, c, per, s.grid, out / "split.png")
    print(f"-> {out}")


def figure(split, st, r, c, per, grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)
    img = np.where(split == EXCLUDED, np.nan, split.astype(float))
    ext = (grid.xmin, grid.xmax, grid.ymin, grid.ymax)
    ax.imshow(img, cmap=ListedColormap(["#dfe9f5", "#f6dcc8", "#cfeede"]), extent=ext,
              vmin=0, vmax=2, interpolation="nearest")
    x = grid.xmin + (c + 0.5) * grid.res
    y = grid.ymax - (r + 0.5) * grid.res
    cols = np.array(["#4a3aa7", "#2a78d6", "#1baf7a"])[st[r, c]]
    ax.scatter(x, y, s=0.3, c=cols, lw=0)
    ax.plot(*per.exterior.xy, color="#0b0b0b", lw=1)
    ax.set_title("Spatial split (blue: training, orange: validation, green: test)\n"
                 "points: observers (purple roof, blue open ground, green under vegetation)",
                 loc="left", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

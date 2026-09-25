"""
src/surrogate.py — Réseau entièrement convolutif pour la carte de potentiel (§8, étape 5.3).

Entrée (5 canaux, 1 m, altitudes relatives -> invariance verticale, §8.2) :
  0  relief     (MNT − moyenne locale du MNT sur 301 m) / 10
  1  bâti       (z_opaque − MNT) / 10
  2  canopée    (z_veg − z_opaque) / 10
  3  opacité    log(1 + k · (z_veg − z_opaque))   épaisseur optique verticale de la colonne
  4  sol        1 si un intrus peut s'y tenir
Sortie : 80 canaux = (hauteur, anneau, secteur), sigmoïde, V ∈ [0, 1] en chaque pixel.

Architecture : U-Net (Ronneberger et al., 2015) à `niveaux` sous-échantillonnages et goulot à
convolutions dilatées, pour un champ récepteur ≥ 2R + 1 = 301 m (vérifié par un test).
Entraînement : tuiles tirées au hasard, augmentation D4 exacte (les secteurs sont permutés avec
le terrain, §8.2), perte quadratique sur les seuls pixels étiquetés dont le champ récepteur est
entier (à plus de `marge_px` du bord de la tuile).

Sorties (surrogate.out_dir) : modele.pt, historique.csv, carte.npz (prédiction sur toute la grille),
                              surrogate.csv (métriques sur le test, mêmes colonnes que baselines.csv)
Usage : python -m src.surrogate entrainer
        python -m src.surrogate evaluer
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

from src.eval_surrogate import interieur_site, rapport, surfaces_bins
from src.rasters import load_grid
from src.site import ROOT, load_config
from src.viewshed import prepare_surfaces, read


# =============================================================================
# Étape 1 — Entrées et étiquettes sous forme de rasters
# =============================================================================
def entrees(cfg) -> np.ndarray:
    s = prepare_surfaces(cfg)
    mnt = read(ROOT / cfg["rasters"]["out_dir"] / "mnt.tif")
    fen = int(2 * cfg["surrogate"]["rayon_m"] / cfg["site"]["resolution_m"]) + 1
    relief = mnt - uniform_filter(mnt, size=fen, mode="nearest")
    can = np.maximum(s.z_veg - s.z_opaque, 0.0)            # nul sur le bâti
    X = np.stack([relief / 10.0,
                  (s.z_opaque - mnt) / 10.0,
                  can / 10.0,
                  np.log1p(s.k_ext * can),
                  (s.z_opaque < s.z_tgt).astype(np.float64)])
    return np.nan_to_num(X).astype(np.float32)


def etiquettes_raster(d, forme, split_voulu) -> np.ndarray:
    """(80, H, W) avec NaN hors des positions étiquetées du split demandé."""
    Y = np.full((80, *forme), np.nan, np.float32)
    m = np.isin(d["split"], split_voulu) if "split" in d else np.ones(d["rows"].size, bool)
    Y[:, d["rows"][m], d["cols"][m]] = d["Y"][m].reshape(m.sum(), -1).T
    return Y


# =============================================================================
# Étape 2 — Augmentation D4 exacte
# =============================================================================
def d4(x: np.ndarray, y: np.ndarray, k_rot: int, miroir: bool, ns: int, forme_y):
    """x (C, T, T), y (80, T, T). Rotation horaire de k_rot × 90°, puis miroir est-ouest.
    Secteurs : rotation -> s + 2·k_rot ; miroir -> −s (tests/test_surrogate_data.py)."""
    x = np.rot90(x, k=-k_rot, axes=(1, 2))
    y = np.rot90(y, k=-k_rot, axes=(1, 2))
    perm = (np.arange(ns) + 2 * k_rot) % ns
    if miroir:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
        perm = (ns - perm) % ns
    nh, na, _ = forme_y
    y4 = y.reshape(nh, na, ns, *y.shape[1:])
    out = np.empty_like(y4)
    out[:, :, perm] = y4                                      # secteur s -> perm[s]
    return np.ascontiguousarray(x), np.ascontiguousarray(out.reshape(y.shape))


class Tuiles:
    def __init__(self, X, Y, taille, marge, ns, forme_y, rng):
        self.X, self.Y, self.T, self.m = X, Y, taille, marge
        self.ns, self.forme_y, self.rng = ns, forme_y, rng
        H, W = X.shape[1:]
        lab = ~np.isnan(Y[0])
        # origines dont la zone intérieure contient au moins une étiquette
        r, c = np.nonzero(lab)
        self.cand = (r, c)
        self.H, self.W = H, W

    def lot(self, n):
        xs, ys = [], []
        T, m = self.T, self.m
        for _ in range(n):
            i = self.rng.integers(self.cand[0].size)            # centre sur une étiquette
            r = int(np.clip(self.cand[0][i] - self.rng.integers(m, T - m), 0, self.H - T))
            c = int(np.clip(self.cand[1][i] - self.rng.integers(m, T - m), 0, self.W - T))
            x = self.X[:, r:r + T, c:c + T]
            y = self.Y[:, r:r + T, c:c + T].copy()
            y[:, :m] = y[:, T - m:] = np.nan                    # champ récepteur incomplet
            y[:, :, :m] = y[:, :, T - m:] = np.nan
            x, y = d4(x, y, int(self.rng.integers(4)), bool(self.rng.integers(2)),
                      self.ns, self.forme_y)
            xs.append(x)
            ys.append(y)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# =============================================================================
# Étape 3 — Réseau
# =============================================================================
def bloc(i, o, dil=1):
    # BatchNorm et non GroupNorm : en évaluation, GroupNorm normalise sur toute l'image et
    # rendrait chaque pixel dépendant de toute la tuile (plus d'équivariance par translation).
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=dil, dilation=dil), nn.BatchNorm2d(o),
                         nn.SiLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=dil, dilation=dil), nn.BatchNorm2d(o),
                         nn.SiLU(inplace=True))


class UNetPotentiel(nn.Module):
    def __init__(self, c_in=5, c_out=80, base=16, niveaux=5, dilatations=(2, 4)):
        super().__init__()
        ch = [min(base * 2 ** i, 8 * base) for i in range(niveaux + 1)]
        self.desc = nn.ModuleList([bloc(c_in, ch[0])] +
                                  [bloc(ch[i], ch[i + 1]) for i in range(niveaux)])
        self.goulot = nn.Sequential(*[bloc(ch[-1], ch[-1], d) for d in dilatations])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2)
                                 for i in reversed(range(niveaux))])
        self.mont = nn.ModuleList([bloc(2 * ch[i], ch[i]) for i in reversed(range(niveaux))])
        self.tete = nn.Conv2d(ch[0], c_out, 1)
        self.facteur = 2 ** niveaux

    def forward(self, x):
        sauts = []
        for i, b in enumerate(self.desc):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            sauts.append(x)
        x = self.goulot(sauts.pop())
        for u, b in zip(self.up, self.mont):
            x = b(torch.cat([u(x), sauts.pop()], 1))
        return torch.sigmoid(self.tete(x))


def champ_recepteur(model, taille=512) -> int:
    """Largeur (px) de la zone d'entrée qui influence le pixel central : gradient non nul."""
    model = model.eval().float()
    x = torch.zeros(1, model.desc[0][0].in_channels, taille, taille, requires_grad=True)
    y = model(x)
    y[0, :, taille // 2, taille // 2].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1))
    cols = torch.nonzero(g.sum(0) > 0).ravel()
    return int(cols.max() - cols.min() + 1)


def appareil():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predire(model, X, dev) -> np.ndarray:
    """Carte complète en une passe : (80, H, W)."""
    f = model.facteur
    H, W = X.shape[1:]
    ph, pw = (-H) % f, (-W) % f
    x = torch.from_numpy(np.pad(X, ((0, 0), (0, ph), (0, pw)), mode="reflect"))[None].to(dev)
    model.eval()
    with torch.no_grad():
        y = model(x)[0, :, :H, :W]
    return y.float().cpu().numpy()


# =============================================================================
# Étape 4 — Entraînement
# =============================================================================
def entrainer(cfg) -> None:
    sc, nc = cfg["surrogate"], cfg["surrogate"]["reseau"]
    out = ROOT / sc["out_dir"]
    torch.manual_seed(cfg["project"]["random_seed"])
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    dev = appareil()

    X = entrees(cfg)
    d = np.load(out / "etiquettes.npz")
    forme_y = d["Y"].shape[1:]
    Ytr = etiquettes_raster(d, X.shape[1:], [0])
    Yva = etiquettes_raster(d, X.shape[1:], [1])
    va = ~np.isnan(Yva[0])

    model = UNetPotentiel(base=nc["base"], niveaux=nc["niveaux"],
                          dilatations=tuple(nc["dilatations"]))
    rf = champ_recepteur(model)
    marge = (rf + 1) // 2
    assert rf >= 2 * sc["rayon_m"] + 1, f"champ récepteur {rf} px < 301 px"
    assert nc["tuile_px"] > 2 * marge, "tuile trop petite pour le champ récepteur"
    model = model.to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{dev} ; {n_par / 1e6:.2f} M paramètres ; champ récepteur {rf} px ; "
          f"marge {marge} px ; tuile {nc['tuile_px']} px")

    tuiles = Tuiles(X, Ytr, nc["tuile_px"], marge, forme_y[-1], forme_y, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=nc["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=nc["lr"], total_steps=nc["iterations"],
                                                pct_start=0.05)
    hist, best = [], np.inf
    t0 = time.time()
    for it in range(1, nc["iterations"] + 1):
        model.train()
        xb, yb = tuiles.lot(nc["lot"])
        xb, yb = xb.to(dev), yb.to(dev)
        m = ~torch.isnan(yb)
        loss = F.mse_loss(model(xb)[m], yb[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % nc["eval_tous"] == 0 or it == nc["iterations"]:
            P = predire(model, X, dev)
            eam = float(np.nanmean(np.abs(P[:, va] - Yva[:, va])))
            hist.append(dict(iteration=it, perte=float(loss.detach()), eam_validation=eam,
                             temps_s=time.time() - t0))
            flag = ""
            if eam < best:
                best = eam
                torch.save({"etat": model.state_dict(), "config": nc, "rf": rf}, out / "modele.pt")
                flag = "  *"
            print(f"  it {it:5d}  perte {float(loss.detach()):.5f}  EAM validation {eam:.4f}  "
                  f"{time.time() - t0:5.0f} s{flag}", flush=True)
    pd.DataFrame(hist).to_csv(out / "historique.csv", index=False)
    print(f"Meilleure EAM de validation : {best:.4f} ; {time.time() - t0:.0f} s -> {out}")


# =============================================================================
# Étape 5 — Évaluation sur le bloc test et chronométrage
# =============================================================================
def chronometrer(model, X, dev, n=5) -> float:
    predire(model, X, dev)
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        predire(model, X, dev)
    return (time.perf_counter() - t0) / n


def evaluer(cfg) -> None:
    sc = cfg["surrogate"]
    out = ROOT / sc["out_dir"]
    ck = torch.load(out / "modele.pt", map_location="cpu")
    nc = ck["config"]
    model = UNetPotentiel(base=nc["base"], niveaux=nc["niveaux"],
                          dilatations=tuple(nc["dilatations"]))
    model.load_state_dict(ck["etat"])
    X = entrees(cfg)
    te = np.load(out / "test.npz")
    Yt = te["Y"]

    temps = {}
    for dev in {appareil(), torch.device("cpu")}:
        temps[dev.type] = chronometrer(model.to(dev), X, dev)
    dev = appareil()
    P = predire(model.to(dev), X, dev)
    n_pos = int((np.asarray(__import__("rasterio").open(out / "decoupage.tif").read(1)) != 255).sum())
    Pt = P[:, te["rows"], te["cols"]].T.reshape(Yt.shape)
    inter, xy = interieur_site(cfg, ROOT, te["rows"], te["cols"])
    res = rapport(Pt, Yt, te["strate"].astype(int), te["hauteurs"], surfaces_bins(cfg),
                  inter, xy, nom="U-Net (test)")
    res["cout_ms_position"] = 1e3 * temps[dev.type] / n_pos
    res.to_csv(out / "surrogate.csv", index=False)
    np.savez_compressed(out / "carte.npz", V=P.astype(np.float16))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    for k, t in temps.items():
        print(f"Carte complète ({X.shape[1]}×{X.shape[2]}, {n_pos:,} positions valides) sur {k} : "
              f"{t:.2f} s, soit {1e6 * t / n_pos:.1f} µs par position")


def main() -> None:
    cfg = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "entrainer"
    {"entrainer": entrainer, "evaluer": evaluer}[cmd](cfg)


if __name__ == "__main__":
    main()

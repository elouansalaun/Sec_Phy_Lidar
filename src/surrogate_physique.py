"""
src/surrogate_physique.py — U-Net informé par la physique, conditionnel en h (§8.8, étape 6.1).

Idée : le U-Net brut (src/surrogate.py) doit reconstruire l'occultation, un maximum le long de
rayons divisé par la distance, avec des convolutions 3 × 3 : il sous-apprend. On lui donne en
entrée les horizons denses de la pyramide log-polaire (src/horizons.py), calculés pour la hauteur
demandée. Le contexte lointain est alors porté par les entrées : un champ récepteur de ≈ 85 px
suffit, et le réseau corrige ce que les horizons par blocs ne disent pas (courbure exacte des rayons,
végétation traversée, forme fine du sol visible).

Entrée pour une hauteur h (166 canaux) :
  5 canaux de base (src.surrogate.entrees), h / 10 (constant),
  horizons à h : 2 (opaque, végétal) × 5 bandes × 8 secteurs,
  moyennes     : 2 (opacité, part de sol) × 5 bandes × 8 secteurs.
Sortie : 16 canaux = (anneau, secteur) à cette hauteur. Une passe par hauteur.
Augmentation D4 : les canaux indexés par secteur (entrées ET sorties) sont permutés avec le terrain.

Sorties (surrogate.out_dir) : modele_physique.pt, historique_physique.csv, surrogate_physique.csv,
                              carte_physique.npz
Usage : python -m src.surrogate_physique entrainer
        python -m src.surrogate_physique evaluer
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from src.eval_surrogate import interieur_site, rapport, surfaces_bins
from src.horizons import Horizons
from src.site import ROOT, load_config
from src.surrogate import UNetPotentiel, appareil, champ_recepteur, entrees, etiquettes_raster
from src.viewshed import prepare_surfaces


# =============================================================================
# Étape 1 — Entrées : base + horizons de la pyramide, pour toute la grille
# =============================================================================
class Entrees:
    def __init__(self, cfg):
        self.base = entrees(cfg)                                   # (5, H, W) float32
        s = prepare_surfaces(cfg)
        hz = Horizons(s, cfg)
        self.hor, self.moy, self.t_horizons = hz.cartes()          # float16
        self.t_horizons += hz.t_prep
        self.h = hz.h
        self.nh, self.ns = self.h.size, hz.ns
        self.c_in = (self.base.shape[0] + 1 + int(np.prod(self.hor.shape[1:-2]))
                     + int(np.prod(self.moy.shape[:-2])))              # 5 + 1 + 80 + 80 = 166

    @property
    def forme(self):
        return self.base.shape[1:]

    def tuile(self, ih, r, c, T):
        """Morceaux d'une tuile, chacun avec son axe secteur juste avant les axes spatiaux."""
        sl = (slice(r, r + T), slice(c, c + T))
        return [self.base[(slice(None), *sl)],                           # pas de secteur
                np.full((1, T, T), self.h[ih] / 10.0, np.float32),         # pas de secteur
                self.hor[ih][(Ellipsis, *sl)].astype(np.float32),          # (2, b, s, T, T)
                self.moy[(Ellipsis, *sl)].astype(np.float32)]              # (2, b, s, T, T)

    def fenetre(self, ih, r, c, T) -> np.ndarray:
        """Entrée (166, T, T) d'une fenêtre qui peut déborder de la grille (bords répliqués)."""
        H, W = self.forme
        rr = np.clip(np.arange(r, r + T), 0, H - 1)
        cc = np.clip(np.arange(c, c + T), 0, W - 1)
        r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
        x = concat(
            [self.base[:, r0:r1, c0:c1],
             np.full((1, r1 - r0, c1 - c0), self.h[ih] / 10.0, np.float32),
             self.hor[ih][..., r0:r1, c0:c1].astype(np.float32),
             self.moy[..., r0:r1, c0:c1].astype(np.float32)])
        return x[:, rr - r0][:, :, cc - c0]


def concat(morceaux) -> np.ndarray:
    return np.concatenate([m.reshape(-1, *m.shape[-2:]) for m in morceaux], axis=0)


# =============================================================================
# Étape 2 — Augmentation D4 sur des tableaux dont l'axe secteur précède les axes spatiaux
# =============================================================================
def transformer(a, k_rot: int, miroir: bool, ns: int, secteur: bool):
    a = np.rot90(a, k=-k_rot, axes=(-2, -1))
    perm = (np.arange(ns) + 2 * k_rot) % ns
    if miroir:
        a = a[..., ::-1]
        perm = (ns - perm) % ns
    if secteur:
        out = np.empty_like(a)
        out[..., perm, :, :] = a                                  # secteur s -> perm[s]
        a = out
    return np.ascontiguousarray(a)


class TuilesH:
    def __init__(self, E: Entrees, Y5, taille, marge, rng):
        """Y5 : (h, anneau, secteur, H, W) avec NaN hors étiquettes."""
        self.E, self.Y5, self.T, self.m, self.rng = E, Y5, taille, marge, rng
        self.cand = np.nonzero(~np.isnan(Y5[0, 0, 0]))
        self.H, self.W = E.forme

    def lot(self, n):
        xs, ys = [], []
        T, m, ns = self.T, self.m, self.E.ns
        for _ in range(n):
            i = self.rng.integers(self.cand[0].size)
            r = int(np.clip(self.cand[0][i] - self.rng.integers(m, T - m), 0, self.H - T))
            c = int(np.clip(self.cand[1][i] - self.rng.integers(m, T - m), 0, self.W - T))
            ih = int(self.rng.integers(self.E.nh))
            k, mi = int(self.rng.integers(4)), bool(self.rng.integers(2))
            x = [transformer(p, k, mi, ns, secteur=j >= 2)
                 for j, p in enumerate(self.E.tuile(ih, r, c, T))]
            y = self.Y5[ih, :, :, r:r + T, c:c + T].copy()
            y[..., :m, :] = y[..., T - m:, :] = np.nan
            y[..., :m] = y[..., T - m:] = np.nan
            y = transformer(y, k, mi, ns, secteur=True)
            xs.append(concat(x))
            ys.append(y.reshape(-1, T, T))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# =============================================================================
# Étape 3 — Prédiction par tuiles recouvrantes (mémoire bornée sur MPS)
# =============================================================================
def predire(model, E: Entrees, dev, marge: int, T: int = 512) -> np.ndarray:
    """-> (h, anneau·secteur, H, W). Tuiles de T px construites à la demande, chevauchement de
    `marge` px : seul l'intérieur de chaque tuile, où le champ récepteur est entier, est gardé."""
    assert T % model.facteur == 0
    H, W = E.forme
    pas = T - 2 * marge
    out = np.zeros((E.nh, 2 * E.ns, H, W), np.float32)
    model.eval()
    with torch.no_grad():
        for ih in range(E.nh):
            for r in range(0, H, pas):
                for c in range(0, W, pas):
                    x = torch.from_numpy(E.fenetre(ih, r - marge, c - marge, T))[None]
                    y = model(x.to(dev))[0].float().cpu().numpy()
                    h_, w_ = min(pas, H - r), min(pas, W - c)
                    out[ih, :, r:r + h_, c:c + w_] = y[:, marge:marge + h_, marge:marge + w_]
    return out


# =============================================================================
# Étape 4 — Entraînement
# =============================================================================
def charger_modele(nc, c_in):
    return UNetPotentiel(c_in=c_in, c_out=16, base=nc["base"], niveaux=nc["niveaux"],
                         dilatations=tuple(nc["dilatations"]))


def entrainer(cfg) -> None:
    sc, nc = cfg["surrogate"], cfg["surrogate"]["reseau_physique"]
    out = ROOT / sc["out_dir"]
    torch.manual_seed(cfg["project"]["random_seed"])
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    dev = appareil()

    E = Entrees(cfg)
    print(f"Horizons de toute la grille : {E.t_horizons:.1f} s ; {E.c_in} canaux d'entrée")
    d = np.load(out / "etiquettes.npz")
    fy = d["Y"].shape[1:]
    Ytr = etiquettes_raster(d, E.forme, [0]).reshape(*fy, *E.forme)
    Yva = etiquettes_raster(d, E.forme, [1]).reshape(*fy, *E.forme)
    va = ~np.isnan(Yva[0, 0, 0])

    model = charger_modele(nc, E.c_in)
    rf = champ_recepteur(model, 256)
    marge = (rf + 1) // 2
    assert nc["tuile_px"] > 2 * marge
    model = model.to(dev)
    print(f"{dev} ; {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M paramètres ; "
          f"champ récepteur {rf} px ; marge {marge} px ; tuile {nc['tuile_px']} px")

    tuiles = TuilesH(E, Ytr, nc["tuile_px"], marge, rng)
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
            P = predire(model, E, dev, marge).reshape(*fy, *E.forme)
            eam = float(np.mean(np.abs(P[..., va] - Yva[..., va])))
            hist.append(dict(iteration=it, perte=float(loss.detach()), eam_validation=eam,
                             temps_s=time.time() - t0))
            flag = ""
            if eam < best:
                best = eam
                torch.save({"etat": model.state_dict(), "config": nc, "rf": rf, "c_in": E.c_in},
                           out / "modele_physique.pt")
                flag = "  *"
            print(f"  it {it:5d}  perte {float(loss.detach()):.5f}  EAM validation {eam:.4f}  "
                  f"{time.time() - t0:5.0f} s{flag}", flush=True)
    pd.DataFrame(hist).to_csv(out / "historique_physique.csv", index=False)
    print(f"Meilleure EAM de validation : {best:.4f} ; {time.time() - t0:.0f} s -> {out}")


# =============================================================================
# Étape 5 — Évaluation et chronométrage (horizons + réseau)
# =============================================================================
def evaluer(cfg) -> None:
    sc = cfg["surrogate"]
    out = ROOT / sc["out_dir"]
    ck = torch.load(out / "modele_physique.pt", map_location="cpu")
    model = charger_modele(ck["config"], ck["c_in"])
    model.load_state_dict(ck["etat"])
    marge = (ck["rf"] + 1) // 2
    E = Entrees(cfg)
    te = np.load(out / "test.npz")
    Yt = te["Y"]
    n_pos = int((rasterio.open(out / "decoupage.tif").read(1) != 255).sum())

    temps = {}
    for dev in {appareil(), torch.device("cpu")}:
        m = model.to(dev)
        predire(m, E, dev, marge)                                  # échauffement
        t0 = time.perf_counter()
        P = predire(m, E, dev, marge)
        temps[dev.type] = time.perf_counter() - t0
    Pt = P.reshape(E.nh, 2, E.ns, *E.forme)[..., te["rows"], te["cols"]]
    Pt = np.moveaxis(Pt, -1, 0)
    inter, xy = interieur_site(cfg, ROOT, te["rows"], te["cols"])
    res = rapport(Pt, Yt, te["strate"].astype(int), te["hauteurs"], surfaces_bins(cfg),
                  inter, xy, nom="U-Net physique (test)")
    dev = appareil().type
    res["cout_us_position"] = 1e6 * (temps[dev] + E.t_horizons) / n_pos
    res.to_csv(out / "surrogate_physique.csv", index=False)
    np.savez_compressed(out / "carte_physique.npz", V=P.astype(np.float16))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"Horizons (toute la grille, CPU) : {E.t_horizons:.2f} s")
    for k, t in temps.items():
        print(f"Réseau, 5 hauteurs, sur {k} : {t:.2f} s ; total avec horizons "
              f"{1e6 * (t + E.t_horizons) / n_pos:.1f} µs par position ({n_pos:,} positions)")


def main() -> None:
    cfg = load_config()
    {"entrainer": entrainer, "evaluer": evaluer}[sys.argv[1] if len(sys.argv) > 1
                                                  else "entrainer"](cfg)


if __name__ == "__main__":
    main()

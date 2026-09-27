"""
src/demo.py — Logique de la démo Streamlit (app.py), sans interface : testable seule.

Deux usages :
  - consulter les solutions déjà certifiées (src.boucle, src.nsga) ;
  - évaluer en direct un dispositif modifié à la main : détection par cellule, couloirs de
    l'intrus (p*, p̃*), temps restant d(v), coût, couverture F. Une évaluation prend quelques
    dizaines de millisecondes : l'intrus « répond » à chaque capteur ajouté ou retiré.

Rien n'est optimisé ici : la démo réutilise les fonctions des modules de calcul.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.boucle import front_nsga
from src.instance import cost, from_configs, violations
from src.intrus import PROFILS, detection, exposition, load_domaine, seuil_couloir
from src.intrusion import config_key
from src.opportune import temps_restant


# =============================================================================
# Étape 1 — Chargement (une fois, mis en cache par l'application)
# =============================================================================
@dataclass
class Donnees:
    cfg: dict
    dom: object
    A: object
    conf: pd.DataFrame
    inst: object
    cles: np.ndarray
    w: np.ndarray
    grid: object
    perimetre: object
    actifs: object
    candidats: object
    fond: np.ndarray            # image RGB de la zone (ombrage + bâti)
    fenetre: tuple              # (r0, r1, c0, c1) : cadrage sur le domaine
    front: pd.DataFrame
    solutions: pd.DataFrame


def charger(root, site: str | None = None) -> Donnees:
    """site : dossier de data/sites/ ; "" = site de référence ; None = variable LIDAR_SITE."""
    import geopandas as gpd

    from src.rasters import load_grid
    from src.viewshed import read
    from src.site import load_config

    cfg = load_config(root / "config.yaml", site=site)
    dom, A, conf = load_domaine(cfg, root)
    cells = np.load(root / cfg["coverage"]["out_dir"] / "cellules.npz")
    n_int = int(dom.interieur.sum())
    inst = from_configs(A[:, :n_int], cells["w"], conf, cfg)
    grid = load_grid(root / cfg["site"]["grid_yaml"])
    per = gpd.read_file(root / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    actifs = gpd.read_file(root / cfg["site"]["output_gpkg"], layer="actifs")
    cand = gpd.read_file(root / cfg["candidates"]["output"])

    r, c = dom.rows, dom.cols
    fen = (max(r.min() - 5, 0), min(r.max() + 6, grid.height),
           max(c.min() - 5, 0), min(c.max() + 6, grid.width))
    d = root / cfg["rasters"]["out_dir"]
    mnt, bati = read(d / "mnt.tif"), read(d / "bati_h.tif")
    fond = image_fond(mnt, bati)[fen[0]:fen[1], fen[2]:fen[3]]

    front = pd.read_csv(root / cfg["nsga"]["out_dir"] / "front_global.csv")
    front["T"] = [np.array(str(s).split(), np.int64) for s in front["configs"]]
    front = front.drop_duplicates("configs").reset_index(drop=True)
    f_sol = root / cfg["boucle"]["out_dir"] / "solutions.csv"
    try:                                # absent, ou vide si aucun optimum n'a été certifié
        sol = pd.read_csv(f_sol)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        sol = pd.DataFrame(columns=["theta", "T_R", "fusion", "config"])
    return Donnees(cfg, dom, A, conf, inst, config_key(conf).to_numpy(), cells["w"], grid, per,
                   actifs, cand, fond, fen, front, sol)


def image_fond(mnt: np.ndarray, bati: np.ndarray) -> np.ndarray:
    """Ombrage du relief (soleil au nord-ouest) et bâti en gris : repère visuel neutre."""
    z = np.nan_to_num(mnt + np.nan_to_num(bati), nan=float(np.nanmean(mnt)))
    gy, gx = np.gradient(z)
    az, alt = np.radians(315), np.radians(45)
    pente = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    ombre = np.sin(alt) * np.cos(pente) + np.cos(alt) * np.sin(pente) * np.cos(az - aspect)
    v = 0.80 + 0.15 * np.clip(ombre, 0, 1)
    img = np.stack([v, v, v], axis=-1)
    img[np.nan_to_num(bati) > 0] = (0.62, 0.61, 0.59)
    return (255 * img).astype(np.uint8)


# =============================================================================
# Étape 2 — Évaluation d'un dispositif
# =============================================================================
def evaluer(D: Donnees, T, ic: dict, T_R: float) -> dict:
    """Tout ce que la démo affiche pour un dispositif T et des paramètres d'intrus ic."""
    T = np.unique(np.asarray(T, np.int64))
    P, sigma = detection(D.A, T)
    n_int = int(D.dom.interieur.sum())
    out = dict(T=T, P=P, cout=cost(D.inst, T) if T.size else 0.0,
               F=float(D.w @ P[:n_int]) / float(D.w.sum()),
               violations=violations(D.inst, T, 1e9) if T.size else [])
    for pr in PROFILS:
        d = temps_restant(D.dom, ic, pr)
        p, couloir = seuil_couloir(D.dom, P, pr)
        pt, couloir_t = seuil_couloir(D.dom, np.where(d >= T_R, P, 0.0), pr)
        lam, _ = exposition(D.dom, sigma, pr, ic, ic["tau_obs_s"])
        out.update({f"d_{pr}": d, f"T_min_{pr}": float(d[D.dom.source].min()),
                    f"p_star_{pr}": p, f"couloir_{pr}": couloir,
                    f"p_tilde_{pr}": pt, f"couloir_opp_{pr}": couloir_t,
                    f"P_chemin_{pr}": float(max(-np.expm1(-lam), 0.0))})
    return out


def decrire(D: Donnees, T) -> pd.DataFrame:
    """Tableau lisible d'un dispositif."""
    T = np.unique(np.asarray(T, np.int64))
    if T.size == 0:
        return pd.DataFrame(columns=["config", "emplacement", "support", "h (m)", "capteur",
                                     "azimut", "inclinaison", "coût capteur"])
    c = D.conf.iloc[T]
    return pd.DataFrame({
        "config": T, "emplacement": c["cand"].to_numpy(), "support": c["support"].to_numpy(),
        "h (m)": c["h"].to_numpy(), "capteur": c["capteur"].to_numpy(),
        "azimut": c["azimut"].map(lambda a: "360°" if pd.isna(a) else f"{a:g}°").to_numpy(),
        "inclinaison": c["inclinaison"].map(lambda a: f"{a:.0f}°").to_numpy(),
        "coût capteur": c["cout_capteur"].to_numpy()})


# =============================================================================
# Étape 3 — Solutions de référence
# =============================================================================
def solutions_certifiees(D: Donnees) -> pd.DataFrame:
    """Une ligne par optimum certifié de src.boucle : θ, T_R, fusion, coût, configurations."""
    rows = []
    for (th, T_R, fu), g in D.solutions.groupby(["theta", "T_R", "fusion"]):
        T = g["config"].to_numpy(np.int64)
        rows.append(dict(theta=th, T_R=T_R, fusion=fu, cout=cost(D.inst, T), T=T))
    return pd.DataFrame(rows, columns=["theta", "T_R", "fusion", "cout", "T"])


def meilleure_du_front(D: Donnees, budget: float, critere: str) -> pd.Series | None:
    """Solution du front NSGA-II de coût ≤ budget, meilleure selon « F » ou « p*_F »."""
    f = D.front[D.front["cout"] <= budget + 1e-9]
    if f.empty:
        return None
    cles = ["F", "p_star_F"] if critere == "F" else ["p_star_F", "F"]
    return f.sort_values(cles, ascending=False).iloc[0]


def configurations(D: Donnees, cand: str, h: float, capteur: str) -> pd.DataFrame:
    """Configurations disponibles pour un emplacement, une hauteur et un capteur."""
    c = D.conf
    m = (c["cand"] == cand) & np.isclose(c["h"], h) & (c["capteur"] == capteur)
    return c[m]


# =============================================================================
# Étape 4 — Carte (plotly)
# =============================================================================
C_F, C_G, C_OPP, INK = "#eb6834", "#4a3aa7", "#b8456a", "#0b0b0b"


def carte(D: Donnees, res: dict, T_R: float, montrer_zone_morte: bool = True,
          profil_opp: str = "F", hauteur: int = 640):
    import plotly.graph_objects as go
    import shapely

    g, (r0, r1, c0, c1) = D.grid, D.fenetre
    x0, y0 = g.xmin + c0 * g.res, g.ymax - r0 * g.res

    def xy(rows, cols):
        return g.xmin + (cols + 0.5) * g.res - x0, g.ymax - (rows + 0.5) * g.res - y0

    fig = go.Figure()
    fig.add_trace(go.Image(z=D.fond, x0=0.5, dx=g.res, y0=-0.5, dy=-g.res, hoverinfo="skip"))

    img = np.full((r1 - r0, c1 - c0), np.nan)
    img[D.dom.rows - r0, D.dom.cols - c0] = res["P"]
    fig.add_trace(go.Heatmap(z=img, x0=0.5, dx=g.res, y0=-0.5, dy=-g.res, zmin=0, zmax=1,
                             colorscale=[[0, "rgba(42,120,214,0.05)"], [1, "rgba(42,120,214,0.85)"]],
                             colorbar=dict(title="P détection", thickness=12, len=0.6),
                             hovertemplate="P = %{z:.2f}<extra></extra>"))
    if montrer_zone_morte and T_R > 0:
        dz = np.full_like(img, np.nan)
        d = res[f"d_{profil_opp}"]
        morte = d < T_R
        dz[D.dom.rows[morte] - r0, D.dom.cols[morte] - c0] = 1.0
        fig.add_trace(go.Heatmap(z=dz, x0=0.5, dx=g.res, y0=-0.5, dy=-g.res, showscale=False,
                                 colorscale=[[0, "rgba(120,120,120,0.35)"],
                                             [1, "rgba(120,120,120,0.35)"]],
                                 hovertemplate="détection trop tardive (d < T_R)<extra></extra>"))

    def ligne(geom, **kw):
        xs, ys = shapely.get_coordinates(geom).T
        fig.add_trace(go.Scatter(x=xs - x0, y=ys - y0, mode="lines", hoverinfo="skip",
                                 showlegend=False, **kw))

    for geom, nom in zip(D.actifs.geometry, D.actifs["nom"]):
        xs, ys = shapely.get_coordinates(geom.exterior).T
        fig.add_trace(go.Scatter(x=xs - x0, y=ys - y0, mode="lines", fill="toself",
                                 fillcolor="rgba(214,212,207,0.9)", line=dict(color="#52514e", width=1),
                                 name=nom, hovertemplate=f"{nom}<extra></extra>", showlegend=False))
    ligne(D.perimetre.exterior, line=dict(color=INK, width=2))

    for key, col, nom, dash in ((f"couloir_G", C_G, "Couloir opportuniste (G)", "solid"),
                                (f"couloir_F", C_F, "Couloir déterminé (F)", "solid"),
                                (f"couloir_opp_{profil_opp}", C_OPP,
                                 f"Couloir opportun ({profil_opp}, T_R = {T_R:g} s)", "dot")):
        p = res[key]
        if key.startswith("couloir_opp") and T_R == 0:
            continue
        x, y = xy(D.dom.rows[p], D.dom.cols[p])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="white", width=5),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nom,
                                 line=dict(color=col, width=2.5, dash=dash), hoverinfo="skip"))

    T = res["T"]
    if T.size:
        c = D.conf.iloc[T].merge(D.candidats[["id", "geometry"]], left_on="cand", right_on="id")
        px = np.array([g_.x for g_ in c["geometry"]]) - x0
        py = np.array([g_.y for g_ in c["geometry"]]) - y0
        radar = c["capteur"].str.startswith("radar").to_numpy()
        txt = [f"{a} · {b} · h = {h:g} m" + ("" if pd.isna(z) else f" · azimut {z:g}°")
               for a, b, h, z in zip(c["cand"], c["capteur"], c["h"], c["azimut"])]
        fig.add_trace(go.Scatter(x=px[radar], y=py[radar], mode="markers", name="Radar",
                                 marker=dict(symbol="circle", size=13, color="#1f9e6e",
                                             line=dict(color="white", width=2)),
                                 text=np.array(txt)[radar], hovertemplate="%{text}<extra></extra>"))
        cam = ~radar
        if cam.any():
            az = np.radians(c["azimut"].to_numpy()[cam].astype(float))
            L = 12.0                                    # flèche de visée (azimut depuis le nord)
            for xa, ya, a in zip(px[cam], py[cam], az):
                fig.add_trace(go.Scatter(x=[xa, xa + L * np.sin(a)], y=[ya, ya + L * np.cos(a)],
                                         mode="lines", line=dict(color="#0b0b0b", width=2),
                                         hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=px[cam], y=py[cam], mode="markers", name="Caméra",
                                     marker=dict(symbol="square", size=10, color="#0b0b0b",
                                                 line=dict(color="white", width=2)),
                                     text=np.array(txt)[cam], hovertemplate="%{text}<extra></extra>"))

    W, H = (c1 - c0) * g.res, (r1 - r0) * g.res
    fig.update_xaxes(range=[0, W], visible=False, constrain="domain")
    fig.update_yaxes(range=[-H, 0], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=hauteur, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.08, x=0, font=dict(size=11)))
    return fig

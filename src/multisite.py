"""
src/multisite.py — Même chaîne de calcul sur plusieurs sites dessinés dans QGIS, puis comparaison.

Un site est un dossier, n'importe où sur le disque (ou data/sites/<nom>/ dans le dépôt) :
    site.gpkg ou site/site.gpkg   dessiné dans QGIS (perimetre, zones, acces) — jamais modifié
    *.LAZ                         facultatif : tuiles du site (sinon celles de data/raw)
    config.yaml                   facultatif : surcharges propres au site
Toutes les sorties vont dans <dossier>/sorties/ (site/, processed/ avec logs/, figures/).
Le site de référence (data/site, data/processed) n'est jamais modifié.

Usage :
    python -m src.multisite lancer ~/Desktop/SITES_QGIS/SITE2 [--depuis ETAPE] [--jusqua ETAPE]
    python -m src.multisite lancer <nom> … | --tous       (sites de data/sites et sites déjà lancés)
    python -m src.multisite etat                            avancement par site
    python -m src.multisite comparer [--sans-reference]     tableau et figures
    python -m src.multisite nouveau <nom> [--copier <site|reference>]   gabarit QGIS vide

Le mécanisme : la variable d'environnement LIDAR_SITE=<nom ou chemin> est lue par src.site.load_config,
qui fusionne les surcharges et déplace les chemins. Chaque étape reste `python -m src.<module>`,
inchangée ; ce module ne fait qu'enchaîner, journaliser et comparer.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.site import ENV_SITE, ROOT, dossier_site, load_config

REFERENCE = "reference"          # pseudo-site : data/site et data/processed (config.yaml brut)


# =============================================================================
# Étape 1 — Les étapes de la chaîne et leur sortie témoin
# =============================================================================
@dataclass(frozen=True)
class Etape:
    nom: str
    module: str
    args: tuple
    temoin: callable            # cfg -> Path : sortie dont l'existence et la date font foi
    note: str


def _o(cfg, bloc, fichier):
    return ROOT / cfg[bloc]["out_dir"] / fichier


ETAPES = (
    Etape("preparer", "src.multisite", ("_preparer",), lambda c: ROOT / c["site"]["gpkg"],
          "copie contrôlée du site QGIS (SCR, type des zones)"),
    Etape("site", "src.site", (), lambda c: ROOT / c["site"]["grid_yaml"],
          "validation du site, zones pondérées, grille"),
    Etape("rasters", "src.rasters", (), lambda c: _o(c, "rasters", "mnt.tif"),
          "MNT, MNS, bâti, végétation (tuiles du dossier du site, sinon data/raw)"),
    Etape("candidats", "src.candidates", (), lambda c: ROOT / c["candidates"]["output"],
          "positions candidates"),
    Etape("couverture", "src.coverage", (), lambda c: _o(c, "coverage", "A.npz"),
          "visibilité exacte, matrice de couverture"),
    Etape("glouton", "src.greedy", (), lambda c: _o(c, "optimisation", "greedy.csv"),
          "glouton, balayage du budget"),
    Etape("milp", "src.milp", (), lambda c: _o(c, "optimisation", "milp.csv"),
          "optimum encadré par coupes sous-modulaires"),
    Etape("pareto", "src.pareto", (), lambda c: _o(c, "optimisation", "pareto.csv"),
          "front couverture / budget"),
    Etape("analyse", "src.analyse", (), lambda c: _o(c, "optimisation", "zones.csv"),
          "cartes des solutions"),
    Etape("intrusion", "src.intrusion", (), lambda c: _o(c, "intrus", "A_intrus.npz"),
          "domaine de l'intrus, graphe, A_intrus"),
    Etape("intrus", "src.intrus", (), lambda c: _o(c, "intrus", "chemins_milp.png"),
          "couloirs de l'intrus"),
    Etape("eval_intrus", "src.evaluation_intrus", (), lambda c: _o(c, "intrus", "semaine3.csv"),
          "solutions de couverture face à l'intrus"),
    Etape("nsga", "src.nsga", (),
          lambda c: _o(c, "nsga", f"front_{c['nsga']['graines'][-1]}.csv"),
          "NSGA-II coût / couverture / pire cas"),
    Etape("analyse_nsga", "src.analyse_nsga", (), lambda c: _o(c, "nsga", "front_global.csv"),
          "front global, validation contre les bornes"),
    Etape("opportune", "src.opportune", (), lambda c: _o(c, "intrus", "opportune_stats.csv"),
          "temps d'intrusion, détection opportune"),
    Etape("boucle", "src.boucle", ("cout",), lambda c: _o(c, "boucle", "cout_robuste.csv"),
          "coût minimal certifié c(θ, T_R)"),
)
NOMS = [e.nom for e in ETAPES]


def dossier_sites(cfg=None) -> Path:
    cfg = cfg or load_config(site="")
    return ROOT / cfg["multisite"]["dossier"]


def _registre() -> Path:
    """Sites extérieurs déjà lancés (chemins propres à cette machine, non versionnés)."""
    return ROOT / load_config(site="")["multisite"]["out_dir"] / "sites_externes.txt"


def identifiant(site: str) -> str:
    """Nom d'un site de data/sites, ou chemin absolu d'un dossier extérieur."""
    if site == REFERENCE:
        return site
    d = dossier_site(site, ROOT, load_config(site="")["multisite"]["dossier"])
    return d.name if d.parent == dossier_sites() else str(d)


def enregistrer(site: str) -> None:
    if os.sep in site:
        f = _registre()
        f.parent.mkdir(parents=True, exist_ok=True)
        deja = f.read_text(encoding="utf-8").split("\n") if f.exists() else []
        if site not in deja:
            f.write_text("\n".join([x for x in deja if x] + [site]) + "\n", encoding="utf-8")


def sites_connus() -> list[str]:
    d = dossier_sites()
    internes = sorted(p.name for p in d.glob("*") if (p / "site.gpkg").exists()
                      or (p / "site" / "site.gpkg").exists()) if d.is_dir() else []
    f = _registre()
    externes = [x for x in f.read_text(encoding="utf-8").split("\n") if x and Path(x).is_dir()] \
        if f.exists() else []
    return internes + externes


def config_site(nom: str) -> dict:
    return load_config(site="" if nom == REFERENCE else nom)


def etiquette(nom: str) -> str:
    return "référence" if nom == REFERENCE else config_site(nom)["site_nom"]


def statut(nom: str) -> list[str]:
    """Pour chaque étape : 'ok', 'perime' (plus ancienne qu'une entrée) ou 'absent'."""
    cfg = config_site(nom)
    if nom == REFERENCE:
        t_min = 0.0
    else:
        entrees = [Path(cfg["site"]["gpkg_source"]), Path(cfg["site_dossier"]) / "config.yaml"]
        t_min = max(p.stat().st_mtime for p in entrees if p.exists())
    out = []
    for e in ETAPES:
        t = e.temoin(cfg)
        if not t.exists():
            out.append("absent")
            t_min = np.inf
        elif nom != REFERENCE and t.stat().st_mtime < t_min:   # référence : existence seule
            out.append("perime")
            t_min = np.inf               # tout l'aval doit être recalculé
        else:
            out.append("ok")
            t_min = t.stat().st_mtime
    return out


# =============================================================================
# Étape 2 — Nouveau site : gabarit GeoPackage à ouvrir dans QGIS
# =============================================================================
def nouveau(nom: str, copier: str | None) -> None:
    import geopandas as gpd
    import shapely

    base = load_config(site="")
    d = dossier_sites(base) / nom
    gpkg = d / "site.gpkg"
    if gpkg.exists():
        sys.exit(f"{gpkg} existe déjà : rien n'est écrasé.")
    d.mkdir(parents=True, exist_ok=True)
    crs = base["lidar"]["crs"]

    if copier:
        c = config_site(copier)
        src = ROOT / c["site"].get("gpkg_source", c["site"]["gpkg"])
        for layer in ("perimetre", "zones", "acces"):
            gpd.read_file(src, layer=layer).to_file(gpkg, layer=layer)
    else:
        vide = lambda champs, geom: gpd.GeoDataFrame(
            {k: pd.Series(dtype=t) for k, t in champs.items()},
            geometry=gpd.GeoSeries([], crs=crs), crs=crs)
        vide({"nom": str, "type_cloture": str}, "Polygon").to_file(
            gpkg, layer="perimetre", geometry_type="Polygon")
        vide({"nom": str, "niveau": "int32", "poids": float, "cible_intrus": bool, "type": str},
             "Polygon").to_file(gpkg, layer="zones", geometry_type="Polygon")
        vide({"nom": str, "type": str}, "Point").to_file(gpkg, layer="acces",
                                                        geometry_type="Point")

    # Cadre autorisé : le site + study_buffer_m doit rester dans les données LiDAR
    x0, y0, x1, y1 = base["lidar"]["data_extent"]
    b = base["site"]["study_buffer_m"]
    gpd.GeoDataFrame({"nom": ["cadre_autorise"]}, geometry=[shapely.box(x0 + b, y0 + b, x1 - b, y1 - b)],
                     crs=crs).to_file(gpkg, layer="cadre_autorise")

    (d / "config.yaml").write_text(
        f"# Surcharges propres au site « {nom} » (facultatif), fusionnées après multisite.profil.\n"
        "# Exemples :\n"
        "# rasters:\n"
        "#   tiles_glob: \"data/raw/autre_secteur/*.LAZ\"   # tuiles d'un autre secteur\n"
        "# lidar:\n"
        "#   data_extent: [xmin, ymin, xmax, ymax]         # emprise de ces tuiles\n"
        "# intrus:\n"
        "#   t_portail_s: 5.0\n", encoding="utf-8")
    print(f"Gabarit créé : {gpkg}\n"
          f"  couches perimetre, zones, acces{' (copiées de ' + copier + ')' if copier else ' (vides)'}"
          f" + cadre_autorise (à ne pas dépasser)\n"
          f"Ouvre-le dans QGIS, dessine le site, puis : python -m src.multisite lancer {nom}")


# =============================================================================
# Étape 2 bis — Préparer : copie contrôlée du site dessiné (l'original n'est jamais modifié)
# =============================================================================
def preparer() -> None:
    """Site QGIS -> <sorties>/site/site.gpkg, dans le SCR du projet, avec le champ type.

    - couche dans un autre SCR (souvent acces en EPSG:4326) : reprojetée ;
    - zones sans champ type : déduit du nuage LiDAR (part de points bâtiment ≥ 50 % -> batiment,
      sinon peripherie), tableau imprimé pour contrôle ; un champ type rempli dans QGIS prime.
    """
    import geopandas as gpd

    cfg = load_config()
    src, dst = Path(cfg["site"]["gpkg_source"]), ROOT / cfg["site"]["gpkg"]
    crs = cfg["lidar"]["crs"]
    couches = {}
    for nom in ("perimetre", "zones", "acces"):
        g = gpd.read_file(src, layer=nom)
        if g.crs is None:
            raise SystemExit(f"{nom} : aucun SCR défini dans {src}")
        if g.crs.to_epsg() != int(crs.split(":")[1]):
            print(f"{nom} : SCR EPSG:{g.crs.to_epsg()} -> reprojetée en {crs}")
            g = g.to_crs(crs)
        couches[nom] = g
    z = couches["zones"]
    if "type" not in z or z["type"].isna().any() or (z["type"].astype(str).str.strip() == "").any():
        part = part_bati(couches["perimetre"].geometry.iloc[0], z, cfg)
        deduit = np.where(part >= 0.5, "Batiment", "Peripherie")
        z["type"] = z["type"].where(z["type"].notna() & (z["type"].astype(str).str.strip() != ""),
                                    deduit) if "type" in z else deduit
        print("\nChamp « type » absent ou incomplet : déduit du LiDAR (à vérifier)")
        print(pd.DataFrame({"zone": z["nom"], "points_bati_pct": np.round(100 * part, 1),
                            "type": z["type"], "cible_intrus": z["cible_intrus"]}).to_string(index=False))
        cible = z.loc[z["cible_intrus"].fillna(False).astype(bool), "type"].str.lower()
        if (cible == "peripherie").any():
            raise SystemExit("La zone cible n'est pas couverte par du bâti : ajoute le champ type "
                             "(Batiment ou Equipement) dans QGIS.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    for nom, g in couches.items():
        g.to_file(dst, layer=nom)
    print(f"\n{src}\n  -> {dst}")


def part_bati(perimetre, zones, cfg) -> np.ndarray:
    """Part des points classés bâtiment dans chaque zone (tuiles qui recoupent le site seulement)."""
    import glob

    import laspy
    import shapely

    x0, y0, x1, y1 = perimetre.bounds
    classes = cfg["rasters"]["classes"]["batiment"]
    X, Y, B = [], [], []
    for f in sorted(glob.glob(str(ROOT / cfg["rasters"]["tiles_glob"]))):
        if f.lower().endswith(".copc.laz"):
            continue
        with laspy.open(f) as r:
            h = r.header
            if h.maxs[0] < x0 or h.mins[0] > x1 or h.maxs[1] < y0 or h.mins[1] > y1:
                continue
            for pts in r.chunk_iterator(5_000_000):
                m = (pts.x >= x0) & (pts.x <= x1) & (pts.y >= y0) & (pts.y <= y1)
                X.append(np.asarray(pts.x[m])); Y.append(np.asarray(pts.y[m]))
                B.append(np.isin(np.asarray(pts.classification[m]), classes))
    if not X:
        raise SystemExit("aucune tuile LAZ ne recouvre le site")
    P = shapely.points(np.concatenate(X), np.concatenate(Y))
    B = np.concatenate(B)
    out = []
    for geom in zones.geometry:
        k = shapely.contains(geom, P)
        out.append(B[k].mean() if k.any() else 0.0)
    return np.array(out)


# =============================================================================
# Étape 3 — Lancer la chaîne sur un ou plusieurs sites
# =============================================================================
def lancer(nom: str, depuis: str | None, jusqua: str | None) -> bool:
    cfg = config_site(nom)
    logs = ROOT / cfg["paths"]["processed"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    i0 = NOMS.index(depuis) if depuis else 0
    i1 = NOMS.index(jusqua) if jusqua else len(ETAPES) - 1
    etat = statut(nom)
    env = {**os.environ, ENV_SITE: nom, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    enregistrer(nom)
    print(f"\n=== Site « {cfg['site_nom']} » ({cfg['site_dossier']}) ===\n"
          f"    sorties : {cfg['site_sorties']}")
    force = depuis is not None
    for i, e in enumerate(ETAPES[: i1 + 1]):
        if i < i0:
            continue
        if etat[i] == "ok" and not force:
            print(f"  [{i + 1:2d}] {e.nom:<13} à jour")
            continue
        force = True                        # une étape recalculée invalide tout l'aval
        journal = logs / f"{i + 1:02d}_{e.nom}.log"
        print(f"  [{i + 1:2d}] {e.nom:<13} {e.note} … ", end="", flush=True)
        t0 = time.time()
        with open(journal, "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable, "-m", e.module, *e.args], cwd=ROOT, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"ÉCHEC ({dt:.0f} s) — voir {journal}")
            print("      " + "\n      ".join(journal.read_text(encoding="utf-8").splitlines()[-8:]))
            return False
        print(f"{dt:.0f} s")
    return True


def afficher_etat() -> None:
    noms = [REFERENCE] + sites_connus()
    sym = {"ok": "✔", "perime": "~", "absent": "·"}
    t = pd.DataFrame({etiquette(n): [sym[s] for s in statut(n)] for n in noms}, index=NOMS).T
    print(t.to_string())
    print("\n✔ à jour   ~ périmé (une entrée est plus récente)   · absent")


# =============================================================================
# Étape 4 — Comparaison : indicateurs par site
# =============================================================================
def descripteurs(cfg) -> dict:
    """Ce qui distingue les sites avant tout calcul : taille, obstacles, relief, catalogue."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask

    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre")
    P = per.geometry.iloc[0]
    derive = ROOT / cfg["site"]["output_gpkg"]
    actifs = gpd.read_file(derive, layer="actifs")
    acces = gpd.read_file(derive, layer="acces")
    cible = actifs.loc[actifs["cible_intrus"].astype(bool)].geometry.iloc[0]
    zone = P.buffer(cfg["intrus"]["bande_ext_m"])      # domaine de l'intrus : site + bande
    rd = ROOT / cfg["rasters"]["out_dir"]
    with rasterio.open(rd / "mnt.tif") as s:
        mnt, tr, shape = s.read(1, masked=True).filled(np.nan), s.transform, s.shape
    with rasterio.open(rd / "bati_mask.tif") as s:
        bati = s.read(1) == 1
    with rasterio.open(rd / "autre_h.tif") as s:
        veg = s.read(1, masked=True).filled(0)
    m = ~geometry_mask([zone], shape, tr)
    z = mnt[m]
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    conf = pd.read_csv(ROOT / cfg["intrus"]["out_dir"] / "configs_intrus.csv")
    return dict(site=per["nom"].iloc[0], surface_ha=P.area / 1e4, cloture_m=P.length,
                n_actifs=len(actifs), n_acces=len(acces),
                dist_cible_cloture_m=float(cible.distance(P.boundary)),
                bati_pct=100 * bati[m].mean(), vegetation_pct=100 * ((veg[m] > 2) & ~bati[m]).mean(),
                denivele_m=float(np.nanpercentile(z, 95) - np.nanpercentile(z, 5)),
                n_candidats=len(cand), n_configurations=len(conf))


def lire_csv(path: Path, colonnes: list[str]) -> pd.DataFrame:
    """CSV absent ou vide (aucune solution) -> tableau vide aux bonnes colonnes."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=colonnes)


def resultats(cfg) -> dict:
    mc = cfg["multisite"]["comparaison"]
    o = ROOT / cfg["optimisation"]["out_dir"]
    par = pd.read_csv(o / "pareto.csv").set_index("budget")
    s3 = pd.read_csv(ROOT / cfg["intrus"]["out_dir"] / "semaine3.csv")
    milp = s3[s3.methode == "milp"]
    front = pd.read_csv(ROOT / cfg["nsga"]["out_dir"] / "front_global.csv")
    opp = pd.read_csv(ROOT / cfg["intrus"]["out_dir"] / "opportune_stats.csv").set_index("profil")
    r = dict(F_B5=par.lb.get(5.0, np.nan), F_B10=par.lb.get(10.0, np.nan),
             F_B15=par.lb.get(15.0, np.nan),
             ecart_certifie_max_pct=100 * par.ecart_certifie.max(),
             perte_glouton_max_pt=100 * par.perte_glouton.max(),
             F_aveugle_max=float(milp.loc[milp.p_star_F == 0, "F"].max()) if (milp.p_star_F == 0).any() else 0.0,
             T_min_s=float(opp.loc["F", "T_min"]))
    for s in mc["seuils_p_star"]:
        ok = front[front.p_star_F >= s - 1e-9]
        r[f"cout_p{s:g}"] = float(ok.cout.min()) if len(ok) else np.nan
    f = ROOT / cfg["boucle"]["out_dir"] / "cout_robuste.csv"
    cr = lire_csv(f, ["theta", "T_R", "statut_max", "cout_max"])
    for T_R in mc["t_reponse_s"]:
        k = cr[np.isclose(cr.theta, mc["theta"]) & np.isclose(cr.T_R, T_R)]
        c = float(k.cout_max.iloc[0]) if len(k) else np.nan
        r[f"cout_certifie_TR{T_R:g}"] = c if np.isfinite(c) else np.nan    # inf : non trouvé
        r[f"statut_TR{T_R:g}"] = k.statut_max.iloc[0] if len(k) else "non calculé"
    return r


def courbes(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F*(B) encadré et meilleur p*_F du front à coût ≤ B."""
    par = pd.read_csv(ROOT / cfg["optimisation"]["out_dir"] / "pareto.csv")
    front = pd.read_csv(ROOT / cfg["nsga"]["out_dir"] / "front_global.csv")
    B = np.arange(2.0, float(cfg["optimisation"]["budget"]["max"]) + 0.01, 0.1)
    p = [front.loc[front.cout <= b + 1e-9, "p_star_F"].max() for b in B]
    return par[["budget", "lb", "ub"]], pd.DataFrame({"budget": B, "p_star": np.nan_to_num(p)})


# =============================================================================
# Étape 5 — Figures de comparaison
# =============================================================================
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e2dd"
# Ordre catégoriel fixe (validé : CVD adjacent ΔE ≥ 9) ; la référence en encre neutre.
COULEURS = ("#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#b8456a")


def couleurs(noms):
    autres = [n for n in noms if n != REFERENCE]
    if len(autres) > len(COULEURS):
        raise ValueError(f"{len(autres)} sites : au plus {len(COULEURS)} (+ référence) par figure")
    return {n: (INK if n == REFERENCE else COULEURS[autres.index(n)]) for n in noms}


def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8)


def figure_courbes(donnees, etiq, col, path):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, layout="constrained")
    for n, (par, ps) in donnees.items():
        a1.fill_between(par.budget, 100 * par.lb, 100 * par.ub, color=col[n], alpha=0.15, lw=0)
        a1.plot(par.budget, 100 * par.lb, color=col[n], lw=2, label=etiq[n])
        a2.step(ps.budget, ps.p_star, where="post", color=col[n], lw=2, label=etiq[n])
        a1.annotate(etiq[n], (par.budget.iloc[-1], 100 * par.lb.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, color=INK_2, va="center")
    a1.set(xlabel="Budget B (unités)", ylabel="Couverture pondérée F* (%)", ylim=(0, 100))
    a1.set_title("Couvrir : optimum encadré [LB, UB]", loc="left", fontsize=11, color=INK)
    a2.set(xlabel="Budget B (unités)", ylabel="p*_F, détection garantie sur le pire couloir",
           ylim=(0, 1.02))
    a2.set_title("Protéger : meilleur p*_F du front à coût ≤ B", loc="left", fontsize=11, color=INK)
    for a in (a1, a2):
        _style(a)
    a2.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_exigences(tab, etiq, col, mc, path):
    """Coût minimal certifié par site, pour chaque temps de réponse, avec T_min en regard."""
    import matplotlib.pyplot as plt

    trs = mc["t_reponse_s"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.45 * len(tab) + 1.6), dpi=150,
                                 layout="constrained", gridspec_kw={"width_ratios": [2, 1]},
                                 sharey=True)
    y = np.arange(len(tab))[::-1]
    marq = ("o", "D", "s", "^")
    for (n, r), yi in zip(tab.iterrows(), y):
        vals = [r[f"cout_certifie_TR{t:g}"] for t in trs]
        a1.plot([v for v in vals if np.isfinite(v)], [yi] * sum(np.isfinite(vals)), color=GRID,
                lw=2, zorder=1)
        manque = []
        for t, v, m in zip(trs, vals, marq):
            if np.isfinite(v):
                a1.plot(v, yi, marker=m, ms=8, color=col[n], mec="white", mew=1.5, zorder=3)
                a1.annotate(f"{v:.1f}", (v, yi), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7, color=INK_2)
            else:
                manque.append(f"T_R = {t:g} s : {r[f'statut_TR{t:g}']}")
        if manque:
            a1.annotate("non certifié — " + " ; ".join(manque), (0, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK_3)
        a2.barh(yi, r["T_min_s"], color=col[n], height=0.35)
        a2.annotate(f"{r['T_min_s']:.0f} s", (r["T_min_s"], yi), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK_2)
    a1.set_yticks(y, [etiq[n] for n in tab.index])
    a1.set_xlabel(f"Coût minimal certifié pour p̃* ≥ {mc['theta']:g} (unités)")
    a1.set_title("Ce que coûte la même exigence selon le site", loc="left", fontsize=11, color=INK)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker=m, ls="", color=INK_2, label=f"T_R = {t:g} s")
                       for t, m in zip(trs, marq)], frameon=False, fontsize=8, labelcolor=INK_2,
              loc="lower right")
    a2.set_xlabel("T_min : temps d'intrusion minimal (s)")
    a2.set_title("Marge de temps du site", loc="left", fontsize=11, color=INK)
    for a in (a1, a2):
        _style(a)
        a.grid(axis="y", visible=False)
    finis = tab[[f"cout_certifie_TR{t:g}" for t in trs]].to_numpy(float)
    a1.set_xlim(0, 1.12 * np.nanmax(finis) if np.isfinite(finis).any() else 1)
    a2.set_xlim(0, 1.15 * tab["T_min_s"].max())
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_atlas(noms, etiq, col, mc, path):
    """Un panneau par site : dispositif certifié (θ, T_R minimal), P de détection, pire couloir."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.lines import Line2D

    from src.analyse import SEQ, base_map, draw_sensors, hillshade, to_grid
    from src.intrus import detection, load_domaine, seuil_couloir

    T_R = float(mc["t_reponse_s"][0])
    n = len(noms)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.6 * nrow), dpi=150,
                             layout="constrained", squeeze=False)
    im = None
    for ax, nom in zip(axes.ravel(), noms):
        cfg = config_site(nom)
        dom, A, conf = load_domaine(cfg, ROOT)
        sol = lire_csv(ROOT / cfg["boucle"]["out_dir"] / "solutions.csv",
                       ["theta", "T_R", "fusion", "config"])
        sol = sol[(sol.fusion == "max") & np.isclose(sol.theta, mc["theta"])
                  & np.isclose(sol.T_R, T_R)]
        T = sol.config.to_numpy(np.int64)
        P, _ = detection(A, T)
        with rasterio.open(ROOT / cfg["rasters"]["out_dir"] / "mns.tif") as s:
            mns, tr = s.read(1).astype(float), s.transform
        m = 15
        r0, r1 = max(dom.rows.min() - m, 0), min(dom.rows.max() + m + 1, mns.shape[0])
        c0, c1 = max(dom.cols.min() - m, 0), min(dom.cols.max() + m + 1, mns.shape[1])
        hs = hillshade(np.nan_to_num(mns[r0:r1, c0:c1], nan=np.nanmedian(mns)))
        extent = (tr.c + c0 * tr.a, tr.c + c1 * tr.a, tr.f + r1 * tr.e, tr.f + r0 * tr.e)
        couches = {"actifs": gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs"),
                   "perimetre": gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre")}
        base_map(ax, hs, extent, couches)
        im = ax.imshow(to_grid(P, dom.rows, dom.cols, mns.shape)[r0:r1, c0:c1], extent=extent,
                       cmap=SEQ, vmin=0, vmax=1, zorder=2, interpolation="nearest")
        cible = couches["actifs"][couches["actifs"]["cible_intrus"].astype(bool)]
        cible.boundary.plot(ax=ax, color=INK, lw=2, zorder=5)
        titre = f"{etiq[nom]}"
        if T.size:
            p, couloir = seuil_couloir(dom, P, "F")
            x = tr.c + (dom.cols[couloir] + 0.5) * tr.a
            y = tr.f + (dom.rows[couloir] + 0.5) * tr.e
            ax.plot(x, y, ls="", marker=".", ms=2.5, color="#b8456a", zorder=6)
            cand = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id")
            cand["x"], cand["y"] = cand.geometry.x, cand.geometry.y
            hfov = {c["name"]: c["hfov_deg"] for c in cfg["sensors"]["cameras"]}
            draw_sensors(ax, T, conf, cand, hfov, r_wedge=0.04 * (extent[1] - extent[0]))
            cout = pd.read_csv(ROOT / cfg["boucle"]["out_dir"] / "cout_robuste.csv")
            c = cout[np.isclose(cout.theta, mc["theta"]) & np.isclose(cout.T_R, T_R)].cout_max
            titre += f"  —  coût {c.iloc[0]:.1f}, p* = {p:.2f}"
        else:
            titre += "  —  pas de dispositif certifié"
        ax.set_title(titre, loc="left", fontsize=10, color=col[nom] if nom != REFERENCE else INK)
        # Échelle : 50 m
        xb, yb = extent[0] + 0.05 * (extent[1] - extent[0]), extent[2] + 0.05 * (extent[3] - extent[2])
        ax.plot([xb, xb + 50], [yb, yb], color=INK, lw=2, zorder=7)
        ax.annotate("50 m", (xb + 25, yb), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7, color=INK)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.set_label("P(détection) de l'intrus", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.6, ms=8, label="Radar"),
           Line2D([], [], marker="s", ls="", mfc=INK, mec=INK, ms=5, label="Support caméras"),
           Line2D([], [], color=INK, lw=2, label="Cible de l'intrus"),
           Line2D([], [], ls="", marker=".", color="#b8456a", ms=6, label="Pire couloir"),
           Line2D([], [], color=INK_2, lw=1.2, ls="--", label="Clôture")]
    fig.legend(handles=leg, loc="outside lower center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle(f"Dispositif de coût minimal certifié pour p̃* ≥ {mc['theta']:g} "
                 f"(T_R = {T_R:g} s), site par site", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Étape 6 — Comparer
# =============================================================================
COLONNES_MD = {                     # colonne -> (en-tête, format)
    "surface_ha": ("Surface (ha)", "{:.2f}"), "bati_pct": ("Bâti (%)", "{:.0f}"),
    "vegetation_pct": ("Végét. > 2 m (%)", "{:.0f}"), "denivele_m": ("Dénivelé (m)", "{:.1f}"),
    "n_configurations": ("Configurations", "{:,.0f}"),
    "dist_cible_cloture_m": ("Cible–clôture (m)", "{:.1f}"), "T_min_s": ("T_min (s)", "{:.0f}"),
    "F_B10": ("F* à B = 10", "{:.0%}"), "perte_glouton_max_pt": ("Perte glouton max (pt)", "{:.1f}"),
    "F_aveugle_max": ("F max avec p* = 0", "{:.0%}"),
    "cout_p0.9": ("Coût p*_F ≥ 0,9 (front)", "{:.1f}"),
    "cout_certifie_TR0": ("Coût certifié T_R = 0", "{:.1f}"),
    "cout_certifie_TR10": ("Coût certifié T_R = 10 s", "{:.1f}"),
}


def comparer(avec_reference: bool) -> None:
    base = load_config(site="")
    ms = base["multisite"]
    noms = ([REFERENCE] if avec_reference else []) + sites_connus()
    prets = [n for n in noms if statut(n)[NOMS.index("analyse_nsga")] == "ok"
             and all(s == "ok" for s in statut(n)[: NOMS.index("analyse_nsga") + 1])]
    manquants = sorted(set(noms) - set(prets))
    if manquants:
        print(f"Sites ignorés (chaîne incomplète jusqu'à analyse_nsga) : {manquants}")
    if not prets:
        sys.exit("Aucun site prêt.")
    out, figs = ROOT / ms["out_dir"], ROOT / ms["figures_dir"]
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    lignes, donnees = [], {}
    for n in prets:
        cfg = config_site(n)
        lignes.append(dict(id=n, **descripteurs(cfg), **resultats(cfg)))
        donnees[n] = courbes(cfg)
    tab = pd.DataFrame(lignes).set_index("id")
    tab.to_csv(out / "comparaison.csv")
    etiq = {n: (tab.loc[n, "site"] + " (réf.)") if n == REFERENCE else etiquette(n)
            for n in prets}
    col = couleurs(prets)

    md = ["| Site | " + " | ".join(h for h, _ in COLONNES_MD.values()) + " |",
          "|---|" + "---:|" * len(COLONNES_MD)]
    for n, r in tab.iterrows():
        cell = [(f.format(r[c]) if pd.notna(r[c]) else "—") for c, (_, f) in COLONNES_MD.items()]
        md.append(f"| {etiq[n]} | " + " | ".join(cell) + " |")
    (out / "comparaison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    pd.set_option("display.width", 220)
    print(tab.drop(columns=[c for c in tab if c.startswith("statut")]).T.round(3).to_string())
    figure_courbes(donnees, etiq, col, figs / "courbes.png")
    boucle_ok = [n for n in prets if statut(n)[NOMS.index("boucle")] == "ok"]
    if boucle_ok:
        figure_exigences(tab.loc[boucle_ok], etiq, col, ms["comparaison"], figs / "exigences.png")
        figure_atlas(boucle_ok, etiq, col, ms["comparaison"], figs / "atlas.png")
    print(f"\n-> {out / 'comparaison.csv'}\n-> {out / 'comparaison.md'} (à coller dans le README)"
          f"\n-> {figs}")


# =============================================================================
# Étape 7 — Ligne de commande
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.multisite", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("nouveau", help="gabarit site.gpkg à dessiner dans QGIS")
    p.add_argument("nom")
    p.add_argument("--copier", help="partir du site.gpkg d'un autre site (ou 'reference')")
    sp.add_parser("etat", help="avancement de chaque site")
    p = sp.add_parser("lancer", help="enchaîne les étapes (reprend là où c'est nécessaire)")
    p.add_argument("noms", nargs="*")
    p.add_argument("--tous", action="store_true")
    p.add_argument("--depuis", choices=NOMS, help="force le recalcul à partir de cette étape")
    p.add_argument("--jusqua", choices=NOMS, help="s'arrête après cette étape")
    sp.add_parser("_preparer", help=argparse.SUPPRESS)
    p = sp.add_parser("comparer", help="tableau et figures de comparaison")
    p.add_argument("--sans-reference", action="store_true")
    a = ap.parse_args()

    if a.cmd == "nouveau":
        nouveau(a.nom, a.copier)
    elif a.cmd == "etat":
        afficher_etat()
    elif a.cmd == "lancer":
        noms = sites_connus() if a.tous else [identifiant(n) for n in a.noms]
        if not noms:
            sys.exit("Aucun site : python -m src.multisite lancer <dossier du site>")
        for n in noms:
            try:
                config_site(n)
            except FileNotFoundError as err:
                sys.exit(str(err))
        bilan = {n: lancer(n, a.depuis, a.jusqua) for n in noms}
        print("\nBilan : " + ", ".join(f"{etiquette(n)} {'✔' if ok else '✗'}"
                                       for n, ok in bilan.items()))
    elif a.cmd == "_preparer":
        preparer()
    elif a.cmd == "comparer":
        comparer(not a.sans_reference)


if __name__ == "__main__":
    main()

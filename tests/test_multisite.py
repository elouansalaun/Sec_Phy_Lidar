"""Tests du mécanisme multisite : fusion des surcharges et déplacement des chemins (src.site)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.site import _fusion, load_config

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "paths": {"raw": "data/raw", "processed": "data/processed"},
    "site": {"gpkg": "data/site/site.gpkg", "resolution_m": 1.0},
    "rasters": {"tiles_glob": "data/raw/*.LAZ", "out_dir": "data/processed/rasters"},
    "validation": {"mrnf_ref": "data/external/mnt_mrnf.tif"},
    "optimisation": {"figures_dir": "figures", "milp": {"temps_max_s": 900, "bloc_m": 10.0}},
    "boucle": {"theta": [0.5, 0.9, 0.95]},
    "multisite": {"dossier": "data/sites", "out_dir": "data/processed/multisite",
                  "profil": {"optimisation": {"milp": {"temps_max_s": 120}},
                             "boucle": {"theta": [0.9]}}},
}


@pytest.fixture
def depot(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(BASE), encoding="utf-8")
    (tmp_path / "data/sites/beta").mkdir(parents=True)
    (tmp_path / "data/sites/beta/site.gpkg").touch()
    return tmp_path


def test_sans_site_config_inchangee(depot, monkeypatch):
    monkeypatch.delenv("LIDAR_SITE", raising=False)
    assert load_config(depot / "config.yaml") == BASE


def test_chemins_deplaces_donnees_partagees(depot):
    c = load_config(depot / "config.yaml", site="beta")
    S = depot.resolve() / "data/sites/beta/sorties"
    assert c["site"]["gpkg_source"] == str(depot.resolve() / "data/sites/beta/site.gpkg")
    assert c["site"]["gpkg"] == str(S / "site/site.gpkg")                 # copie contrôlée
    assert c["rasters"]["out_dir"] == str(S / "processed/rasters")
    assert c["paths"]["processed"] == str(S / "processed")
    assert c["optimisation"]["figures_dir"] == str(S / "figures")
    assert c["rasters"]["tiles_glob"] == "data/raw/*.LAZ"                 # partagé
    assert c["validation"]["mrnf_ref"] == "data/external/mnt_mrnf.tif"    # partagé
    assert c["multisite"]["out_dir"] == "data/processed/multisite"        # jamais déplacé
    assert c["site"]["resolution_m"] == 1.0 and c["site_actif"] == "beta"


def test_ordre_des_surcharges(depot):
    """config.yaml < multisite.profil < data/sites/<nom>/config.yaml ; les listes remplacent."""
    c = load_config(depot / "config.yaml", site="beta")
    assert c["optimisation"]["milp"] == {"temps_max_s": 120, "bloc_m": 10.0}
    assert c["boucle"]["theta"] == [0.9]
    (depot / "data/sites/beta/config.yaml").write_text(
        "optimisation: {milp: {temps_max_s: 30}}\nrasters: {tiles_glob: data/raw/b/*.LAZ}\n")
    c = load_config(depot / "config.yaml", site="beta")
    assert c["optimisation"]["milp"] == {"temps_max_s": 30, "bloc_m": 10.0}
    assert c["rasters"]["tiles_glob"] == "data/raw/b/*.LAZ"


def test_variable_environnement(depot, monkeypatch):
    monkeypatch.setenv("LIDAR_SITE", "beta")
    assert load_config(depot / "config.yaml")["site_actif"] == "beta"
    assert "site_actif" not in load_config(depot / "config.yaml", site="")   # "" force la référence


def test_site_inconnu(depot):
    with pytest.raises(FileNotFoundError, match="gamma"):
        load_config(depot / "config.yaml", site="gamma")


def test_dossier_exterieur(depot, tmp_path_factory):
    """Un site rangé n'importe où : site/site.gpkg, sorties dans <dossier>/sorties."""
    ext = tmp_path_factory.mktemp("SITE2")
    (ext / "site").mkdir()
    (ext / "site/site.gpkg").touch()
    c = load_config(depot / "config.yaml", site=str(ext))
    assert c["site"]["gpkg_source"] == str(ext.resolve() / "site/site.gpkg")
    assert c["rasters"]["out_dir"] == str(ext.resolve() / "sorties/processed/rasters")
    assert c["site_nom"] == ext.name
    assert c["rasters"]["tiles_glob"] == "data/raw/*.LAZ"      # pas de tuiles dans le dossier


def test_buffer_auto():
    """Tampon réduit près du bord des données, plafonné loin du bord, refusé trop près."""
    import shapely
    from src.site import SiteValidationError, buffer_auto
    carre = shapely.box(0, 0, 100, 100)
    assert buffer_auto(carre, [-1000, -1000, 1100, 1100], 1.0, 50, 400) == 400
    assert buffer_auto(carre, [-1000, -1000, 245, 1100], 1.0, 50, 400) == 144
    with pytest.raises(SiteValidationError):
        buffer_auto(carre, [-1000, -1000, 130, 1100], 1.0, 50, 400)


def test_fusion_ne_modifie_pas_la_base():
    base = {"a": {"b": [1, 2]}}
    out = _fusion(base, {"a": {"b": [3]}})
    assert out == {"a": {"b": [3]}} and base == {"a": {"b": [1, 2]}}


@pytest.mark.skipif(not (ROOT / "data/processed/coverage/cellules.npz").exists(),
                    reason="lancer src.coverage d'abord")
def test_k_couverture_par_nom_redonne_les_indices_de_reference():
    """« equipements » résolu par nom = indices [4, 5] choisis à la main pour le site de référence."""
    from src.nsga import zones_k_couverture
    cfg = load_config(site="")
    zone = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")["zone"]
    assert zones_k_couverture("equipements", cfg, ROOT, zone) == [int(z) for z in
                                                                   cfg["nsga"]["k_couverture"]["zones"]]

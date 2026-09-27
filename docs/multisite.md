# Lancer le projet sur un autre site

Objectif : appliquer la chaîne complète à un site dessiné dans QGIS et rangé **n'importe où** (avec ses
propres tuiles LiDAR), toutes les sorties allant dans un dossier dédié à ce site ; puis comparer les
sites entre eux.

Le site de référence (`data/site`, `data/processed`, `figures`) n'est **jamais modifié**.

## Le dossier d'un site

```
SITE2/                          (n'importe où, par ex. ~/Desktop/SITES_QGIS/SITE2)
├── site.qgz                    projet QGIS (facultatif, non lu)
├── site/site.gpkg              dessiné dans QGIS : perimetre, zones, acces   (ou site.gpkg à la racine)
├── *.LAZ                       tuiles du site (facultatif : sinon celles de data/raw)
├── config.yaml                 surcharges propres au site (facultatif)
└── sorties/                    ← créé par la chaîne, TOUT y est écrit
    ├── site/                   copie contrôlée du site, site_derive.gpkg, grid.yaml, poids.csv,
    │                           candidats.gpkg, site.png
    ├── processed/              rasters/, coverage/, optim/, intrus/, nsga/, boucle/
    │   └── logs/               un journal par étape (01_preparer.log … 16_boucle.log)
    └── figures/
```

Les fichiers d'entrée (`site.gpkg`, tuiles) ne sont jamais modifiés.

## Lancer

```bash
conda activate lidar-project
cd ~/Desktop/Projets/Sec_Phy_Lidar
python -m src.multisite lancer ~/Desktop/SITES_QGIS/SITE2
```

Étapes, dans l'ordre : `preparer, site, rasters, candidats, couverture, glouton, milp, pareto,
analyse, intrusion, intrus, eval_intrus, nsga, analyse_nsga, opportune, boucle`.

- **Reprise automatique** : une étape est sautée si sa sortie existe et est plus récente que celle de
  l'étape précédente. Si tu modifies `site.gpkg` ou `config.yaml` du site, tout est recalculé.
- `--jusqua couverture` s'arrête après une étape (utile pour vérifier un site avant la nuit de calcul) ;
  `--depuis milp` force le recalcul à partir d'une étape (l'amont n'est alors pas revérifié).
- En cas d'échec, les dernières lignes du journal s'affichent ; le journal complet est dans
  `sorties/processed/logs/`.
- Plusieurs sites d'un coup : `python -m src.multisite lancer ~/…/SITE2 ~/…/SITE3`. Un site déjà
  lancé est mémorisé : `lancer --tous` et `etat` le retrouvent.

**Durées indicatives** (profil réduit, Mac 4 cœurs) : quelques minutes jusqu'à `glouton` ; `milp` de
30 min à 1 h 30 selon la taille ; `intrusion` à `opportune` ≈ 10 min ; `boucle` ≤ 40 min.

### Ce que fait l'étape `preparer`

Elle copie `site.gpkg` dans `sorties/site/site.gpkg` et corrige ce qui empêcherait `src.site` de le lire :

- **couche dans un autre SCR** (typiquement `acces` saisie en EPSG:4326) : reprojetée en EPSG:2950 ;
- **zones sans champ `type`** : le type est déduit du nuage LiDAR — plus de 50 % de points classés
  bâtiment dans la zone → `Batiment`, sinon `Peripherie`. Le tableau est imprimé dans
  `01_preparer.log` : **vérifie-le**. Un champ `type` rempli dans QGIS a toujours priorité (utile pour
  un `Equipement`, qu'on ne peut pas distinguer d'un bâtiment dans le nuage).

### Ce qui s'adapte au site

| Réglage | Référence | Autre site |
|---|---|---|
| Tuiles | `data/raw/*.LAZ` | celles du dossier du site, sinon `data/raw` |
| Emprise des données | `lidar.data_extent` | lue dans les en-têtes des tuiles |
| Tampon de la zone d'étude | 400 m | **auto** : 400 m au plus, réduit si le site est près du bord des tuiles (minimum `bande_ext_m` + 20 m) |
| k-couverture de NSGA-II | zones `[4, 5]` | résolue par nom (`equipements`) |

Le tampon réduit ne change pas les résultats : les rayons de visibilité ne relient que des capteurs du
site à des cellules du domaine de l'intrus (site + 30 m), toutes contenues dans la grille.

### Profil de calcul réduit (à écrire dans le README)

| Étape | Référence | Autres sites |
|---|---|---|
| Optimum de couverture | certifié à 0,5 % (900 s par budget) | encadré [LB, UB], 120 s par budget ; l'écart est reporté |
| NSGA-II | 5 graines | 2 graines |
| Boucle adversariale | θ ∈ {0,5 ; 0,9 ; 0,95}, T_R ∈ {0 ; 10 ; 20 ; 30}, fusion produit | θ ∈ {0,5 ; 0,9}, T_R ∈ {0 ; 10}, fusion max |
| Sensibilité, surrogate, validation MNT | oui | non |

Les coûts de la boucle restent des optima prouvés (statut `optimal`) ; seule la liste des cas est plus
courte. Pour traiter un site aussi finement que la référence, remets les valeurs d'origine dans son
`config.yaml`, par exemple :

```yaml
optimisation:
  milp: {temps_maitre_s: 120, temps_max_s: 900}
nsga:
  graines: [1, 2, 3, 4, 5]
```

## Comparer

```bash
python -m src.multisite etat         # avancement de chaque site
python -m src.multisite comparer     # référence + tous les sites connus
```

| Sortie (dans le dépôt) | Contenu |
|---|---|
| `data/processed/multisite/comparaison.csv` | tous les indicateurs, un site par ligne |
| `data/processed/multisite/comparaison.md` | tableau prêt à coller dans le README |
| `figures/multisite/atlas.png` | un panneau par site : dispositif de coût minimal certifié (θ = 0,9, T_R = 0), P de détection, pire couloir |
| `figures/multisite/courbes.png` | F\*(B) encadré et meilleur p\*_F du front, sites superposés |
| `figures/multisite/exigences.png` | coût certifié pour T_R = 0 et 10 s, et T_min, par site |

Avec 4 ou 5 sites, on décrit, on ne régresse pas : « sur les 5 sites, F atteint X à Y % avec p\* = 0 »
tient ; « le coût suit la végétation » demande plus de points. Choisis les sites pour faire varier un
facteur à la fois (végétation, bâti, taille, distance cible–clôture) et écris ta prédiction avant de
lancer, comme pour les protocoles préenregistrés de la semaine 7.

## Démo

`python -m streamlit run app.py` : un sélecteur **Site** apparaît dans la barre latérale dès qu'un
autre site a terminé l'étape `boucle`.

## Dessiner un site dans QGIS : ce que `src.site` vérifie

| Couche / champ | Règle |
|---|---|
| `perimetre` | un seul polygone |
| `zones` | partitionnent le périmètre (écart et chevauchement < 0,5 m²) : découper le périmètre copié |
| `zones.type` | `Batiment`, `Equipement` ou `Peripherie` (déduit par `preparer` s'il manque) |
| `zones.poids` | > 0 |
| `zones.cible_intrus` | vrai pour **exactement une** zone `Batiment` ou `Equipement` |
| `acces` | points accrochés sur la clôture (0,5 m) |
| Emprise | site à au moins 50 m du bord des tuiles |

Pour partir d'un gabarit vide rangé dans le dépôt : `python -m src.multisite nouveau <nom>` crée
`data/sites/<nom>/site.gpkg` avec le bon schéma et une couche `cadre_autorise`.

## Erreurs fréquentes

| Message | Correction |
|---|---|
| `partition : … m² d'écart` | redécouper le périmètre copié, accrochage activé |
| `acces : points hors clôture` | accrocher le portail sur la clôture |
| `La zone cible n'est pas couverte par du bâti` | remplir `type` dans QGIS (`Batiment` ou `Equipement`) |
| `le site est à … m du bord des données LiDAR` | ajouter les tuiles voisines dans le dossier du site |
| `ni …/site.gpkg ni …/site/site.gpkg n'existe` | vérifier le chemin passé à `lancer` |

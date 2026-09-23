# À intégrer dans `README.md` — Semaine 2

Mode d'emploi (dans l'ordre du README) :

| # | Endroit du README | Action |
|---|---|---|
| A | `## Sommaire` | remplacer la liste (bloc A) |
| B | `## Avancement` | remplacer la ligne « 2 » (bloc B) |
| C | `## Installation et reproduction` | remplacer la ligne `conda create`, les commandes et l'arborescence (bloc C) |
| D | après la section « Semaine 1 », avant « Choix méthodologiques » | insérer la section Semaine 2 (bloc D) |
| E | fin de `## Choix méthodologiques` | ajouter les points 16 à 28 (bloc E) |
| F | `## Limites connues` | ajouter les puces (bloc F) |
| G | `## Prochaines étapes` | remplacer tout le contenu (bloc G) |
| H | fin du fichier | **supprimer** l'ancien bloc « Validation du moteur de visibilité » (collé après la mention de licence) : il est intégré au bloc D avec les valeurs définitives |

---

## Bloc A — Sommaire

```markdown
1. [Problème](#problème)
2. [Avancement](#avancement)
3. [Données](#données)
4. [Installation et reproduction](#installation-et-reproduction)
5. [Semaine 1 — Site, rasters, validation, positions candidates](#semaine-1--site-rasters-validation-positions-candidates)
6. [Semaine 2 — Visibilité, capteurs, matrice de couverture](#semaine-2--visibilité-capteurs-matrice-de-couverture)
7. [Choix méthodologiques](#choix-méthodologiques)
8. [Limites connues](#limites-connues)
9. [Prochaines étapes](#prochaines-étapes)
```

## Bloc B — Avancement (ligne 2)

```markdown
| 2 | Moteur de visibilité exact, validation (tests analytiques, GDAL, arbitre exact), modèles caméra/radar, matrice de couverture | ✅ |
```

## Bloc C — Installation et reproduction

```markdown
conda create -n lidar-project -c conda-forge python=3.11 \
    pdal python-pdal gdal laspy lazrs-python rasterio "geopandas>=1.0" "shapely>=2" \
    pyproj pyogrio pyyaml matplotlib numba scipy pandas pytest
```

```bash
python -m src.site                  # 1. validation du site, zones pondérées, grille
python -m src.rasters               # 2. MNT, MNS, hauteurs, végétation, fraction de trouée
python -m src.validation pdal       # 3a. génère le pipeline PDAL de référence
pdal pipeline data/processed/validation/mnt_pdal.json
python -m src.validation compare    # 3b. compare le MNT à PDAL et au MRNF
python -m src.candidates            # 4. positions candidates des capteurs
python -m pytest tests              # 5. tests analytiques (moteur de visibilité, capteurs)
python -m src.validation_viewshed   # 6. validation externe contre gdal_viewshed
python -m src.validation_viewshed diag C034 4   #    diagnostic d'un couple (arbitre exact)
python -m src.sensors C052 4        # 7. démo : surface efficace par capteur pour un candidat
python -m src.coverage              # 8. matrice de couverture du MCLP
```

```
├── config.yaml                 paramètres (entrées choisies, jamais de valeurs calculées)
├── src/
│   ├── site.py                 site : validation, actifs, zones d'approche, grille
│   ├── rasters.py              rasters de base par accumulation par blocs
│   ├── validation.py           validation du MNT (PDAL, MRNF)
│   ├── candidates.py           positions candidates (mâts, toits)
│   ├── viewshed.py             épaisseur optique τ de la ligne de visée (numba)
│   ├── validation_viewshed.py  comparaison à gdal_viewshed, diagnostic, arbitre exact
│   ├── sensors.py              modèles caméra (EN 62676-4) et radar (Swerling I)
│   └── coverage.py             matrice de couverture creuse (configurations × cellules)
├── tests/
│   ├── test_viewshed.py        solutions analytiques du moteur de visibilité
│   └── test_sensors.py         propriétés des modèles de capteurs, contrôle de configuration
└── data/
    ├── raw/                    tuiles LAZ (non versionnées)
    ├── external/               références externes (non versionnées)
    ├── site/                   site.gpkg (dessiné) + sorties dérivées
    └── processed/              rasters, visibilités, validation, couverture (régénérables)
```

---

## Bloc D — Section Semaine 2

```markdown
## Semaine 2 — Visibilité, capteurs, matrice de couverture

**Idée directrice : séparer la géométrie du capteur.** Pour chaque couple (position, hauteur), le moteur
calcule une seule fois, sur 360°, l'épaisseur optique τ de la ligne de visée vers chaque cellule :
τ = ∞ si un obstacle opaque est touché, τ = Σ kᵢ·ℓᵢ sinon (végétation traversée). Chaque capteur applique
ensuite son propre modèle, pour un coût de calcul négligeable :

P_det(x) = T(x) × P_capteur(distance, angles), avec T = e^(−α·τ)

- les orientations (azimut des caméras, inclinaison des radars) ne demandent aucun nouveau calcul de
  visibilité : **176 calculs de τ** pour **4 928 configurations** ;
- le coefficient α distingue les capteurs (un radar traverse mieux le feuillage qu'une caméra) ;
- on stocke τ, qui reste linéaire, plutôt que T.

### 1. Végétation semi-transparente : fraction de trouée (`src/rasters.py`)

**Modèle.** Loi de Beer-Lambert : la fraction de trouée verticale vaut P_gap = e^(−k·h_veg), d'où
k = −ln(P_gap) / h_veg [m⁻¹]. Avec une distribution sphérique des angles de feuilles (G = 0,5), le même k
s'applique aux rayons obliques.

**Estimateur.** Premiers retours uniquement (un retour n° 2 au sol n'existe que parce que l'impulsion a déjà
touché quelque chose), avec correction de Laplace : P̂_gap = (n_sol + 0,5) / (n + 1). Sans elle, un pixel
sans retour sol donnerait k = ∞.

| Surface | P_gap p10 / p50 / p90 | Lecture |
|---|---|---|
| Sol nu | 0,83 / 0,93 / 0,98 | 1 à 2 impulsions sur 22 touchent un objet bas de classe 1 (herbe, bordure). Sans effet : k n'est calculé qu'au-dessus de 0,3 m. |
| Bâtiments | 0,02 / 0,02 / 0,10 | Plancher de Laplace 0,5/23 : aucune impulsion n'atteint le sol. |
| Végétation > 2 m | 0,02 / 0,25 / 0,72 | Érables sans feuilles et conifères (p10 au plancher). |

Premiers retours par pixel : 22 (médiane).

| P_gap (arbre de 11 m) | k (m⁻¹) | Transmittance sur 10 m de branches |
|---|---|---|
| 0,72 (p90) | 0,03 | 0,74 |
| 0,25 (médiane) | 0,13 | 0,28 |
| 0,02 (p10) | 0,36 | 0,03 |

Le même estimateur traite les **objets pleins** sans règle supplémentaire : une voiture n'a aucun premier
retour au sol (k ≈ 2,7 m⁻¹ sur 1,5 m), un rayon qui la traverse sur 3 m a une transmittance d'environ
3·10⁻⁴.

**Bruit et biais.** Avec 22 impulsions, l'écart-type binomial de P̂_gap est d'environ ±0,09 par pixel. Pas de
lissage : τ additionne une dizaine de pixels le long du rayon, et ce bruit indépendant s'y moyenne. Comme
−ln est convexe, le bruit biaise k légèrement vers le haut (inégalité de Jensen) : l'erreur est pessimiste.

### 2. Moteur de visibilité (`src/viewshed.py`)

| Élément | Choix |
|---|---|
| Surface opaque | MNT + hauteur du bâti |
| Couche de végétation | du MNT à MNT + `autre_h` (si > 0,3 m), coefficient k |
| Cible | MNT + 1,0 m (centre du torse d'un intrus debout) |
| Observateur | altitude du support + hauteur du mât |
| Portée | 400 m ; courbure négligée (d²/2R = 1,3 cm à 400 m) |
| Algorithme | parcours exact de toutes les cellules traversées (Amanatides & Woo, 1987) |
| Codage | NaN = hors portée, ∞ = bloqué, T = e^(−τ) sinon |

Dans chaque cellule traversée, le rayon est bloqué si son point le plus bas (entrée ou sortie de la cellule)
passe sous la surface opaque. Sinon, la fraction exacte du passage située sous la canopée contribue
k · fraction · longueur 3D. On utilise la **longueur 3D** : un rayon qui plonge traverse plus de végétation
par mètre au sol. Les cellules du mât et de la cible sont exclues par convention.

**Performance** : environ **16 ms** par calcul sur 5·10⁵ cellules (numba, parallèle).

**Premiers constats**
- **Un coin de toit ne voit qu'environ un quart des directions.** Sur les autres, le rayon doit d'abord
  franchir le toit : pour un capteur 4 m au-dessus d'un bâtiment de 24 m, le sol n'est revu qu'à environ
  L × 24/4 derrière le bord (180 m pour L = 30 m).
- **Un mât dans les arbres voit à travers les branches, pas par-dessus.** Aucun obstacle opaque près du mât,
  mais une visibilité surtout partielle. Passer de 4 à 10 m aide un peu entre 25 et 100 m, sans lever les
  obstacles bâtis.

### 3. Tests analytiques (`tests/test_viewshed.py`)

Cinq terrains synthétiques à solution exacte, chacun isolant un terme du modèle :

| Test | Terme vérifié | Solution de référence |
|---|---|---|
| Terrain plat | géométrie du disque, codage NaN / 0 | exact |
| Mur | occultation opaque | longueur d'ombre D* = D_w·(h − t)/(h − H_w) (Thalès) |
| Dalle de végétation infinie | Beer-Lambert, longueur 3D, axe et diagonale | τ = k × longueur 3D hors cellules extrêmes |
| Dalle partielle | entrée du rayon dans la canopée | fraction (H_v − t)/(h − t) du trajet |
| Pixel du mât | convention d'exclusion | exact |

Tolérance relative de 10⁻⁵ : le parcours exact reproduit les formules à la précision du float32.

### 4. Validation externe contre `gdal_viewshed`

**Protocole.** Même surface (MNS entièrement opaque), mêmes hauteurs d'observateur et de cible (1,0 m),
courbure désactivée (`-cc 0`), observateur au centre de cellule. Seule différence : l'algorithme (parcours
exact ici, interpolation de l'horizon de Wang, Robinson & White, 2000, dans GDAL). **144 couples comparés.**
32 couples exclus, car le capteur est sous le MNS (mât dans la végétation) : un outil 2,5D ne peut pas
représenter ce cas, que ce moteur traite par semi-transparence.

**Critères fixés avant l'expérience**

| Critère | Seuil | Résultat |
|---|---|---|
| κ de Cohen médian | ≥ 0,85 | 0,870 ✅ |
| IoU médian des zones visibles | ≥ 0,85 | 0,779 ❌ |
| Désaccords à ≤ 2 px d'une frontière de visibilité | ≥ 80 % | 86,1 % ✅ |

IoU agrégé (pondéré par la surface visible) : 0,804. On suit le κ plutôt que le seul pourcentage d'accord
(99,2 %) : avec environ 80 % de cellules invisibles, l'accord brut est gonflé par le hasard.

**Analyse de l'échec sur l'IoU**

| Hypothèse | Test | Verdict |
|---|---|---|
| Effet de taille : l'IoU est instable pour les petites zones visibles | ρ(IoU, part visible) = 0,79 ; IoU médian de 0,60 (< 0,5 % visible) à 0,82 (> 5 %) | Confirmée : explique la queue, pas le plateau |
| Rayons rasants (capteur à peine au-dessus du MNS) | ρ(κ, hauteur au-dessus du MNS) = −0,22 | Rejetée |
| Décalage uniforme de l'horizon entre les moteurs | IoU en fonction d'une tolérance ε : non monotone | Rejetée |
| Obstacle au voisinage immédiat du mât | 1 % des désaccords > 1 m dus à un obstacle à < 3 m | Rejetée |
| **Un des deux moteurs est inexact** | **Arbitre exact** sur les cellules de désaccord | **Confirmée** |

**Arbitre exact.** Pour le modèle en prismes (chaque pixel est un prisme de hauteur MNS), la visibilité a une
solution exacte : parcourir toutes les cellules traversées et comparer le point le plus bas du rayon au
sommet de chaque prisme. Appliqué aux cellules de désaccord :

- **GDAL** : 100 % des cellules qu'il est seul à voir sont en réalité masquées. Son interpolation de
  l'horizon fait « fuir » la visibilité par-dessus les obstacles, ce qui surestime la zone visible d'environ
  15 % (cas médian).
- **Version initiale de ce moteur** (rayon échantillonné tous les 0,5 px) : 17 à 26 % des cellules qu'il
  était seul à voir étaient masquées. L'échantillonnage manquait les cellules effleurées par un coin, d'où
  environ 1 % de faux visible. **Corrigé** en adoptant le parcours exact, qui donne aussi la longueur exacte
  traversée dans la végétation et réduit le temps de calcul de 28 à 16 ms. Après correction : 100 % d'accord
  avec l'arbitre.

**Conclusion.** L'écart résiduel avec GDAL vient de l'approximation de GDAL, pas d'une erreur de ce moteur.
Le critère d'IoU ≥ 0,85 était mal calibré : la variabilité entre algorithmes de visibilité est connue
(Fisher, 1993), et elle est plus forte que je ne l'avais anticipé en tissu urbain dense. Le modèle en prismes
est le plus conservateur des deux, puisqu'il épaissit les obstacles jusqu'à un pixel. Il sous-estime la
couverture plutôt que de la surestimer, ce qui est le bon côté de l'erreur pour un système de sûreté.

### 5. Modèles de capteurs (`src/sensors.py`)

| | Caméra | Radar |
|---|---|---|
| Propagation | aller simple : T = e^(−α_c·τ) | aller-retour, sur la puissance : T² = e^(−2α_r·τ) |
| Critère | densité de pixels sur la cible | rapport signal sur bruit |
| Loi de détection | TTPF sur x = ppm / 25 px/m | Swerling I : P_d = P_fa^(1/(1+SNR)) |
| Champ | sténopé exact, orientable (8 azimuts) | 360°, faisceau de ±15°, inclinaison 0/5/10/15° |
| Végétation | α_c = 1 (justifié physiquement) | α_r = 0,3 (hypothèse) |

**Caméra.**
- **Géométrie en sténopé exacte.** Chaque cellule est projetée dans le repère de la caméra : l'empreinte
  d'une caméra inclinée est un trapèze, pas un secteur. La densité de pixels se calcule à la profondeur
  z = v·f le long de l'axe optique : ppm = N_h / (2z·tan(HFOV/2)).
- **Seuil normatif et forme de la courbe.** Seuil de détection d'une personne : **25 px/m** (EN 62676-4,
  critères DORI). La norme donne un seuil, pas une probabilité. La forme de la courbe est la TTPF issue des
  critères de Johnson : P = x^E/(1 + x^E), E = 2,7 + 0,7x. Soit P = 0,5 au seuil, 0,94 au double et 0,11 à la
  moitié. Calculée sous forme logistique en ln x pour éviter tout débordement numérique.
- **Inclinaison automatique** selon la pratique d'installation : le bord haut de l'image vise le sol à
  r_design, distance où la densité vaut 25 px/m. Seul l'azimut reste une variable de décision.
- **α_c = 1.** Aux longueurs d'onde optiques, feuilles et branches sont beaucoup plus grandes que la longueur
  d'onde : l'extinction est géométrique et ne dépend pas de la longueur d'onde. Le k mesuré par le LiDAR à
  1 064 nm s'applique donc à une caméra visible ou proche infrarouge.

| Objectif (1080p) | HFOV | r_design | Coût relatif |
|---|---|---|---|
| Grand angle | 90° | 38 m | 1,0 |
| Standard | 60° | 67 m | 1,0 |
| Téléobjectif | 30° | 143 m | 1,3 |

**Radar.**
- **Équation radar** calibrée sur une portée de référence : SNR(R) = SNR₉₀·(R₉₀/R)⁴·e^(−2α_r·τ), avec
  R₉₀ = 150 m, P_fa = 10⁻⁶, d'où SNR₉₀ = ln(P_fa)/ln(0,9) − 1 = 130 (21,1 dB). Coût relatif : 5,0.
- **Masque d'élévation** : faisceau de ±15° autour d'une inclinaison choisie parmi 0, 5, 10 et 15°.
- **α_r = 0,3 est une hypothèse.** Aux longueurs d'onde centimétriques, rameaux et feuilles ont une taille
  comparable à la longueur d'onde : l'optique géométrique ne s'applique plus, et l'atténuation dépend de la
  fréquence et de la saison (ITU-R P.833). Ce paramètre sera soumis à l'analyse de sensibilité.

**Contrôle de la troncature.** Le moteur ne calcule rien au-delà de 400 m. Un test de configuration vérifie
que la probabilité idéale au bord du disque reste sous 5 % pour chaque capteur : 4,6 % pour le téléobjectif,
2,1 % pour le radar.

**Tests (`tests/test_sensors.py`).** Seuil DORI sur l'axe optique (P = 0,5 exactement), limites du champ
horizontal et cible derrière la caméra, végétation en aller simple pour la caméra et en aller-retour pour le
radar, portée de référence du radar (P = 0,9), masque d'élévation et inclinaison, opacité conservée quel que
soit α, stabilité numérique de la TTPF, troncature du disque.

**Démonstration** (surface efficace A_eff = Σ P·a sur le sol ; le rendement la compare au même capteur, au
même endroit, sans aucun obstacle) :

| Candidat | Capteur | Meilleur azimut | A_eff (m²) | Rendement |
|---|---|---|---|---|
| C052, coin de toit, 4 m | grand angle | 135° | 608 | 0,45 |
| | standard | 180° | 1 055 | 0,59 |
| | téléobjectif | 90° | 1 425 | 0,50 |
| | radar | 360° | 18 212 | 0,14 |
| C015, mât dans les arbres, 10 m | grand angle | 270° | 863 | 0,71 |
| | standard | 225° | 1 090 | 0,49 |
| | téléobjectif | 225° | 1 657 | 0,39 |

- **C052** : les trois caméras ne couvrent rien à 0° ni à 315°. Seul le quadrant extérieur au bâtiment est
  ouvert.
- **C015** : le grand angle a un meilleur rendement dans les arbres (0,71) que sur le toit (0,45). Sur sa
  courte portée, il voit à travers la végétation, alors que le toit perd ses directions derrière son propre
  bâtiment. **Le meilleur emplacement dépend du capteur.**
- **En surface brute, le téléobjectif l'emporte**, car la surface nominale varie comme HFOV × r_design²
  (rapport 1 : 2,1 : 4,7 du grand angle au téléobjectif). Ce classement s'inverse une fois la couverture
  pondérée par le site (section suivante).

### 6. Matrice de couverture (`src/coverage.py`)

**Formulation.**
- Une **configuration** i est un choix de (position, hauteur, capteur, azimut ou inclinaison).
- Une **cellule** j est un pixel de sol d'une zone de couverture, de poids w_j (poids de la zone × surface du
  pixel).
- La matrice contient **A_ij = P_det**. Elle est stockée **creuse** (format CSR) : les P < 0,01 sont mises à
  zéro, et la perte est mesurée.
- Les coûts du support (mât ou toit) et du capteur restent **séparés** : un mât payé une fois peut porter
  plusieurs capteurs.

**Contrôle de rastérisation** (règle du centre de pixel, sol uniquement) :

| Zone | Poids | Surface du polygone (m²) | Pixels | Pixels retenus (sol) |
|---|---|---|---|---|
| Bat.3 | 10 | 3 255 | 3 259 | 3 175 |
| Bat.1 | 5 | 1 603 | 1 599 | 1 533 |
| Bat.2 | 4 | 223 | 223 | 217 |
| Bat.4 | 4 | 565 | 564 | 534 |
| Gaz Bat.3 | 4 | 119 | 119 | 119 |
| Générateur | 4 | 29 | 32 | 32 |
| Périphérie | 1 | 4 362 | 4 363 | 4 363 |

Le masque sol retire 2 à 5 % des pixels autour des bâtiments (bords de toits vus par le LiDAR au-delà des
polygones dessinés). Le Générateur montre l'effet de la résolution sur une petite zone : +10 %.

**Résultats**

| Indicateur | Valeur |
|---|---|
| Cellules / poids total W | 9 973 / 47 386 |
| Configurations utiles / écartées (aucune couverture) | 3 259 / 1 669 |
| Non-nuls (densité) | 3,04 millions (9,4 %) |
| Perte due au seuil p_min = 0,01 | 0,011 % de la couverture pondérée |
| Temps de construction | 7 s |

Part du poids couverte par **une seule** configuration (Σ w·P / W) :

| Capteur | Configurations | Médiane | p90 | Max |
|---|---|---|---|---|
| Grand angle | 1 165 | 0,009 | 0,053 | 0,105 |
| Standard | 885 | 0,009 | 0,051 | 0,129 |
| Téléobjectif | 597 | 0,004 | 0,033 | 0,122 |
| Radar | 612 | 0,147 | 0,451 | 0,486 |

| Médiane | Grand angle | Standard | Téléobjectif | Radar |
|---|---|---|---|---|
| Mât de clôture | 0,005 | 0,004 | 0,004 | 0,347 |
| Mât intérieur | 0,035 | 0,017 | 0,004 | 0,313 |
| Toit | 0,006 | 0,010 | 0,003 | 0,019 |

**Lecture**
- **Une fois pondéré par le site, le téléobjectif perd son avantage** (médiane 0,004 contre 0,009) : sa
  portée utile tombe surtout hors du site.
- **Même le meilleur radar ne couvre que 49 % du poids** (C020, mât intérieur de 4 m). Un seul capteur ne
  suffit pas : les ombres des quatre bâtiments imposent des combinaisons.
- **Le catalogue de positions suffit** : seulement 0,6 % des cellules (0,6 % du poids) ne sont vues à
  P ≥ 0,5 par aucune configuration, même seule (`p_max.tif`). Elles sont surtout autour de Bat.1 et Bat.2.
  Le problème est un arbitrage de budget, pas un manque de positions.

**Une correction de modèle en cours de route.** La première version plaçait le faisceau radar sur
l'horizontale (±15°). Les radars de toit, environ 27 m au-dessus du sol, ne voyaient alors le sol qu'au-delà
d'environ 100 m, donc aucune cellule du site (médiane 0,003, 39 couples sans aucune couverture). Un
installateur inclinerait le radar vers le bas : l'inclinaison est devenue une variable de décision.
Résultats :
- **Toits** : médiane de 0,003 à 0,019. Les radars de toit restent limités, car même incliné le faisceau
  laisse un cône aveugle d'environ 47 m.
- **Mâts** : les plus hauts remontent dans le classement (C014 à 10 m : 0,483).
- **L'inclinaison optimale place le bord haut du faisceau à l'horizon** (15°), avec moins de 0,2 point
  d'écart entre 5 et 15° : sur un site plat, toutes les cibles ont un angle entre 0 et 30° sous l'horizontale.
  L'inclinaison reste une variable, car elle redevient déterminante sur un terrain en pente.

### 7. Ce qu'il faut retenir de la semaine 2

1. **Valider contre un outil établi ne suffit pas** : GDAL et le moteur étaient en désaccord. C'est un
   arbitre exact qui a montré que GDAL surestimait la visibilité (environ 15 %) et qu'une première version
   du moteur la surestimait aussi (environ 1 %, corrigé).
2. **Un coin de toit ne voit qu'environ un quart des directions** : il convient à une caméra orientée vers
   l'extérieur, pas à un radar à 360°.
3. **Le meilleur emplacement dépend du capteur** : un mât dans les arbres convient à un grand angle, pas à
   un téléobjectif.
4. **La surface brute trompe** : le téléobjectif domine en surface, mais pas en couverture pondérée du site.
5. **Un seul capteur ne suffit pas** : même le meilleur radar couvre moins de la moitié du poids.
```

---

## Bloc E — Choix méthodologiques (à ajouter après le point 15)

```markdown
**Visibilité**
16. **Séparer la géométrie du capteur** : τ est calculé une fois par couple (position, hauteur), les modèles
    de capteurs s'appliquent ensuite. On stocke τ (linéaire), pas T.
17. **Fraction de trouée par colonne plutôt que voxels 3D** : mesurable directement sur les premiers retours,
    robuste avec 22 impulsions par pixel. Les voxels restent une amélioration possible, notamment pour
    estimer la base du houppier.
18. **Surface en prismes et parcours exact** : c'est le modèle le plus conservateur ; il ne rate aucun
    obstacle.
19. **Cible à 1,0 m au-dessus du MNT, cellules du mât et de la cible exclues** : conventions explicites,
    vérifiées par des tests.

**Validation**
20. **Critères fixés avant l'expérience** et échec documenté plutôt que seuil ajusté a posteriori.
21. **Des hypothèses testées une par une**, y compris celles qui ont été réfutées.
22. **Un arbitre exact plutôt qu'une référence logicielle** : GDAL est une autre approximation, pas la vérité.

**Capteurs**
23. **Seuil normatif et forme de courbe séparés** : EN 62676-4 fixe le point à 50 %, la TTPF donne la forme.
    C'est un choix de modélisation, assumé comme tel.
24. **Réglages d'installateur** : l'inclinaison des caméras est déterminée par r_design, tandis que l'azimut
    des caméras et l'inclinaison des radars sont des variables de décision discrètes.
25. **Propagation aller simple pour la caméra (passive), aller-retour pour le radar (actif).**
26. **Troncature contrôlée par un test de configuration** : changer une portée sans agrandir le disque de
    calcul fait échouer les tests.

**Couverture**
27. **Cellules = sol des zones pondérées uniquement** ; matrice creuse, perte due au seuil mesurée (0,011 %).
28. **Coûts du support et du capteur séparés** : un mât est un coût fixe partagé entre ses capteurs.
```

## Bloc F — Limites connues (puces à ajouter)

```markdown
- **Résolution de 1 m** : la géométrie sous le pixel (bords de toits, cimes) n'est pas représentée. Le modèle
  en prismes en est une borne conservative.
- **Végétation** : couche homogène du sol au sommet (pas de base de houppier), angles de feuilles supposés
  sphériques (G = 0,5), angle de scan du LiDAR non corrigé, bruit binomial d'environ ±0,09 sur P_gap par
  pixel. Acquisition sans feuilles : transmittances optimistes pour l'été.
- **La semi-transparence n'est validée que par des tests analytiques** : aucune mesure de terrain (par
  exemple, des photos prises depuis un point de vue connu) ne la confirme encore.
- **P_det = T × P_capteur suppose l'indépendance** entre la fraction visible de la cible et sa
  détectabilité.
- **Radar** : α_r = 0,3 est une hypothèse ; fouillis (clutter), trajets multiples et Doppler sont ignorés.
- **Caméra** : conditions de jour et contraste idéal ; les critères DORI sont conçus pour des opérateurs
  humains.
- **32 couples** (mât sous la surface du MNS) ne peuvent pas être validés contre un outil 2,5D.
```

## Bloc G — Prochaines étapes (remplace tout le contenu)

```markdown
**Semaine 3 — Baselines glouton et MILP**
- **Couverture probabiliste** : avec plusieurs capteurs, une cellule est détectée avec la probabilité
  1 − Π(1 − P_ij). Le glouton s'en accommode (fonction sous-modulaire, garantie (1 − 1/e)) ; le MILP
  demande une linéarisation (logarithme de la non-détection, ou seuils de couverture).
- **Coût fixe des supports** : variables binaires d'ouverture de mât, avec plusieurs capteurs par mât.
- **Contraintes d'exclusion** : une seule hauteur par position, et pas deux capteurs au même emplacement
  physique.
- **Métriques** : couverture pondérée, k-couverture des équipements, écart du glouton à l'optimum.
```

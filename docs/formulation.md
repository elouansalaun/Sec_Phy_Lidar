# Formulation du problème de placement de capteurs

Ce document est la spécification mathématique de la semaine 3. Il sert de référence pour `src/greedy.py`, `src/milp.py` et leurs tests. Chaque paramètre cité renvoie à une clé de `config.yaml`.

---

## 1. Données

| Symbole | Signification | Source |
|---|---|---|
| $I$ | cellules pondérées du site, indicées par $i$ ($\lvert I\rvert = 9\,973$) | `coverage/cellules.npz` |
| $w_i \ge 0$ | poids de la cellule $i$ (zone × surface) | `cellules.npz["w"]` |
| $J$ | configurations capteur, indicées par $j$ ($\lvert J\rvert = 3\,259$) : un capteur, un support, un azimut, une inclinaison | `coverage/configs.csv` |
| $P_{ij} \in [0,1]$ | probabilité que la configuration $j$ détecte un intrus en $i$ (visibilité × modèle capteur) | `coverage/A.npz`, stockée **transposée** : `A[j, i]` |
| $C$ | emplacements physiques (colonne `cand`) | `configs.csv` |
| $S \subseteq C \times H$ | supports $s = (c, h)$ : un emplacement et une hauteur | `configs.csv` (`cand`, `h`) |
| $s(j)$ | support qui porte la configuration $j$ | `configs.csv` |
| $\kappa(j) \in \{\text{radar}, \text{camera}\}$ | type de capteur de $j$, déduit de la section `sensors` | `config.yaml` |
| $a_j$ | coût du capteur | `cout_capteur` |
| $f_s$ | coût fixe d'ouverture du support (mât ou fixation en toit) | `cout_support` |
| $B$ | budget | `optimisation.budget` |
| $K_\kappa$ | capacité d'un support pour le type $\kappa$ (1 radar, 4 caméras) | `optimisation.capacite_support` |

Les valeurs $P_{ij} < p_{\min}$ ont été mises à zéro à la semaine 2 (`coverage.p_min`, perte mesurée : 0,011 % du poids). 

## 2. Variables de décision

- $x_j \in \{0,1\}$ : la configuration $j$ est installée.
- $y_s \in \{0,1\}$ : le support $s$ est construit.
- $\eta_g \ge 0$ : majorant de la couverture pondérée du bloc $g$ dans le problème maître (§4.2).

## 3. Modèle exact

### 3.1 Objectif

On suppose que, pour un intrus situé en $i$, les détections des différents capteurs sont des événements **indépendants**. L'intrus n'échappe à la détection que s'il échappe à chaque capteur installé :

$$
\Pr[\text{non-détection en } i] = \prod_{j \in J} (1 - P_{ij})^{x_j}.
$$

On maximise l'espérance du poids détecté :

$$
\max_{x, y} \; F(x) = \sum_{i \in I} w_i \left( 1 - \prod_{j \in J} (1 - P_{ij})^{x_j} \right).
$$

### 3.2 Contraintes

**(C1) Budget.** Le coût d'un support est payé une seule fois, quel que soit le nombre de capteurs qu'il porte :

$$
\sum_{j \in J} a_j x_j + \sum_{s \in S} f_s y_s \le B.
$$

**(C2) Un capteur exige son support.**

$$
x_j \le y_{s(j)} \qquad \forall j \in J.
$$

**(C3) Une seule hauteur par emplacement.** On ne pose pas un mât de 4 m et un mât de 10 m au même endroit :

$$
\sum_{h : (c,h) \in S} y_{(c,h)} \le 1 \qquad \forall c \in C.
$$

**(C4) Capacité par type.** Un radar au plus, qui a besoin du sommet dégagé, et quatre caméras au plus par support :

$$
\sum_{j : s(j) = s,\ \kappa(j) = \kappa} x_j \le K_\kappa \, y_s \qquad \forall s \in S,\ \forall \kappa.
$$

Grâce au facteur $y_s$, (C4) implique déjà (C2) pour des solutions entières. On garde quand même (C2) : en relaxation continue, $x_j \le y_s$ est plus serrée que $\sum x_j \le K_\kappa y_s$, ce qui améliore la borne et accélère le branch-and-bound.

### 3.3 Hypothèses et limites

1. **Indépendance conditionnelle.** Les détections de deux capteurs qui visent la même cellule sont supposées indépendantes. Les corrélations réelles (brouillard, pluie, intrus qui rampe) rendent cette hypothèse optimiste : la vraie détection combinée se situe entre $\max_j P_{ij}$ et $1 - \prod_j (1 - P_{ij})$. Les deux cas extrêmes pourront servir d'analyse de sensibilité.
2. **Visibilité statique.** $P_{ij}$ provient d'un MNS figé au moment du vol LiDAR : ni végétation saisonnière ni véhicules stationnés.
3. **Espérance et non pire cas.** On maximise la détection moyenne pondérée. Un adversaire qui connaît le dispositif choisit son chemin : c'est la question de la couverture de chemins, hors du périmètre de cette semaine.

## 4. Résolution exacte

### 4.1 Modèle compact à coupes tangentes : correct mais intraitable

La première idée consiste à linéariser directement l'objectif. On pose $\lambda_{ij} = -\ln(1 - \min(P_{ij}, 0{,}999))$ et $\sigma_i = \sum_j \lambda_{ij} x_j$. On a alors l'identité exacte $\prod_j (1 - P_{ij})^{x_j} = e^{-\sigma_i}$. Comme $g(\sigma) = 1 - e^{-\sigma}$ est concave, on la majore par des tangentes :

$$
z_i \le 1 - e^{-\sigma_k} + e^{-\sigma_k}(\sigma_i - \sigma_k) \quad \forall i, k.
$$

Avec 10 points de tangence, la surestimation reste inférieure à 0,010 par cellule. Le modèle donne donc une borne supérieure valide.

**Mesures sur le site** (B = 12, HiGHS, 4 cœurs) :

| Grandeur | Valeur |
|---|---|
| Taille | 113 356 lignes × 23 376 colonnes, **3,26 M non-nuls** (ceux de $A$) |
| Branch-and-bound | aucune solution réalisable trouvée en 150 s |
| Relaxation continue (simplexe) | non résolue en 150 s |
| Relaxation continue (point intérieur) | 84 s, borne **0,927**, avec des fractions de capteur sur 31 configurations |
| Meilleure solution connue (glouton) | 0,680 |

Chaque nœud de l'arbre coûte plus d'une minute, et la borne à la racine laisse un écart de 36 %. Le problème vient de la relaxation elle-même : des fractions $x_j = 0{,}3$ réparties sur de nombreuses configurations « voient » presque tout le site. **Modèle rejeté.**

### 4.2 Décomposition par coupes sous-modulaires multi-blocs

On découpe les cellules en blocs spatiaux $g$, des carrés de `optimisation.milp.bloc_m` de côté, et on écrit $F = \sum_g F_g$ avec

$$
F_g(T) = \sum_{i \in g} w_i \Big(1 - \prod_{j \in T} (1 - P_{ij})\Big).
$$

Chaque $F_g$ est une somme de fonctions monotones sous-modulaires (§5.1) : elle est donc elle-même monotone sous-modulaire.

**Inégalité de Nemhauser et Wolsey.** Pour tous $S, T \subseteq J$ :

$$
F_g(T) \le F_g(S \cup T) \le F_g(S) + \sum_{j \in T \setminus S} \rho_{gj}(S),
\qquad \rho_{gj}(S) = \sum_{i \in g} w_i \, q_i(S) \, P_{ij}.
$$

La première inégalité découle de la monotonie. La seconde s'obtient en ajoutant les éléments de $T \setminus S$ un par un : par sous-modularité, chaque gain est au plus le gain calculé en $S$.

**Problème maître**, sur $(x, y, \eta)$ seulement, soit $\lvert J\rvert + \lvert S\rvert + G$ variables :

$$
\max \sum_g \eta_g \quad \text{s.c.} \quad \text{(C1)–(C4)}, \qquad
\eta_g \le F_g(S_k) + \sum_{j \notin S_k} \rho_{gj}(S_k)\, x_j \quad \forall g,\ \forall S_k \in \mathcal{S},
\qquad 0 \le \eta_g \le \textstyle\sum_{i \in g} w_i.
$$

**Algorithme.**

1. Initialiser le pool $\mathcal{S}$ avec $\emptyset$, les ensembles successifs du glouton et l'optimum du budget précédent. Les coupes ne dépendent pas de $B$ : elles restent valides d'un budget à l'autre.
2. Résoudre le maître, ce qui donne $T_k$ et la borne duale $\mathrm{UB}_k$.
3. Évaluer $F(T_k)$ exactement, puis poser $\mathrm{LB} = \max(\mathrm{LB}, F(T_k))$.
4. Ajouter les $G$ coupes en $S = T_k$, puis revenir à l'étape 2 tant que $(\mathrm{UB} - \mathrm{LB})/\mathrm{UB} >$ `ecart_cible`.

### 4.3 Validité et convergence

- **Borne supérieure.** Pour tout $T$ réalisable, $\eta_g = F_g(T)$ satisfait toutes les coupes. Le maître a donc une valeur $\ge F(T)$, en particulier $\ge \mathrm{OPT}$. On retient la **borne duale** de HiGHS, qui reste valide même quand le maître s'arrête sur sa limite de temps.
- **Borne inférieure.** Chaque $T_k$ est réalisable, puisqu'il satisfait (C1)–(C4). $F(T_k)$ est calculé exactement, sans logarithme, sans plafonnement et sans tangente.
- **Convergence finie.** Si le maître renvoie un $T_k$ déjà présent dans le pool, sa coupe impose $\eta_g \le F_g(T_k)$, donc $\mathrm{UB} = F(T_k) = \mathrm{LB}$ : $T_k$ est optimal. Le nombre d'ensembles étant fini, l'algorithme termine.

### 4.4 Pourquoi plusieurs blocs

Avec un seul bloc ($G = 1$), on retrouve l'algorithme original de Nemhauser et Wolsey. Une coupe agrège alors tout le site : le maître peut « emprunter » la couverture promise à une zone pour la créditer à une autre. Plus les blocs sont petits, plus les coupes sont fortes, mais plus le maître grossit.

Mesures à B = 12, avec 150 s au total :

| Découpage | Itérations | Écart certifié |
|---|---|---|
| 1 bloc (site entier) | 11 | 6,4 % |
| 163 blocs de 10 m | 4 | 0,95 % |

Chaque coupe multi-blocs est **creuse**, puisque seules les configurations qui voient le bloc $g$ y apparaissent. Elle ajoute environ 60 000 non-nuls, contre 3,26 M pour le modèle compact.

## 5. Heuristique gloutonne

### 5.1 Sous-modularité

À support fixé, $F$ est une fonction d'ensemble $F(T)$ sur $T \subseteq J$. Le gain marginal de $j \notin T$ vaut

$$
\Delta_j(T) = F(T \cup \{j\}) - F(T) = \sum_i w_i \, P_{ij} \prod_{k \in T} (1 - P_{ik}) = \sum_i w_i \, q_i(T) \, P_{ij},
$$

où $q_i(T)$ est la probabilité de non-détection courante de la cellule $i$. Comme $q_i(T)$ décroît lorsque $T$ grandit, $\Delta_j(T)$ est décroissant en $T$ : $F$ est **monotone et sous-modulaire** (rendements décroissants).

**Conséquence pratique.** Tous les gains se calculent en un seul produit matrice creuse × vecteur :

$$
\Delta = A \,(w \odot q) \in \mathbb{R}^{\lvert J\rvert},
\qquad \text{puis, après avoir choisi } j^\star : \ q \leftarrow q \odot (1 - A_{j^\star,\cdot}).
$$

### 5.2 Algorithme

```
q ← 1, T ← ∅, supports ouverts ← ∅, dépense ← 0
répéter
    Δ ← A @ (w * q)
    coût_j ← a_j + f_{s(j)} · [s(j) non ouvert]
    admissible_j ← dépense + coût_j ≤ B
                   ∧ aucune autre hauteur ouverte en c(j)           (C3)
                   ∧ capacité κ(j) de s(j) non atteinte             (C4)
                   ∧ j ∉ T
    si aucun j admissible avec Δ_j > 0 : arrêter
    j* ← argmax_{j admissible} Δ_j / coût_j
    T ← T ∪ {j*}, ouvrir s(j*), dépense += coût_j*, q ← q ⊙ (1 − A[j*])
renvoyer le meilleur entre T et le meilleur singleton {j} tel que a_j + f_{s(j)} ≤ B
```

### 5.3 Garanties et absence de garantie

| Cadre | Garantie | Référence |
|---|---|---|
| Cardinalité ($\lvert T\rvert \le k$), glouton sur $\Delta$ | $1 - 1/e \approx 0{,}632$ | Nemhauser, Wolsey et Fisher (1978) |
| Budget avec coûts **additifs**, glouton au ratio seul | aucune (arbitrairement mauvais) | — |
| Budget additif, meilleur entre ratio et singleton | $\ge \tfrac{1}{2}(1 - 1/e) \approx 0{,}316$ | Khuller, Moss et Naor (1999) |
| Budget additif, énumération partielle des triplets | $1 - 1/e$ | Sviridenko (2004) |
| **Notre cas** : coûts de support partagés + (C3) + (C4) | **aucune** | — |

Deux mécanismes font tomber la garantie dans notre cas :

- Le coût d'une configuration dépend de ce qui est déjà choisi, parce que le support peut être déjà payé. La fonction de coût n'est donc plus modulaire.
- Le premier choix de hauteur en $c$ verrouille l'emplacement (C3). Le glouton ne revient jamais sur une décision.

Le glouton sert donc de **référence rapide et de borne inférieure**. Seule la décomposition du §4 certifie la qualité d'une solution.

## 6. Protocole expérimental

1. Balayer le budget de `budget.min` à `budget.max` par pas de `budget.pas`.
2. Pour chaque $B$, enregistrer :
   - $F(x^{\text{glouton}})$ ;
   - $F(\hat{x})$, la meilleure solution de la décomposition ;
   - $\mathrm{UB}$, la borne supérieure certifiée ;
   - le temps de calcul ;
   - la composition de la solution (nombre de radars, de caméras et de supports, hauteurs retenues).
3. Tracer le front couverture/budget avec les trois courbes et la bande $[\mathrm{LB}, \overline{\mathrm{UB}}]$.

**Prédictions à vérifier :**

- L'écart entre glouton et MILP est maximal à petit budget, à cause du verrouillage de hauteur.
- À budget moyen, le MILP associe un radar et des caméras qui couvrent son cône aveugle.
- Le front présente un coude, au-delà duquel les capteurs ajoutés ne font que de la redondance.

## Références

- G. L. Nemhauser, L. A. Wolsey, M. L. Fisher (1978). *An analysis of approximations for maximizing submodular set functions — I.* Mathematical Programming 14, 265–294.
- S. Khuller, A. Moss, J. Naor (1999). *The budgeted maximum coverage problem.* Information Processing Letters 70(1), 39–45.
- M. Sviridenko (2004). *A note on maximizing a submodular set function subject to a knapsack constraint.* Operations Research Letters 32(1), 41–43.
- G. L. Nemhauser, L. A. Wolsey (1981). *Maximizing submodular set functions: formulations and analysis of algorithms.* Annals of Discrete Mathematics 11, 279–301.
- R. Church, C. ReVelle (1974). *The maximal covering location problem.* Papers of the Regional Science Association 32, 101–118.

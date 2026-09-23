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

Les valeurs $P_{ij} < p_{\min}$ ont été mises à zéro à la semaine 2 (`coverage.p_min`, perte mesurée : 0,011 % du poids). Avant tout logarithme, on plafonne $P_{ij} \le p_{\text{clip}} = 0{,}999$ (`optimisation.p_clip`).

## 2. Variables de décision

- $x_j \in \{0,1\}$ : la configuration $j$ est installée.
- $y_s \in \{0,1\}$ : le support $s$ est construit.
- $\sigma_i \ge 0$ : « exposition » de la cellule $i$ (définie en §4).
- $z_i \in [0,1]$ : probabilité de détection de la cellule $i$, dans sa version linéarisée.

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

## 4. Linéarisation exacte de l'objectif

### 4.1 Passage au logarithme

On pose $\lambda_{ij} = -\ln(1 - P_{ij}) \ge 0$. Ce paramètre est fini grâce au plafonnement $p_{\text{clip}}$, et $\lambda_{ij} = 0$ lorsque $P_{ij} = 0$, ce qui conserve le caractère creux de la matrice. On définit ensuite

$$
\sigma_i = \sum_{j \in J} \lambda_{ij} \, x_j.
$$

**Identité.** Pour tout $x \in \{0,1\}^{J}$,

$$
\prod_j (1 - P_{ij})^{x_j} = \exp\Big( \sum_j x_j \ln(1 - P_{ij}) \Big) = \exp(-\sigma_i),
$$

et donc $F(x) = \sum_i w_i \big(1 - e^{-\sigma_i}\big)$. L'égalité est **exacte**, sans approximation : la non-linéarité est entièrement reportée dans la fonction scalaire $g(\sigma) = 1 - e^{-\sigma}$, tandis que $\sigma_i$ est linéaire en $x$.

### 4.2 Coupes tangentes

La fonction $g$ est concave : $g''(\sigma) = -e^{-\sigma} < 0$. Toute tangente à $g$ la majore donc. Pour une liste de points $0 = \sigma_0 < \sigma_1 < \dots < \sigma_{K-1}$ (`optimisation.milp.tangentes_s`) :

$$
z_i \le 1 - e^{-\sigma_k} + e^{-\sigma_k} (\sigma_i - \sigma_k) \qquad \forall i \in I,\ \forall k,
\qquad z_i \le 1.
$$

La tangente en $\sigma_0 = 0$ se réduit à $z_i \le \sigma_i$.

Le programme linéaire en nombres entiers résolu est :

$$
\text{(MILP)} \quad \max \sum_i w_i z_i \quad \text{s.c. (C1)–(C4), définition de } \sigma_i, \text{ coupes tangentes}, \ x, y \text{ binaires}.
$$

### 4.3 Pourquoi c'est une borne supérieure

Soit $x$ une solution réalisable du modèle exact. En prenant $z_i = g(\sigma_i(x))$, toutes les coupes sont satisfaites, puisque chaque tangente majore $g$. Ce choix de $(x, z)$ est donc réalisable pour le MILP, et sa valeur vaut $F(x)$. Il s'ensuit que $\mathrm{OPT}_{\text{MILP}} \ge \max_x F(x) = \mathrm{OPT}$.

Inversement, pour la solution $\hat{x}$ renvoyée par le MILP, on calcule **exactement** $F(\hat{x})$. C'est une borne inférieure, puisque $\hat{x}$ est réalisable. On obtient l'encadrement

$$
F(\hat{x}) \;\le\; \mathrm{OPT} \;\le\; \overline{\mathrm{UB}},
$$

où $\overline{\mathrm{UB}}$ est la **borne duale** du solveur, et non la valeur de sa meilleure solution : on résout avec une tolérance `mip_rel_gap`, si bien que l'optimum du MILP lui-même n'est connu qu'à cette tolérance près. L'écart certifié est $(\overline{\mathrm{UB}} - \mathrm{LB}) / \overline{\mathrm{UB}}$, avec $\mathrm{LB} = \max\big(F(\hat{x}), F(x^{\text{glouton}})\big)$.

### 4.4 Choix des points de tangence

Entre deux points de tangence $\sigma_k < \sigma_{k+1}$, l'erreur maximale de l'enveloppe des tangentes est atteinte à leur intersection. Voici l'erreur absolue maximale sur $g$, calculée numériquement sur $\sigma \in [0, 20]$ :

| Points | $K$ | Erreur max | Atteinte en |
|---|---|---|---|
| `[0, 0.1, 0.25, 0.5, 1, 2, 3, 5]` | 8 | 0,028 | $\sigma \approx 1{,}42$ ($P \approx 0{,}76$) |
| `[0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]` | 10 | 0,010 | $\sigma \approx 2{,}42$ |
| `[0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]` | 13 | 0,005 | $\sigma \approx 1{,}73$ |

Conséquence : l'écart de relaxation vérifie $\overline{\mathrm{UB}} - \mathrm{OPT} \le \varepsilon_{\max} \sum_i w_i$. **Retenu : $K = 10$**, soit au plus 1 % du poids total. Avec la liste à 8 points prévue au départ, l'erreur maximale tombait précisément dans la zone de $P$ comprise entre 0,6 et 0,9, qui est celle qui nous intéresse.

### 4.5 Taille et forme creuse

On garde $\sigma_i$ comme **variable explicite**, avec une égalité par cellule. Si on la substituait dans les coupes, chacune recopierait toute la ligne $\lambda_{i\cdot}$, et le nombre de coefficients non nuls serait multiplié par $K$.

| Bloc | Variables | Contraintes | Non-zéros |
|---|---|---|---|
| $x$, $y$ | $\lvert J\rvert + \lvert S\rvert$ | — | — |
| $\sigma$, $z$ | $2\lvert I\rvert \approx 20\,000$ | — | — |
| Définition de $\sigma$ | — | $\lvert I\rvert$ | $\mathrm{nnz}(A) + \lvert I\rvert$ |
| Coupes | — | $K\lvert I\rvert \approx 100\,000$ | $2K\lvert I\rvert$ |
| (C1)–(C4) | — | $\approx \lvert J\rvert + \lvert C\rvert + 2\lvert S\rvert$ | $\approx 3\lvert J\rvert$ |

Solveur : `scipy.optimize.milp`, qui s'appuie sur HiGHS et accepte des matrices creuses.

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

Le glouton sert donc de **référence rapide et de borne inférieure**. Seul le MILP certifie la qualité d'une solution (§4.3).

## 6. Protocole expérimental

1. Balayer le budget de `budget.min` à `budget.max` par pas de `budget.pas`.
2. Pour chaque $B$, enregistrer :
   - $F(x^{\text{glouton}})$ ;
   - $F(\hat{x})$ ;
   - $\overline{\mathrm{UB}}$ ;
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
- R. Church, C. ReVelle (1974). *The maximal covering location problem.* Papers of the Regional Science Association 32, 101–118.

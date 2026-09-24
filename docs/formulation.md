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

## 7. Évaluation par l'intrus (semaine 4)

Les sections 3 à 6 maximisent une couverture **moyenne**. Un intrus qui connaît le dispositif ne se déplace pas au hasard : il choisit le chemin le moins surveillé. Cette section définit ce chemin et deux façons de mesurer sa détection. Les paramètres sont dans la section `intrus` de `config.yaml`.

### 7.1 Diagnostic préalable : l'intérieur ne suffit pas

Sur les solutions MILP de la semaine 3, avec un intrus qui part de la clôture et vise Bat.3, la détection sur le pire chemin est **nulle à tous les budgets** (B = 4, 13 et 30), pour un chemin d'une seule cellule. Un coin de Bat.3 est à 1,18 m de la clôture : l'intrus franchit le grillage et touche le mur sans traverser aucune cellule intérieure. Un actif critique collé à la clôture ne peut pas être protégé par des capteurs qui ne surveillent que l'intérieur. Le domaine de l'intrus doit donc inclure une **bande extérieure**.

### 7.2 Domaine et graphe

| Élément | Définition | Paramètre |
|---|---|---|
| Nœuds $V$ | cellules sol de l'intérieur (les $I$ de la §1, soit 9 973) ∪ cellules sol extérieures à moins de $b$ de la clôture | `bande_ext_m` $= b$ |
| Obstacles | bâti (`bati_mask`), emprises des actifs, cellules intérieures hors masque sol : absents du graphe | — |
| Arêtes $E$ | 8-connexité, de longueur $\ell_{uv} \in \{1, \sqrt 2\}$ m ; une diagonale est interdite si ses deux voisins orthogonaux sont des obstacles (pas de passage « entre deux coins ») | — |
| Arêtes de franchissement $E_\times \subset E$ | arêtes dont le segment coupe la ligne de clôture | — |
| Portails $E_\text{p} \subset E_\times$ | arêtes de franchissement qui coupent l'ouverture d'un portail : segment de clôture centré sur le point `acces`, de largeur lue dans son type (« Portail 3m ») | — |
| Sources $V_s$ | cellules extérieures à une distance de la clôture $\ge b - 1{,}5$ m (bord extérieur de la bande) | — |
| Cibles $V_t$ | cellules **intérieures** à moins de $d_t$ de l'emprise de l'actif `cible_intrus` | `cible_contact_m` $= d_t$ |

Les cibles doivent être intérieures : une cellule extérieure peut se trouver à 1,3 m de Bat.3, et la compter comme cible ferait gagner l'intrus sans franchir la clôture.

La clôture est portée par des **arêtes**, pas par des cellules. Un scénario d'intrusion choisit donc simplement un sous-ensemble d'arêtes de franchissement autorisées.

**Probabilité par cellule.** Pour une solution $T$, $P_v(T) = 1 - q_v(T)$ avec $q_v(T) = \prod_{j \in T}(1 - P_{vj})$. On garde l'hypothèse d'indépendance entre capteurs de la §3.1. Les $P_{vj}$ sont calculés par les mêmes modèles de capteurs sur l'ensemble $V$ et stockés dans une matrice distincte `A_intrus.npz`. `A.npz` n'est pas modifiée : la semaine 3 reste reproductible.

### 7.3 Deux profils d'intrus

| Profil | Franchissements autorisés | Temps sur place au franchissement |
|---|---|---|
| **Déterminé** ($\mathcal{F}$) | toute arête de $E_\times$ | $t_\text{c}$ sur la clôture, $t_\text{p}$ à un portail |
| **Opportuniste** ($\mathcal{G}$) | portails seulement ($E_\text{p}$) | $t_\text{p}$ |

Tout chemin de $\mathcal{G}$ est un chemin de $\mathcal{F}$, avec le même coût. Les deux métriques ci-dessous vérifient donc toujours $\text{métrique}_\mathcal{F} \le \text{métrique}_\mathcal{G}$ : le profil déterminé est le **pire cas**.

Le franchissement par un portail est jugé plus probable, parce que les portails prolongent les axes d'approche naturels. On note $\pi$ (`pi_portail`) la part d'intrus opportunistes, et on définit l'indicateur de mélange :

$$
D_\pi = \pi \cdot m_\mathcal{G} + (1 - \pi) \cdot m_\mathcal{F}.
$$

**Mise en garde.** $D_\pi$ est une espérance sur une *population* d'intrus, pas une garantie. Une solution qui laisse la clôture sans surveillance ($m_\mathcal{F} = 0$) mais verrouille les portails ($m_\mathcal{G} = 1$) obtient $D_{0{,}8} = 0{,}8$, alors qu'un intrus déterminé passe à coup sûr. La doctrine de conception d'un système de protection physique raisonne sur une **menace de référence** (*design basis threat*, Garcia 2007), donc sur le pire cas $m_\mathcal{F}$. $D_\pi$ sert à l'analyse de risque, pas au dimensionnement.

### 7.4 Métrique 1 : seuil du couloir $p^\star$

$$
p^\star = \min_{\gamma : V_s \to V_t} \ \max_{v \in \gamma} P_v.
$$

**Proposition (borne inférieure).** Quelle que soit la corrélation entre les détections le long d'un chemin, l'intrus qui suit son meilleur chemin est détecté avec une probabilité $\ge p^\star$.

*Preuve.* Soit $D_v$ l'événement « détecté en $v$ ». Pour tout chemin $\gamma$, l'événement « détecté sur $\gamma$ » contient chaque $D_v$, $v \in \gamma$, donc $\Pr \ge \max_{v \in \gamma} P_v \ge p^\star$. ∎

Seul le modèle par cellule intervient. Il n'y a ni paramètre de vitesse, ni hypothèse sur la succession des observations. $p^\star$ ne dépend pas des temps de franchissement.

**Calcul par percolation.** Soit $G_t$ le sous-graphe des nœuds tels que $P_v \le t$. La famille $G_t$ est croissante en $t$, et $p^\star$ est le plus petit $t$ pour lequel $G_t$ relie $V_s$ à $V_t$. On l'obtient exactement en une passe : les nœuds sont ajoutés par $P$ croissante dans une structure union-find, avec une super-source reliée à $V_s$ et une super-cible reliée à $V_t$. $p^\star$ est la détection du nœud dont l'ajout les relie. Le coût est en $O(\lvert E\rvert\,\alpha(\lvert V\rvert))$ après le tri, soit environ 14 ms sur le site. Un test compare le résultat à une dichotomie sur les composantes connexes. Le couloir est ensuite tracé par un plus court chemin, en mètres, dans $G_{p^\star}$.

Interprétation : **il existe un couloir dans lequel la détection ne dépasse jamais $p^\star$.** C'est la notion de *maximal breach path* de la couverture de barrière (Meguerdichian et al., 2001a).

### 7.5 Métrique 2 : exposition $\Lambda$

**Modèle.** Un capteur fait une observation indépendante toutes les $\tau_\text{obs}$ secondes. L'intrus avance à vitesse $v$ et reste $\Delta t_u$ secondes dans la cellule $u$ : il y subit $n_u = \Delta t_u / \tau_\text{obs}$ observations, qu'il esquive chacune avec la probabilité $q_u$. Avec $\sigma_u = -\ln q_u$ :

$$
\Pr[\text{non-détection sur } \gamma] = \prod_{u \in \gamma} q_u^{\,n_u}
= \exp\Big(-\sum_{u \in \gamma} \sigma_u \frac{\Delta t_u}{\tau_\text{obs}}\Big) = e^{-\Lambda(\gamma)}.
$$

**Poids des arêtes.** Le temps d'une arête est réparti à parts égales entre ses deux cellules (méthode des trapèzes) :

$$
c_{uv} = \frac{\sigma_u + \sigma_v}{2} \cdot \frac{1}{\tau_\text{obs}} \Big( \frac{\ell_{uv}}{v} + t_\times(u,v) \Big),
\qquad
t_\times = \begin{cases} t_\text{c} & (u,v) \in E_\times \setminus E_\text{p} \\ t_\text{p} & (u,v) \in E_\text{p} \\ 0 & \text{sinon.} \end{cases}
$$

$\Lambda^\star = \min_\gamma \Lambda(\gamma)$ se calcule par un Dijkstra multi-source (`scipy.sparse.csgraph.dijkstra`, sources $V_s$, `min_only=True`). On en déduit $P_\text{chemin} = 1 - e^{-\Lambda^\star}$.

**Invariance à la résolution.** $\Lambda$ est une somme de Riemann de $\frac{1}{v\,\tau_\text{obs}} \int_\gamma \sigma \, ds$. À l'inverse, $1 - \prod_{u \in \gamma}(1 - P_u)$, qui compte une observation par cellule, double quand on divise la taille des cellules par deux. C'est la *exposure* de Meguerdichian et al. (2001b).

**Limite.** Deux observations successives d'un même capteur ne sont pas indépendantes : même occultation, même angle de vue. $\Lambda$ surestime donc la détection. Mesuré sur les solutions de la semaine 3 (profil portails, B = 13), $P_\text{chemin}$ passe de 0,78 à 1,00 quand $v\,\tau_\text{obs}$ passe de 30 m à 3 m. **$p^\star$ est donc la métrique de conception, et $\Lambda$ un indicateur**, accompagné d'une analyse de sensibilité sur $\tau_\text{obs}$.

### 7.6 Récapitulatif par solution

| Indicateur | Rôle |
|---|---|
| $p^\star_\mathcal{F}$ | **pire cas garanti** : objectif de NSGA-II (option `pire_cas`) |
| $p^\star_\mathcal{G}$ | profil opportuniste |
| $D_\pi$ | mélange, pour l'analyse de risque (option `melange`) |
| $P_\text{chemin}$ ($\mathcal{F}$, $\mathcal{G}$) | pour chaque $\tau_\text{obs}$ de `tau_obs_sensibilite_s` |
| Couloir et chemin | tracés, avec leur point de franchissement de la clôture |

### 7.7 Hypothèses et limites

1. **Intrus omniscient.** Il connaît le dispositif, et en particulier les capteurs et leurs orientations. Hypothèse pessimiste, cohérente avec une menace de référence.
2. **Vitesse constante, intrus debout** (cible à 1,0 m, §viewshed). Ramper réduit la silhouette et la vitesse : hors modèle.
3. **Toute détection dans la bande extérieure compte.** En pratique, sur une voie publique, les fausses alarmes conduisent souvent à désactiver l'analyse. La largeur $b$ est un paramètre de sensibilité ; $b = 0$ reproduit le cas dégénéré de la §7.1.
4. **Pas de capteur de clôture** (détection périmétrique sur le grillage) : seuls les capteurs du catalogue existent.
5. **Détection et non interruption.** On ne vérifie pas que la détection arrive assez tôt pour que la réponse intervienne avant l'atteinte de la cible (détection opportune, modèle EASI ; Garcia, 2007). Ce point est reporté à la semaine 7.

### 7.8 Tests (`tests/test_intrus.py`)

| Test | Solution de référence |
|---|---|
| Couloir de P connu entre deux murs | $p^\star$ = P du couloir |
| Carte suréchantillonnée ×2 (cellules de 0,5 m) | même $\Lambda^\star$ à la discrétisation près ; le produit par cellule, lui, change |
| Nœud à la fois départ et cible | $\Lambda^\star = 0$, $p^\star = P_v$ |
| Deux cellules de part et d'autre de la clôture | $\Lambda^\star = \sigma(\ell/v + t_\text{c})/\tau_\text{obs}$, ou $t_\text{p}$ au portail |
| Colonne à détection certaine ($\sigma = \infty$) | $\Lambda^\star = \infty$ |
| Union-find | égal à la dichotomie sur 20 instances aléatoires |
| Goulot unique | le chemin passe par le goulot |
| Clôture fermée sauf un portail | $m_\mathcal{G} = m_\mathcal{F}$ quand le portail est le seul passage ; $m_\mathcal{F} \le m_\mathcal{G}$ sur une instance aléatoire |
| Diagonale entre deux obstacles | chemin refusé |

## 8. Modèle de substitution : carte de potentiel d'implantation (semaines 5–6)

Les sections 3 à 7 évaluent 176 couples (position, hauteur) choisis à la main en semaine 1. Cette section remplace la question « quelle visibilité depuis ce candidat ? » par « quelle visibilité depuis **chaque** position de la zone ? ». Un réseau de neurones y apporte un gain de structure, et pas seulement de constante. Les paramètres sont dans la section `surrogate` de `config.yaml`.

### 8.1 Diagnostic préalable : pas de goulot sur le catalogue

Le plan initial prévoyait un U-Net qui prédit la carte de visibilité d'un observateur, pour accélérer le calcul des viewsheds. Les mesures ont été faites sur la même machine (2 cœurs CPU) pour que les rapports soient comparables.

| Mesure | Valeur |
|---|---|
| Calculs de τ nécessaires au catalogue | 176 : τ ne dépend ni du capteur ni de l'orientation (§ semaine 2) |
| Part de τ dans la construction de `A` | 96 % (42 s sur 43,5 s) |
| τ sur le disque de 400 m | 220 ms |
| τ sur les 27 905 cellules du domaine seulement | 6,4 ms, identique bit à bit |
| U-Net 1,9 M paramètres (7,8 M), une carte 816 × 816 | 582 ms (2 070 ms) |

1. **Il n'y a pas de goulot à accélérer.** Les « milliers de viewsheds » n'existent pas : l'orientation et le type de capteur sont analytiques.
2. **À résolution égale, un U-Net est 2,6 à 9 fois plus lent que le moteur exact**, et il n'est pas exact. Un surrogate par observateur serait du ML plaqué.

Le coût exact ne devient prohibitif que quand on veut la visibilité depuis **toutes** les positions d'une zone. C'est le problème du *total viewshed*, dont le coût est en $O(N R^2)$ pour $N$ positions et une portée de $R$ cellules (Tabik et al., 2015). Sur l'emprise actuelle (411 640 positions valides, 5 hauteurs, $R$ = 150 m), il faut environ 8 h sur 2 cœurs, soit 70 ms par position. Un réseau entièrement convolutif produit toute la carte en **une passe**, en $O(N)$.

### 8.2 Cible

Soit un observateur $o$ à la hauteur $h$ au-dessus de la surface opaque ($z_o = z_\text{opaque}(o) + h$ : un toit ou le sol). Le disque de rayon $R$ autour de $o$ est découpé en bins $(a, s)$ : des anneaux de distance $a$ et des secteurs d'azimut $s$ de $360°/n_s$, centrés sur les azimuts des caméras. On définit :

$$
V_{h,a,s}(o) = \frac{1}{\lvert \mathcal{B}_{a,s} \rvert} \sum_{j \in \mathcal{B}_{a,s} \cap \text{Sol}} T_{oj}, \qquad T_{oj} = e^{-\tau_{oj}} \in [0, 1],
$$

où $\lvert \mathcal{B}_{a,s} \rvert$ est le nombre de cellules du bin, qui ne dépend pas du terrain. $V$ est donc la **part de la surface du bin qui est du sol visible**, pondérée par la transmittance, dans $[0, 1]$.

| Choix | Valeur | Justification |
|---|---|---|
| $R$ | 150 m | $R_{90}$ du radar ; caméra télé : $r_\text{design} \approx$ 143 m |
| Anneaux | 0–50, 50–150 m | grand angle ≈ 38 m, standard ≈ 67 m : sépare les usages courts et longs |
| Secteurs | 8 × 45° | centrés sur `camera_azimuths_deg` : $V_{h,\cdot,s}$ note directement une caméra orientée en $s$ |
| Hauteurs | 2, 4, 6, 8, 10 m | union des hauteurs de toit et de mât du catalogue |
| $T$ | $e^{-\tau}$, $\alpha = 1$ | aller simple (caméra). La cible est **géométrique** : les lois de détection restent analytiques et s'appliquent ensuite |

Soit 80 sorties par position. $\tau$ est calculé par le noyau `ray_tau` de `src/viewshed.py` : les étiquettes sont exactes par construction, et un test le vérifie.

**Propriétés utilisées ensuite.**

1. **Équivariance par translation.** $V$ ne dépend que du terrain autour de $o$. La carte $o \mapsto V(o)$ est donc un opérateur équivariant par translation des rasters : c'est exactement ce que représente un réseau entièrement convolutif. Son champ récepteur doit couvrir au moins $2R + 1$ = 301 cellules.
2. **Invariance verticale.** Décaler toutes les altitudes d'une constante ne change pas $V$. Les entrées sont donc des altitudes relatives.
3. **Équivariance D4.** Une rotation de 90° du terrain décale les secteurs de $+2$ ; un miroir est-ouest les envoie de $s$ vers $-s \bmod 8$. Les 8 éléments du groupe diédral agissent exactement sur la grille : ils fournissent une augmentation de données sans approximation (Cohen et Welling, 2016). Deux tests le vérifient.
4. **Monotonie en $h$ : seulement sans végétation.** Sans végétation, le rayon monte en tout point quand $z_o$ monte, donc $V$ est non décroissant en $h$. Avec végétation, un rayon entièrement sous la canopée parcourt $\sqrt{1 + \text{pente}^2}$ m par mètre horizontal, et la pente augmente avec $h$ : $V$ peut décroître. Le test le montre sur une colonne de canopée. **Le réseau ne doit donc pas être contraint à être monotone.**

### 8.3 Pourquoi pas une carte par observateur

Prédire $T_{oj}$ pour tout $j$ du disque revient à produire une image par observateur. On retombe alors sur le coût de 8.1, avec en plus une cible dominée par les zéros : sur un candidat typique, 86 % du disque est occulté, et la transmittance médiane des rayons qui traversent de la végétation vaut 0,01. L'agrégation par bins garde l'information utile à la décision (quelle surface, dans quelle direction, à quelle distance) et rend la sortie dense et lisse.

### 8.4 Données et découpage spatial

| Ensemble | Définition | Taille |
|---|---|---|
| Positions valides | disque complet dans la grille (à plus de $R$ du bord) | 411 640 |
| **Test** | emprise du site + 50 m ; calcul exact un pixel sur deux | 71 980 positions, dont 18 056 étiquetées |
| Tampon | 30 m autour du bloc test, exclu | — |
| Validation | tuiles de 100 m tirées au hasard (15 %) hors tampon | 52 300 positions |
| Entraînement | le reste | 287 360 positions |

- **Échantillonnage stratifié** : 30 000 observateurs (entraînement et validation), répartis en toits (25 %), sol dégagé (40 %) et sol sous végétation (35 %). Tirer uniformément sous-représenterait les toits (23 % de la zone), qui sont pourtant des supports privilégiés.
- **Le réseau prédit une carte dense, mais la perte n'est évaluée qu'aux pixels étiquetés** (supervision creuse, environ 9 % des positions). Elle n'est comptée que sur les pixels dont tout le champ récepteur est dans la tuile, pour ne pas apprendre les effets de bord du rembourrage.
- **Pourquoi un découpage spatial.** Deux positions voisines de 1 m ont des étiquettes presque identiques. Un tirage aléatoire des cellules mesurerait une interpolation, pas une généralisation (Roberts et al., 2017). Le bloc test contient le site : c'est aussi là qu'on validera la décision (§8.5).
- **Fuite résiduelle, assumée.** Avec $R$ = 150 m et une emprise d'environ 1 km², les disques des positions de test chevauchent du terrain vu à l'entraînement. Le test mesure donc la généralisation à des **positions** nouvelles sur un terrain en partie connu, pas à un site nouveau. Le tampon de 30 m supprime l'autocorrélation directe des étiquettes, pas celle des entrées.

### 8.5 Évaluation

**Références de comparaison.** Un score de rang de 0,9 ne veut rien dire sans point de comparaison. Le réseau est donc comparé à deux références :
- **B0** : la moyenne par strate et par hauteur ;
- **B1** : un gradient boosting sur des descripteurs locaux faits à la main (relief relatif, fraction de bâti et de végétation par anneau et par secteur).

**Précision**, sur le bloc test :
- erreur absolue moyenne sur $V$, par hauteur, strate et anneau ;
- corrélation de rang de Spearman sur le potentiel total $\bar V_h = \sum_{a,s} \lvert \mathcal{B}_{a,s}\rvert V_{h,a,s} / \sum \lvert\mathcal{B}\rvert$ ;
- **rappel top-K** : parmi les K meilleures positions exactes (après suppression des non-maxima à 10 m) sur les supports possibles du site, part retrouvée dans le top-K prédit.

**Accélération.** Soit $c_e$ et $c_n$ les coûts par position du calcul exact et du réseau, et $C_0$ le coût fixe du jeu de données et de l'entraînement. Le réseau est rentable au-delà de

$$
N^\star = \frac{C_0}{c_e - c_n}
$$

positions, exprimé en km² à 1 m. Tout est mesuré sur la même machine, sur CPU et sur MPS.

**Décision (étape 6.3).** On génère des candidats à partir de la carte prédite : top-K par support possible, avec suppression des non-maxima. On construit leur matrice `A` avec le moteur **exact**, puis on relance le glouton et NSGA-II. On compare F, $p^\star_\mathcal{F}$ et le front au catalogue de la semaine 1, et à un catalogue tiré de la carte **exacte** du bloc test (borne supérieure de ce que le réseau peut apporter).

### 8.6 Hypothèses et limites

1. **Une seule zone d'environ 1 km²**, en conditions sans feuilles (novembre) : la généralisation à un autre site ou à la saison feuillée n'est pas testée.
2. **Cible géométrique** : $\alpha = 1$ pour tous les capteurs. Le radar ($\alpha_r = 0{,}3$, aller-retour) voit à travers la végétation autrement ; la cible sert à **classer** les positions, et le calcul exact reprend la main pour la décision.
3. **Hauteurs discrètes** (5 valeurs), comme le catalogue.
4. **Positions à moins de $R$ du bord exclues** : la carte prédite n'est fiable qu'à l'intérieur de l'emprise valide.

### 8.7 Tests (`tests/test_surrogate_data.py`)

| Test | Solution de référence |
|---|---|
| Noyau `potentiel` contre agrégation de `optical_depth` (3 terrains aléatoires, 3 hauteurs) | égalité à 10⁻⁶ |
| Terrain plat sans végétation | $V = 1$ dans chaque bin |
| Géométrie des bins | nord = secteur 0, est = secteur 2 ; secteurs équilibrés à 10 % près |
| Rotation de 90° du terrain | secteurs décalés de $+2$ |
| Miroir est-ouest | $s \to -s \bmod 8$ |
| Sans végétation | $V$ non décroissant en $h$ |
| Colonne de canopée | $V$ décroît de $h$ = 2 à 5 m (contre-exemple de la monotonie) |
| Découpage | site entièrement en test ; aucune position d'entraînement à moins de 30 m du test ; disques complets |

## Références

- G. L. Nemhauser, L. A. Wolsey, M. L. Fisher (1978). *An analysis of approximations for maximizing submodular set functions — I.* Mathematical Programming 14, 265–294.
- S. Khuller, A. Moss, J. Naor (1999). *The budgeted maximum coverage problem.* Information Processing Letters 70(1), 39–45.
- M. Sviridenko (2004). *A note on maximizing a submodular set function subject to a knapsack constraint.* Operations Research Letters 32(1), 41–43.
- G. L. Nemhauser, L. A. Wolsey (1981). *Maximizing submodular set functions: formulations and analysis of algorithms.* Annals of Discrete Mathematics 11, 279–301.
- R. Church, C. ReVelle (1974). *The maximal covering location problem.* Papers of the Regional Science Association 32, 101–118.
- M. L. Garcia (2007). *Design and Evaluation of Physical Protection Systems*, 2nd ed. Butterworth-Heinemann.
- S. Meguerdichian, F. Koushanfar, M. Potkonjak, M. B. Srivastava (2001a). *Coverage problems in wireless ad-hoc sensor networks.* IEEE INFOCOM 2001, 1380–1387.
- S. Meguerdichian, F. Koushanfar, G. Qu, M. Potkonjak (2001b). *Exposure in wireless ad-hoc sensor networks.* ACM MobiCom 2001, 139–150.
- E. Israeli, R. K. Wood (2002). *Shortest-path network interdiction.* Networks 40(2), 97–111.
- S. Tabik, A. R. Cervilla, E. Zapata, L. F. Romero (2015). *Efficient data structure and highly scalable algorithm for total-viewshed computation.* IEEE JSTARS 8(1), 304–310.
- O. Ronneberger, P. Fischer, T. Brox (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015, 234–241.
- T. Cohen, M. Welling (2016). *Group equivariant convolutional networks.* ICML 2016, 2990–2999.
- D. R. Roberts et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40(8), 913–929.

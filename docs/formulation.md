# Formulation of the sensor placement problem

This document is the mathematical specification of the project. It started as the specification of phase 3 and is the reference for `src/greedy.py`, `src/milp.py` and their tests; sections 7 to 9 add the intruder, the surrogate model and the adversarial loop. Each parameter refers to a key of `config.yaml`.

The [notebooks](../notebooks) illustrate each section with small examples and figures on the example site.

---

## 1. Data

| Symbol | Meaning | Source |
|---|---|---|
| $I$ | weighted cells of the site, indexed by $i$ | `coverage/cells.npz` |
| $w_i \ge 0$ | weight of cell $i$ (zone × area) | `cells.npz["w"]` |
| $J$ | sensor configurations, indexed by $j$: a sensor, a support, an azimuth, a tilt | `coverage/configs.csv` |
| $P_{ij} \in [0,1]$ | probability that configuration $j$ detects an intruder at $i$ (visibility × sensor model) | `coverage/A.npz`, stored **transposed**: `A[j, i]` |
| $C$ | physical locations (column `cand`) | `configs.csv` |
| $S \subseteq C \times H$ | supports $s = (c, h)$: a location and a height | `configs.csv` (`cand`, `h`) |
| $s(j)$ | support that carries configuration $j$ | `configs.csv` |
| $\kappa(j) \in \{\text{radar}, \text{camera}\}$ | sensor type of $j$, read from the `sensors` section | `config.yaml` |
| $a_j$ | cost of the sensor | `sensor_cost` |
| $f_s$ | fixed cost of opening the support (mast or roof mount) | `support_cost` |
| $B$ | budget | `optimization.budget` |
| $K_\kappa$ | capacity of a support for type $\kappa$ (1 radar, 4 cameras) | `optimization.support_capacity` |

The values $P_{ij} < p_{\min}$ were set to zero in week 2 (`coverage.p_min`, measured loss: 0.011 % of the weight).

## 2. Decision variables

- $x_j \in \{0,1\}$: configuration $j$ is installed.
- $y_s \in \{0,1\}$: support $s$ is built.
- $\eta_g \ge 0$: upper bound of the weighted coverage of block $g$ in the master problem (§4.2).

## 3. Exact model

### 3.1 Objective

We assume that, for an intruder located at $i$, the detections of the different sensors are **independent** events. The intruder only escapes detection if they escape every installed sensor:

$$
\Pr[\text{no detection at } i] = \prod_{j \in J} (1 - P_{ij})^{x_j}.
$$

We maximise the expected detected weight:

$$
\max_{x, y} \; F(x) = \sum_{i \in I} w_i \left( 1 - \prod_{j \in J} (1 - P_{ij})^{x_j} \right).
$$

### 3.2 Constraints

**(C1) Budget.** The cost of a support is paid only once, whatever the number of sensors it carries:

$$
\sum_{j \in J} a_j x_j + \sum_{s \in S} f_s y_s \le B.
$$

**(C2) A sensor requires its support.**

$$
x_j \le y_{s(j)} \qquad \forall j \in J.
$$

**(C3) A single height per location.** A 4 m mast and a 10 m mast are not placed at the same spot:

$$
\sum_{h : (c,h) \in S} y_{(c,h)} \le 1 \qquad \forall c \in C.
$$

**(C4) Capacity per type.** At most one radar, which needs the clear top, and at most four cameras per support:

$$
\sum_{j : s(j) = s,\ \kappa(j) = \kappa} x_j \le K_\kappa \, y_s \qquad \forall s \in S,\ \forall \kappa.
$$

Thanks to the factor $y_s$, (C4) already implies (C2) for integer solutions. We keep (C2) anyway: in the continuous relaxation, $x_j \le y_s$ is tighter than $\sum x_j \le K_\kappa y_s$, which improves the bound and speeds up the branch-and-bound.

### 3.3 Assumptions and limitations

1. **Conditional independence.** The detections of two sensors aiming at the same cell are assumed independent. Real correlations (fog, rain, crawling intruder) make this assumption optimistic: the true combined detection lies between $\max_j P_{ij}$ and $1 - \prod_j (1 - P_{ij})$. Both extreme cases can serve as a sensitivity analysis.
2. **Static visibility.** $P_{ij}$ comes from a DSM frozen at the time of the LiDAR flight: no seasonal vegetation, no parked vehicles.
3. **Expectation, not worst case.** We maximise the weighted mean detection. An adversary who knows the layout chooses their path: this is the question of path coverage, handled in §7.

## 4. Exact solution

### 4.1 Compact model with tangent cuts: correct but intractable

The first idea is to linearise the objective directly. Let $\lambda_{ij} = -\ln(1 - \min(P_{ij}, 0.999))$ and $\sigma_i = \sum_j \lambda_{ij} x_j$. We then have the exact identity $\prod_j (1 - P_{ij})^{x_j} = e^{-\sigma_i}$. Since $g(\sigma) = 1 - e^{-\sigma}$ is concave, it is bounded above by tangents:

$$
z_i \le 1 - e^{-\sigma_k} + e^{-\sigma_k}(\sigma_i - \sigma_k) \quad \forall i, k.
$$

With 10 tangent points, the overestimation stays below 0.010 per cell. The model therefore gives a valid upper bound.

The model is as large as $A$ (every non-zero of $A$ appears in it), and its continuous relaxation is weak: fractions $x_j = 0.3$ spread over many configurations "see" almost the whole site, so the root bound stays far from the best known solution and every node of the tree is slow. **Model rejected.**

### 4.2 Decomposition by multi-block submodular cuts

We cut the cells into spatial blocks $g$, squares of side `optimization.milp.block_m`, and write $F = \sum_g F_g$ with

$$
F_g(T) = \sum_{i \in g} w_i \Big(1 - \prod_{j \in T} (1 - P_{ij})\Big).
$$

Each $F_g$ is a sum of monotone submodular functions (§5.1): it is therefore itself monotone submodular.

**Nemhauser and Wolsey inequality.** For all $S, T \subseteq J$:

$$
F_g(T) \le F_g(S \cup T) \le F_g(S) + \sum_{j \in T \setminus S} \rho_{gj}(S),
\qquad \rho_{gj}(S) = \sum_{i \in g} w_i \, q_i(S) \, P_{ij}.
$$

The first inequality follows from monotonicity. The second is obtained by adding the elements of $T \setminus S$ one by one: by submodularity, each gain is at most the gain computed at $S$.

**Master problem**, on $(x, y, \eta)$ only, i.e. $\lvert J\rvert + \lvert S\rvert + G$ variables:

$$
\max \sum_g \eta_g \quad \text{s.t.} \quad \text{(C1)-(C4)}, \qquad
\eta_g \le F_g(S_k) + \sum_{j \notin S_k} \rho_{gj}(S_k)\, x_j \quad \forall g,\ \forall S_k \in \mathcal{S},
\qquad 0 \le \eta_g \le \textstyle\sum_{i \in g} w_i.
$$

**Algorithm.**

1. Initialise the pool $\mathcal{S}$ with $\emptyset$, the successive sets of the greedy and the optimum of the previous budget. The cuts do not depend on $B$: they remain valid from one budget to the next.
2. Solve the master, which gives $T_k$ and the dual bound $\mathrm{UB}_k$.
3. Evaluate $F(T_k)$ exactly, then set $\mathrm{LB} = \max(\mathrm{LB}, F(T_k))$.
4. Add the $G$ cuts at $S = T_k$, then go back to step 2 as long as $(\mathrm{UB} - \mathrm{LB})/\mathrm{UB} >$ `target_gap`.

### 4.3 Validity and convergence

- **Upper bound.** For every feasible $T$, $\eta_g = F_g(T)$ satisfies every cut. The master therefore has a value $\ge F(T)$, in particular $\ge \mathrm{OPT}$. We keep the **dual bound** of HiGHS, which stays valid even when the master stops on its time limit.
- **Lower bound.** Each $T_k$ is feasible, since it satisfies (C1)-(C4). $F(T_k)$ is computed exactly, without logarithm, capping or tangent.
- **Finite convergence.** If the master returns a $T_k$ already in the pool, its cut imposes $\eta_g \le F_g(T_k)$, hence $\mathrm{UB} = F(T_k) = \mathrm{LB}$: $T_k$ is optimal. The number of sets being finite, the algorithm terminates.

### 4.4 Why several blocks

With a single block ($G = 1$), we recover the original algorithm of Nemhauser and Wolsey. A cut then aggregates the whole site: the master can "borrow" the coverage promised to one zone and credit it to another. The smaller the blocks, the stronger the cuts, but the bigger the master.

Each multi-block cut is **sparse**, since only the configurations that see block $g$ appear in it.

## 5. Greedy heuristic

### 5.1 Submodularity

For fixed supports, $F$ is a set function $F(T)$ on $T \subseteq J$. The marginal gain of $j \notin T$ is

$$
\Delta_j(T) = F(T \cup \{j\}) - F(T) = \sum_i w_i \, P_{ij} \prod_{k \in T} (1 - P_{ik}) = \sum_i w_i \, q_i(T) \, P_{ij},
$$

where $q_i(T)$ is the current probability of non-detection of cell $i$. Since $q_i(T)$ decreases when $T$ grows, $\Delta_j(T)$ is decreasing in $T$: $F$ is **monotone and submodular** (diminishing returns).

**Practical consequence.** All the gains are computed in a single sparse matrix × vector product:

$$
\Delta = A \,(w \odot q) \in \mathbb{R}^{\lvert J\rvert},
\qquad \text{then, after choosing } j^\star : \ q \leftarrow q \odot (1 - A_{j^\star,\cdot}).
$$

### 5.2 Algorithm

```
q ← 1, T ← ∅, open supports ← ∅, spent ← 0
repeat
    Δ ← A @ (w * q)
    cost_j ← a_j + f_{s(j)} · [s(j) not open]
    admissible_j ← spent + cost_j ≤ B
                   ∧ no other height open at c(j)                   (C3)
                   ∧ capacity κ(j) of s(j) not reached              (C4)
                   ∧ j ∉ T
    if no admissible j with Δ_j > 0: stop
    j* ← argmax_{j admissible} Δ_j / cost_j
    T ← T ∪ {j*}, open s(j*), spent += cost_j*, q ← q ⊙ (1 − A[j*])
return the best of T and the best singleton {j} such that a_j + f_{s(j)} ≤ B
```

### 5.3 Guarantees and lack of guarantee

| Setting | Guarantee | Reference |
|---|---|---|
| Cardinality ($\lvert T\rvert \le k$), greedy on $\Delta$ | $1 - 1/e \approx 0.632$ | Nemhauser, Wolsey and Fisher (1978) |
| Budget with **additive** costs, ratio greedy alone | none (arbitrarily bad) | n/a |
| Additive budget, best of ratio and singleton | $\ge \tfrac{1}{2}(1 - 1/e) \approx 0.316$ | Khuller, Moss and Naor (1999) |
| Additive budget, partial enumeration of triples | $1 - 1/e$ | Sviridenko (2004) |
| **Our case**: shared support costs + (C3) + (C4) | **none** | n/a |

Two mechanisms break the guarantee in our case:

- The cost of a configuration depends on what is already chosen, because the support may already be paid. The cost function is therefore no longer modular.
- The first choice of height at $c$ locks the location (C3). The greedy never goes back on a decision.

The greedy therefore serves as a **fast reference and lower bound**. Only the decomposition of §4 certifies the quality of a solution.

## 6. Experimental protocol

1. Sweep the budget from `budget.min` to `budget.max` with step `budget.step`.
2. For each $B$, record:
   - $F(x^{\text{greedy}})$;
   - $F(\hat{x})$, the best solution of the decomposition;
   - $\mathrm{UB}$, the certified upper bound;
   - the computation time;
   - the composition of the solution (number of radars, cameras and supports, selected heights).
3. Plot the coverage/budget front with the three curves and the band $[\mathrm{LB}, \overline{\mathrm{UB}}]$.

**Predictions to check:**

- The gap between greedy and MILP is largest at small budget, because of the height lock-in.
- At medium budget, the MILP combines a radar with cameras that cover its blind cone.
- The front has an elbow, beyond which the added sensors only bring redundancy.

## 7. Evaluation by the intruder (week 4)

Sections 3 to 6 maximise an **average** coverage. An intruder who knows the layout does not move at random: they choose the least watched path. This section defines that path and two ways of measuring its detection. The parameters are in the `intruder` section of `config.yaml`.

### 7.1 Preliminary diagnosis: the inside is not enough

If the domain of the intruder were only the interior cells, an asset close to the fence could be reached without crossing any watched cell: the intruder climbs the fence and touches the wall. A critical asset glued to the fence cannot be protected by sensors that only watch the inside. The intruder domain must therefore include an **outer band**.

### 7.2 Domain and graph

| Element | Definition | Parameter |
|---|---|---|
| Nodes $V$ | interior ground cells (the $I$ of §1) ∪ exterior ground cells closer than $b$ to the fence | `outer_band_m` $= b$ |
| Obstacles | buildings (`building_mask`), footprints of the assets, interior cells outside the ground mask: absent from the graph | n/a |
| Edges $E$ | 8-connectivity, of length $\ell_{uv} \in \{1, \sqrt 2\}$ m; a diagonal is forbidden if both its orthogonal neighbours are obstacles (no passage "between two corners") | n/a |
| Crossing edges $E_\times \subset E$ | edges whose segment cuts the fence line | n/a |
| Gates $E_\text{p} \subset E_\times$ | crossing edges that cut the opening of a gate: fence segment centred on the `access` point, of width read in its type ("Gate 3m") | n/a |
| Sources $V_s$ | exterior cells at a distance from the fence $\ge b - 1.5$ m (outer edge of the band) | n/a |
| Targets $V_t$ | **interior** cells closer than $d_t$ to the footprint of the `intruder_target` asset | `target_contact_m` $= d_t$ |

The targets must be interior: an exterior cell can lie within $d_t$ of an asset close to the fence, and counting it as a target would let the intruder win without crossing the fence.

The fence is carried by **edges**, not by cells. An intrusion scenario therefore simply chooses a subset of allowed crossing edges.

**Probability per cell.** For a solution $T$, $P_v(T) = 1 - q_v(T)$ with $q_v(T) = \prod_{j \in T}(1 - P_{vj})$. We keep the independence assumption between sensors of §3.1. The $P_{vj}$ are computed by the same sensor models on the whole set $V$ and stored in a separate matrix `A_intruder.npz`. `A.npz` is not modified: week 3 stays reproducible.

### 7.3 Two intruder profiles

| Profile | Allowed crossings | Time spent at the crossing |
|---|---|---|
| **Determined** ($\mathcal{F}$) | any edge of $E_\times$ | $t_\text{c}$ on the fence, $t_\text{p}$ at a gate |
| **Opportunistic** ($\mathcal{G}$) | gates only ($E_\text{p}$) | $t_\text{p}$ |

Every path of $\mathcal{G}$ is a path of $\mathcal{F}$, with the same cost. The two metrics below therefore always satisfy $\text{metric}_\mathcal{F} \le \text{metric}_\mathcal{G}$: the determined profile is the **worst case**.

Crossing through a gate is considered more likely, because the gates extend the natural approach axes. We write $\pi$ (`pi_gate`) for the share of opportunistic intruders, and define the mixture indicator:

$$
D_\pi = \pi \cdot m_\mathcal{G} + (1 - \pi) \cdot m_\mathcal{F}.
$$

**Warning.** $D_\pi$ is an expectation over a *population* of intruders, not a guarantee. A solution that leaves the fence unwatched ($m_\mathcal{F} = 0$) but locks the gates ($m_\mathcal{G} = 1$) gets $D_{0.8} = 0.8$, while a determined intruder gets through for sure. The design doctrine of a physical protection system reasons on a **design basis threat** (Garcia 2007), hence on the worst case $m_\mathcal{F}$. $D_\pi$ serves risk analysis, not sizing.

### 7.4 Metric 1: corridor threshold $p^\star$

$$
p^\star = \min_{\gamma : V_s \to V_t} \ \max_{v \in \gamma} P_v.
$$

**Proposition (lower bound).** Whatever the correlation between the detections along a path, the intruder who follows their best path is detected with a probability $\ge p^\star$.

*Proof.* Let $D_v$ be the event "detected at $v$". For every path $\gamma$, the event "detected on $\gamma$" contains each $D_v$, $v \in \gamma$, hence $\Pr \ge \max_{v \in \gamma} P_v \ge p^\star$. ∎

Only the per-cell model is involved. There is no speed parameter and no assumption on the succession of observations. $p^\star$ does not depend on the crossing times.

**Computation by percolation.** Let $G_t$ be the subgraph of the nodes such that $P_v \le t$. The family $G_t$ is increasing in $t$, and $p^\star$ is the smallest $t$ for which $G_t$ connects $V_s$ to $V_t$. It is obtained exactly in one pass: the nodes are added by increasing $P$ into a union-find structure, with a super-source connected to $V_s$ and a super-target connected to $V_t$. $p^\star$ is the detection of the node whose addition connects them. The cost is $O(\lvert E\rvert\,\alpha(\lvert V\rvert))$ after sorting, about 14 ms on the site. A test compares the result with a bisection on the connected components. The corridor is then drawn by a shortest path, in metres, in $G_{p^\star}$.

Interpretation: **there is a corridor in which the detection never exceeds $p^\star$.** It is the notion of *maximal breach path* of barrier coverage (Meguerdichian et al., 2001a).

### 7.5 Metric 2: exposure $\Lambda$

**Model.** A sensor makes an independent observation every $\tau_\text{obs}$ seconds. The intruder moves at speed $v$ and stays $\Delta t_u$ seconds in cell $u$: they undergo $n_u = \Delta t_u / \tau_\text{obs}$ observations there, each of which they dodge with probability $q_u$. With $\sigma_u = -\ln q_u$:

$$
\Pr[\text{no detection on } \gamma] = \prod_{u \in \gamma} q_u^{\,n_u}
= \exp\Big(-\sum_{u \in \gamma} \sigma_u \frac{\Delta t_u}{\tau_\text{obs}}\Big) = e^{-\Lambda(\gamma)}.
$$

**Edge weights.** The time of an edge is split equally between its two cells (trapezoidal rule):

$$
c_{uv} = \frac{\sigma_u + \sigma_v}{2} \cdot \frac{1}{\tau_\text{obs}} \Big( \frac{\ell_{uv}}{v} + t_\times(u,v) \Big),
\qquad
t_\times = \begin{cases} t_\text{c} & (u,v) \in E_\times \setminus E_\text{p} \\ t_\text{p} & (u,v) \in E_\text{p} \\ 0 & \text{otherwise.} \end{cases}
$$

$\Lambda^\star = \min_\gamma \Lambda(\gamma)$ is computed by a multi-source Dijkstra (`scipy.sparse.csgraph.dijkstra`, sources $V_s$, `min_only=True`). We deduce $P_\text{path} = 1 - e^{-\Lambda^\star}$.

**Resolution invariance.** $\Lambda$ is a Riemann sum of $\frac{1}{v\,\tau_\text{obs}} \int_\gamma \sigma \, ds$. Conversely, $1 - \prod_{u \in \gamma}(1 - P_u)$, which counts one observation per cell, doubles when the cell size is halved. This is the *exposure* of Meguerdichian et al. (2001b).

**Limitation.** Two successive observations of the same sensor are not independent: same occlusion, same viewing angle. $\Lambda$ therefore overestimates the detection. Measured on the solutions of week 3 (gate profile, B = 13), $P_\text{path}$ goes from 0.78 to 1.00 when $v\,\tau_\text{obs}$ goes from 30 m to 3 m. **$p^\star$ is therefore the design metric, and $\Lambda$ an indicator**, together with a sensitivity analysis on $\tau_\text{obs}$.

### 7.6 Summary per solution

| Indicator | Role |
|---|---|
| $p^\star_\mathcal{F}$ | **guaranteed worst case**: objective of NSGA-II (option `worst_case`) |
| $p^\star_\mathcal{G}$ | opportunistic profile |
| $D_\pi$ | mixture, for risk analysis (option `mixture`) |
| $P_\text{path}$ ($\mathcal{F}$, $\mathcal{G}$) | for each $\tau_\text{obs}$ of `tau_obs_sensitivity_s` |
| Corridor and path | drawn, with their fence crossing point |

### 7.7 Assumptions and limitations

1. **Omniscient intruder.** They know the layout, in particular the sensors and their orientations. Pessimistic assumption, consistent with a design basis threat.
2. **Constant speed, standing intruder** (target at 1.0 m, see `viewshed`). Crawling reduces the silhouette and the speed: outside the model.
3. **Every detection in the outer band counts.** In practice, on a public road, false alarms often lead to disabling the analysis. The width $b$ is a sensitivity parameter; $b = 0$ reproduces the degenerate case of §7.1.
4. **No fence sensor** (perimeter detection on the wire mesh): only the sensors of the catalog exist.
5. **Detection, not interruption.** We do not check that the detection arrives early enough for the response to act before the target is reached (timely detection, EASI model; Garcia, 2007). This point is handled in week 7 (§9.1).

### 7.8 Tests (`tests/test_intruder.py`)

| Test | Reference solution |
|---|---|
| Corridor of known P between two walls | $p^\star$ = P of the corridor |
| Map oversampled ×2 (cells of 0.5 m) | same $\Lambda^\star$ up to the discretisation; the per-cell product changes |
| Node that is both start and target | $\Lambda^\star = 0$, $p^\star = P_v$ |
| Two cells on either side of the fence | $\Lambda^\star = \sigma(\ell/v + t_\text{c})/\tau_\text{obs}$, or $t_\text{p}$ at the gate |
| Column with certain detection ($\sigma = \infty$) | $\Lambda^\star = \infty$ |
| Union-find | equal to the bisection on 20 random instances |
| Single bottleneck | the path goes through the bottleneck |
| Fence closed except for one gate | $m_\mathcal{G} = m_\mathcal{F}$ when the gate is the only passage; $m_\mathcal{F} \le m_\mathcal{G}$ on a random instance |
| Diagonal between two obstacles | path refused |

## 8. Surrogate model: map of the placement potential (weeks 5-6)

Sections 3 to 7 evaluate the (position, height) pairs of the catalog built by `src.candidates`. This section replaces the question "what visibility from this candidate?" with "what visibility from **every** position of the area?". A neural network brings a structural gain there, not just a constant. The parameters are in the `surrogate` section of `config.yaml`.

### 8.1 Preliminary diagnosis: no bottleneck on the catalog

The initial plan was a U-Net that predicts the visibility map of an observer, to speed up the viewshed computations. Before building it, measure the bottleneck: τ depends neither on the sensor nor on the orientation (week 2), so the catalog needs a single computation of τ per (position, height) pair, and the orientation and the sensor type are analytical. A per-observer surrogate is only worth it if it is faster than the exact engine at equal resolution, and it is not exact.

The exact cost only becomes prohibitive when we want the visibility from **every** position of an area. This is the *total viewshed* problem, whose cost is $O(N R^2)$ for $N$ positions and a range of $R$ cells (Tabik et al., 2015). A fully convolutional network produces the whole map in **one pass**, in $O(N)$.

### 8.2 Target

Consider an observer $o$ at height $h$ above the opaque surface ($z_o = z_\text{opaque}(o) + h$: a roof or the ground). The disc of radius $R$ around $o$ is cut into bins $(a, s)$: distance rings $a$ and azimuth sectors $s$ of $360°/n_s$, centred on the camera azimuths. We define:

$$
V_{h,a,s}(o) = \frac{1}{\lvert \mathcal{B}_{a,s} \rvert} \sum_{j \in \mathcal{B}_{a,s} \cap \text{Ground}} T_{oj}, \qquad T_{oj} = e^{-\tau_{oj}} \in [0, 1],
$$

where $\lvert \mathcal{B}_{a,s} \rvert$ is the number of cells of the bin, which does not depend on the terrain. $V$ is therefore the **share of the bin area that is visible ground**, weighted by the transmittance, in $[0, 1]$.

| Choice | Value | Justification |
|---|---|---|
| $R$ | 150 m | $R_{90}$ of the radar; tele camera: $r_\text{design} \approx$ 143 m |
| Rings | 0-50, 50-150 m | wide ≈ 38 m, standard ≈ 67 m: separates short and long uses |
| Sectors | 8 × 45° | centred on `camera_azimuths_deg`: $V_{h,\cdot,s}$ directly scores a camera oriented towards $s$ |
| Heights | 2, 4, 6, 8, 10 m | union of the roof and mast heights of the catalog |
| $T$ | $e^{-\tau}$, $\alpha = 1$ | one way (camera). The target is **geometric**: the detection laws stay analytical and are applied afterwards |

That is 80 outputs per position. $\tau$ is computed by the `ray_tau` kernel of `src/viewshed.py`: the labels are exact by construction, and a test checks it.

**Properties used afterwards.**

1. **Translation equivariance.** $V$ only depends on the terrain around $o$. The map $o \mapsto V(o)$ is therefore a translation-equivariant operator on the rasters: exactly what a fully convolutional network represents. Its receptive field must cover at least $2R + 1$ = 301 cells.
2. **Vertical invariance.** Shifting every elevation by a constant does not change $V$. The inputs are therefore relative elevations.
3. **D4 equivariance.** A 90° rotation of the terrain shifts the sectors by $+2$; an east-west mirror sends them from $s$ to $-s \bmod 8$. The 8 elements of the dihedral group act exactly on the grid: they provide a data augmentation without approximation (Cohen and Welling, 2016). Two tests check it.
4. **Monotonicity in $h$: only without vegetation.** Without vegetation, the ray goes up at every point when $z_o$ goes up, so $V$ is non-decreasing in $h$. With vegetation, a ray entirely under the canopy travels $\sqrt{1 + \text{slope}^2}$ m per horizontal metre, and the slope increases with $h$: $V$ can decrease. The test shows it on a canopy column. **The network must therefore not be constrained to be monotone.**

### 8.3 Why not a map per observer

Predicting $T_{oj}$ for every $j$ of the disc amounts to producing one image per observer. We then fall back on the cost of 8.1, with in addition a target dominated by zeros (in a built-up area, most of the disc is occluded, and the rays that cross vegetation keep little transmittance). The aggregation by bins keeps the information useful to the decision (which area, in which direction, at which distance) and makes the output dense and smooth.

### 8.4 Data and spatial split

| Set | Definition |
|---|---|
| Valid positions | full disc inside the grid (more than $R$ from the edge) |
| **Test** | site extent + 50 m; exact computation every other pixel |
| Buffer | 30 m around the test block, excluded |
| Validation | 100 m tiles drawn at random (15 %) outside the buffer |
| Training | the rest |

- **Stratified sampling**: 30,000 observers (training and validation), split into roofs (25 %), open ground (40 %) and ground under vegetation (35 %). A uniform draw would under-represent the roofs, which are nonetheless preferred supports.
- **The network predicts a dense map, but the loss is only evaluated at the labelled pixels** (sparse supervision). It only counts the pixels whose whole receptive field is inside the tile, so as not to learn the edge effects of the padding.
- **Why a spatial split.** Two positions 1 m apart have almost identical labels. A random draw of cells would measure an interpolation, not a generalisation (Roberts et al., 2017). The test block contains the site: it is also where the decision will be validated (§8.5).
- **Residual leakage, acknowledged.** With $R$ = 150 m and the extent of the LiDAR tiles, the discs of the test positions overlap terrain seen in training. The test therefore measures the generalisation to new **positions** on a partly known terrain, not to a new site. The 30 m buffer removes the direct autocorrelation of the labels, not that of the inputs.

### 8.5 Evaluation

**Baselines.** A rank score of 0.9 means nothing without a point of comparison. The network is therefore compared with two baselines:
- **B0**: the mean per stratum and per height;
- **B1**: a gradient boosting on hand-crafted local descriptors (horizon angles, vegetation optical depth and ground share per ring and per sector).

**Accuracy**, on the test block:
- mean absolute error (MAE) on $V$, per height, stratum and ring;
- Spearman rank correlation on the total potential $\bar V_h = \sum_{a,s} \lvert \mathcal{B}_{a,s}\rvert V_{h,a,s} / \sum \lvert\mathcal{B}\rvert$;
- **top-K recall**: among the K best exact positions (after non-maximum suppression at 10 m) on the possible supports of the site, share found again in the predicted top-K.

**Speed-up.** Let $c_e$ and $c_n$ be the per-position costs of the exact computation and of the network, and $C_0$ the fixed cost of the dataset and the training. The network pays off beyond

$$
N^\star = \frac{C_0}{c_e - c_n}
$$

positions, expressed in km² at 1 m. Everything is measured on the same machine, on CPU and on MPS.

**Decision (step 6.3).** Candidates are generated from the predicted map: top-K per possible support, with non-maximum suppression. Their matrix `A` is built with the **exact** engine, then the greedy and NSGA-II are run again. F, $p^\star_\mathcal{F}$ and the front are compared with the catalog of week 1, and with a catalog drawn from the **exact** map of the test block (upper bound of what the network can bring).

### 8.6 Assumptions and limitations

1. **A single area of about 1 km²**, leaf-off (November): the generalisation to another site or to the leafy season is not tested.
2. **Geometric target**: $\alpha = 1$ for every sensor. The radar ($\alpha_r = 0.3$, two way) sees through vegetation differently; the target serves to **rank** the positions, and the exact computation takes over for the decision.
3. **Discrete heights** (5 values), like the catalog.
4. **Positions closer than $R$ to the edge excluded**: the predicted map is only reliable inside the valid extent.
5. **Weighted score (§8.9)**: it assumes visibility and weight independent inside a bin.
6. **Selection of the K best positions one by one**: it ignores the submodularity of the coverage. A per-position map is not enough to choose a set.
7. **Limited statistics**: one NSGA-II seed in §8.9, 5 draws for the random controls.

### 8.7 Tests (`tests/test_surrogate_data.py`)

| Test | Reference solution |
|---|---|
| `potential` kernel against the aggregation of `optical_depth` (3 random terrains, 3 heights) | equality to 10⁻⁶ |
| Flat terrain without vegetation | $V = 1$ in every bin |
| Geometry of the bins | north = sector 0, east = sector 2; sectors balanced within 10 % |
| 90° rotation of the terrain | sectors shifted by $+2$ |
| East-west mirror | $s \to -s \bmod 8$ |
| Without vegetation | $V$ non-decreasing in $h$ |
| Canopy column | $V$ decreases from $h$ = 2 to 5 m (counter-example of monotonicity) |
| Split | site entirely in test; no training position within 30 m of the test block; full discs |

### 8.8 Models: dense horizons and physics U-Net (steps 5.3 and 6.1)

Every model is evaluated on the test block, at $h$ = 6 m, with the metrics of §8.5 and its cost per position (`python -m src.surrogate_metrics`).

| Model | Definition |
|---|---|
| Exact computation (`potential`) | reference |
| B0 | mean per stratum |
| B1 | exact horizons + boosting |
| B2 | log-polar pyramid horizons + ground share per bin + boosting |
| Raw U-Net | the 5 base channels (`src/surrogate.py`) |
| Physics U-Net | horizons as input (`src/surrogate_physics.py`) |

**Dense horizons by log-polar pyramid (`src/horizons.py`).** For each band $b$ (`horizons.bands_m`), blocks of $p_b \times p_b$ cells (`horizons.step_px`) are read every $p_b$ pixels, from sliding max and mean filters computed once for the whole grid. The 0-10 m band is exact; the tiling is exact within each band, with small gaps and overlaps at the band boundaries. **The error has no fixed sign**: the highest cell of a block can be closer or farther than its centre, so the approximation is **not** conservative. `python -m src.horizons` measures its cost and its error against the exact horizon.

**B2 adds the ground share per bin**, which directly bounds $V$: without it, the horizons alone miss a variable, not geometric accuracy.

**Physics U-Net.** The network is conditional on $h$: 166 input channels (5 base, $h$, 80 horizons, 80 means), 16 outputs, one pass per height, 3 levels, receptive field of 85 px. Two traps to avoid with the raw U-Net: GroupNorm normalises over the whole image (a receptive field test reveals it, BatchNorm replaces it), and the receptive field must be at least $2R + 1$ = 301 px (checked by a test).

**Break-even point** (§8.5). A surrogate pays off once its fixed costs (data and training) are below the exact cost of the positions it replaces: at the scale of a small site, accuracy decides.

### 8.9 Validation by the decision (step 6.3, `src/surrogate_catalog.py`, `src/reduced_catalog.py`)

**Protocol.** The catalogs of candidate positions have the same counts of masts and roofs as the catalog of `src.candidates`. They are chosen among the possible positions of the test block, on a 2 m step, following the same rules. Each catalog is then evaluated **exactly and in the same way**: matrix on the intruder domain, greedy over the budget grid, $p^\star_\mathcal{F}$ and $p^\star_\mathcal{G}$, NSGA-II seeded with its own greedy solutions. Two controls bracket the comparison: an **oracle**, which applies the same criterion to the exact values, and a **random draw** with the same minimal spacing.

**1. Geometric potential $\bar V$.** Positions ranked by their total potential. Seeing a lot of ground does not necessarily mean seeing the zones that matter: the oracle tells whether a failure comes from the model or from the criterion.

**2. Weighted score.** Each bin is weighted by the mass of weight it contains, $M_b(o) = \sum_{j \in b} w_j$, computed analytically:
- radar: $R_h = \sum_b M_b V_{h,b}$;
- camera: $C_h = \max_s \sum_a M_{a,s} V_{h,a,s}$;
- score = $\max_h \max(R_h / \max R, \; C_h / \max C)$ ($C_h$ alone with `sensors.cameras_only`).

Visibility and weight are assumed independent inside a bin.

**3. Reduced catalogs** (`src/reduced_catalog.py`): $K$ positions only, where the selection should matter most.

**4. Preregistered protocol**, fixed before running (`surrogate.decision.reduced`):
- spacing $R = \tfrac12 \sqrt{A_\text{type}/K_\text{type}}$;
- outer band added to the score with a mass equal to that of the inside;
- $K$ = 5, 10, 15 and 20;
- success if the mean F of B2 (weighted + band) **exceeds the maximum of 5 random draws** for at least 3 sizes out of 4.

**Expected limitation.** Coverage is submodular: the value of a position depends on those already selected. Taking the $K$ best positions one by one ignores their complementarity, and a **per-position** potential map, summarised as sums per bin, cannot carry the information "which cells does this position see?", which the greedy needs for its marginal gains (§5). On a small site, this information is obtained exactly through the matrix `A`.

### 8.10 Additional tests

**`tests/test_surrogate_model.py`**

| Test | Reference solution |
|---|---|
| Receptive field | ≥ $2R + 1$ px, and strictly smaller than the tile (locality) |
| Translation equivariance | shifting the input by a multiple of 16 px shifts the output to within 10⁻⁵ |
| D4 augmentation | **exactly** equal to the labels computed by the exact kernel on the transformed terrain (8 elements of the group) |
| Tiles | no label in the margin where the receptive field is incomplete |

**`tests/test_horizons.py`**

| Test | Reference solution |
|---|---|
| Blocks of 1 px | the pyramid equals the exact horizons of B1, at every height, as well as the ground share |
| Tiling | no overlap within a band; ≥ 90 % of the disc covered exactly once; 0-10 m band exact |
| D4 equivariance of the horizons | 90° rotation of the terrain = sectors shifted by $+2$ |
| `transform` | identical to `surrogate.d4` on the labels |

## 9. Adversarial loop and sensitivity (week 7)

### 9.1 Timely detection (`src/timely.py`)

A detection only helps if the response arrives before the intruder reaches the target (EASI model, Garcia 2007). We assume a deterministic response time $T_R$: a detection at $v$ is **timely** if the time left for the intruder to reach $V_t$ is at least $T_R$.

**Remaining time.** With $t_e = \ell_e / v + t_\times(e)$ (§7.5), let

$$
d(v) = \min_{\gamma : v \to V_t} \sum_{e \in \gamma} t_e .
$$

$d$ **does not depend on the sensors**: a single multi-source Dijkstra from $V_t$, per profile (the graph and the times are symmetric). $d_\mathcal{F} \le d_\mathcal{G}$, since $\mathcal{G}$ has fewer edges.

**Masked metric.** $\tilde P_v = P_v \cdot \mathbb{1}[d(v) \ge T_R]$ and $\tilde p^\star = p^\star(\tilde P)$; likewise $\tilde\sigma$ and $\tilde\Lambda^\star$.

**Proposition 1 (lower bound).** $\tilde p^\star$ is a lower bound on the guaranteed timely detection.

*Proof.* On a path $\gamma$, the time really left at $u$ is $r_\gamma(u) \ge d(u)$. Hence $\{u : d(u) \ge T_R\} \subseteq \{u : r_\gamma(u) \ge T_R\}$, and for every $\gamma$, $\max_{u \in \gamma} \tilde P_u \le \max_{u \in \gamma,\, r_\gamma(u) \ge T_R} P_u$. We conclude by taking the minimum over $\gamma$, then by the argument of §7.4. ∎

**Upper bound.** The real timely detection of a feasible path bounds the minimum from above. We evaluate it, with $r_\gamma$ computed along the path, on two corridors of $G_{\tilde p^\star}$: the shortest in metres and the fastest.

**Proposition 2 (minimal intrusion time).** Let $T_\text{min} = \min_{s \in V_s} d(s)$. If $T_R > T_\text{min}$, then $\tilde p^\star = 0$ for **every** layout.

*Proof.* Along the fastest path from the start that achieves $T_\text{min}$, $d$ decreases; all its nodes satisfy $d(u) \le T_\text{min} < T_R$, hence $\tilde P_u = 0$. ∎

Beyond $T_\text{min}$, no sensor placement can protect: the only levers are **delay** ($t_\text{c}$), **earlier detection** ($b$) or **response** ($T_R$).

**Ceiling of the catalog.** $P$ grows with the set of active sensors: all configurations active bound every feasible layout from above. This ceiling separates what is physically impossible with the catalog from what the solutions simply did not try to do.

| Test (`tests/test_timely.py`) | Reference |
|---|---|
| Straight line, with fence or gate | $d = L/v$, $+ t_\text{c}$ or $+ t_\text{p}$ |
| Fence without gate, profile $\mathcal{G}$ | $d = \infty$ outside |
| Fastest path | $r_\gamma = d$ exactly |
| Detection 1 s from the target | counted if $T_R < 1$ s, ignored otherwise |
| $T_R = 0$ | $\tilde p^\star = p^\star$ |
| $T_R > T_\text{min}$ | $\tilde p^\star = 0$ (proposition 2) |
| 20 random instances | $\tilde p^\star$ decreasing in $T_R$; $\tilde p^\star \le$ upper bound; $d_\mathcal{F} \le d_\mathcal{G}$ |
| Ceiling | equal to `detection` with every configuration |

### 9.2 Exact adversarial loop (`src/adversarial_loop.py`)

**Problem.** Find the minimal-cost layout whose guaranteed timely detection is at least $\theta$ against profile $\mathcal{F}$:

$$
c(\theta, T_R) = \min \Big\{ \textstyle\sum_j a_j x_j + \sum_s f_s y_s \ : \ \text{(C2)-(C4)},\ \tilde p^\star_\mathcal{F}(x) \ge \theta \Big\}.
$$

**Linearisation of the threshold.** With $s_{vj} = -\ln(1 - P_{vj})$ and $s_\theta = -\ln(1 - \theta)$, under the independence assumption of §3.1:

$$
P_v(x) \ge \theta \iff \sum_j s_{vj} x_j \ge s_\theta \iff \sum_j C_{vj} x_j \ge 1, \qquad C_{vj} = \min(s_{vj}/s_\theta,\ 1).
$$

The truncation is exact: if a term equals 1, both sides are true. It removes $s_{vj} = \infty$ ($P_{vj} = 1$) and tightens the continuous relaxation.

**From the corridor to the cuts.** $\tilde p^\star(x) \ge \theta$ if and only if every path $\gamma$ from the starts to the target contains a node $v$ such that $d(v) \ge T_R$ and $P_v(x) \ge \theta$. With $z_v \in \{0,1\}$ and $z_v \le \sum_j C_{vj} x_j$:

$$
\sum_{v \in \gamma,\ d(v) \ge T_R} z_v \ \ge\ 1 \qquad \forall \gamma.
$$

Paths are exponentially many: they are generated on demand.

- **Master**: (C2)-(C4), the $z$-$x$ links and the cuts of the known paths. A variable $z_v$ only exists if $v$ belongs to a cut: elsewhere, fixing $z_v = 0$ changes nothing.
- **Intruder (oracle)**: the union-find of §7.4 on $u_v = \min(\sum_j C_{vj} \hat x_j, 1)$, with $u_v$ set to 0 where $d(v) < T_R$. If there is a corridor where $u < 1$, it becomes a cut. Twenty corridors per iteration, each one avoiding the useful nodes of the previous ones.

**Proposition 3 (validity and termination).** The master is a relaxation: its bound is a lower bound on $c(\theta, T_R)$ at every iteration. Simple paths are finite in number and a path already cut cannot be returned, so the loop terminates. When the intruder finds no more corridor, the solution is feasible, hence optimal. ∎

This is the loop "the optimiser places, the intruder finds the flaw, the optimiser corrects" of the project brief. The naive version, which penalises the last corridor found, can oscillate; this one converges and certifies its result.

**Aggregated cut in $x$.** For every path $\gamma$, $\sum_j \big(\max_{v \in \gamma} C_{vj}\big) x_j \ge 1$ is valid: if a node of $\gamma$ reaches $\theta$, the sum is at least its own sum. In the continuous relaxation, it dominates $\sum_v \sum_j C_{vj} x_j \ge 1$, which the $z$ imply. It is always added. The exact cut in $z$ is only created if $\hat x$ satisfies the cut in $x$ while the corridor stays open, that is when the intruder exploits partial sensors that do not combine at the same place.

**Two fusion assumptions.**

| Fusion | $P_v$ | $C_{vj}$ | Cut in $x$ | Solution |
|---|---|---|---|---|
| Independence (product) | $1 - \prod_j (1 - P_{vj})$ | $\min(s_{vj}/s_\theta, 1)$ | relaxation | bracket within the time limit |
| Full correlation (max) | $\max_j P_{vj}$ | $\mathbb{1}[P_{vj} \ge \theta]$ | **exact** (no $z$) | set cover, certified optimum |

Since $1 - \prod_j (1 - P_{vj}) \ge \max_j P_{vj}$, a layout valid under "max" is also valid under "product":

$$
\underline{c}_\text{product} \ \le\ c_\text{product}(\theta, T_R) \ \le\ c_\text{max}(\theta, T_R).
$$

The "max" fusion is also the pessimistic bound of §3.3 (correlated detections: fog, crawling intruder). Sizing with it is prudent.

**Speed-ups, all exact.**
1. **Shared pool of corridors.** Every start → target path gives a valid cut for every $(\theta, T_R)$, after intersection with the useful nodes of the case. The cases are solved by increasing $(\theta, T_R)$, and each one inherits the corridors of the previous ones.
2. **Useless configurations** (minimal cost only). A configuration without any coefficient on a useful node serves no cut and costs something: it is removed. Under "max" fusion, a configuration whose covered set is included in that of another one, on the same support, of the same type and with a lower or equal sensor cost, is also removed (it can always be swapped for the dominating one). Under "product" fusion, the contributions add up and the argument does not hold.
3. **Cutoff by the incumbent.** Costs are multiples of $q = 0.1$. The master is forced to a cost $\le c^{\text{inc}} - q/2$, with the objective in multiples of $q$. If it becomes infeasible, the incumbent is optimal. The master can then stop at a 5 % gap: exactness comes from the proof of infeasibility, not from its optimality.
4. **Seeding.** The solutions of the NSGA-II front provide cuts (three corridors each, cuts in $x$ only) and the initial incumbent.

| Test (`tests/test_adversarial_loop.py`) | Reference |
|---|---|
| $\min(\sum_{j \in T} C_{vj}, 1) = 1 \iff P_v \ge \theta$ | 40 draws, two fusions, three $\theta$ |
| Minimal cost on 36 toy instances (7 configurations, fence, gate, (C3), (C4)) | **exhaustive enumeration** of the $2^7$ layouts, two fusions, $T_R \in \{0, 6\}$ s |
| Pool inherited from another case | same optimum as from scratch |
| Cost step | detected (0.01 on the toys, 0.1 on the site) |

### 9.3 Sensitivity analysis by decision regret (`src/sensitivity.py`)

**Preregistered protocol**: it is fixed here, and in the `sensitivity` block of `config.yaml`, before any computation.

**Question.** Do the decisions of §9.2 remain good if an assumption is wrong? Varying a metric is not enough: we measure what being wrong costs.

**Decisions evaluated.** $D_0$ and $D_{10}$, the certified optima (max fusion) for $\theta = 0.9$ and $T_R = 0$ or 10 s. A configuration is identified by its key (location, height, sensor, azimuth, tilt), which allows finding it again when a scenario rebuilds `A_intruder`.

**Metrics**, for a scenario $s$ and a decision $D$ computed under the nominal assumption:

| Metric | Definition |
|---|---|
| holds | $\tilde p^\star_s(D) \ge \theta$ |
| $c^\star_s$ | certified minimal cost under $s$ (loop of §9.2) |
| regret | $\text{cost}(D) - c^\star_s \ge 0$ if $D$ holds (over-sizing) |
| failure | if $D$ does not hold: extra cost needed $c^\star_s - \text{cost}(D)$, and $\tilde p^\star_s(D)$ reached |
| stability | Jaccard index between $D$ and $D^\star_s$ |

**Stable decision**: it holds in every scenario, with a regret below 10 % of its cost.

**Scenarios** (one factor at a time around the nominal):

| Factor | Values | What is recomputed |
|---|---|---|
| Delay $(t_\text{c}, t_\text{p})$ | (10, 3)\*, (30, 3), (30, 30), (60, 60), (120, 120) s | $d$ only |
| Intruder speed | 1.0; 1.5\*; 3.0 m/s | $d$ only |
| Band width $b$ | 15; 30\*; 50 m | domain, graph, `A_intruder` |
| Foliage $f$ | 1\*; 2; 4 | `A_intruder` (α × f for every sensor) |

\* nominal value.

- **Foliage.** $\tau = \sum k_i \ell_i$ is linear in $k$: multiplying $k$ by $f$ amounts to multiplying α by $f$, without touching the visibility engine. Applying $f$ to conifers too is pessimistic.
- **Delay-response frontier.** For each $(t_\text{c}, t_\text{p})$, we compute $T_\text{min}$ and $c^\star(\theta = 0.9, T_R)$ for $T_R \in \{10, 30, 60, 120\}$ s, when $T_R < T_\text{min}$ (otherwise, proposition 2 gives the impossibility).
- **Without computation.** τ_obs and π do not enter $\tilde p^\star_\mathcal{F}$ (§7.4, §7.3). The decisions of §9.2 are insensitive to them **by construction**.

**Predictions, written before the computation.** To be written here for the site studied, before running `python -m src.sensitivity` (preregistration), then compared with the results.

**Monotonicity in the delay.** A larger delay increases $d(v)$ wherever the fastest path crosses the fence, hence enlarges the set of useful nodes: $c^\star$ can only decrease, and the range of feasible $T_R$ grows.

## References

- G. L. Nemhauser, L. A. Wolsey, M. L. Fisher (1978). *An analysis of approximations for maximizing submodular set functions, I.* Mathematical Programming 14, 265-294.
- S. Khuller, A. Moss, J. Naor (1999). *The budgeted maximum coverage problem.* Information Processing Letters 70(1), 39-45.
- M. Sviridenko (2004). *A note on maximizing a submodular set function subject to a knapsack constraint.* Operations Research Letters 32(1), 41-43.
- G. L. Nemhauser, L. A. Wolsey (1981). *Maximizing submodular set functions: formulations and analysis of algorithms.* Annals of Discrete Mathematics 11, 279-301.
- R. Church, C. ReVelle (1974). *The maximal covering location problem.* Papers of the Regional Science Association 32, 101-118.
- M. L. Garcia (2007). *Design and Evaluation of Physical Protection Systems*, 2nd ed. Butterworth-Heinemann.
- S. Meguerdichian, F. Koushanfar, M. Potkonjak, M. B. Srivastava (2001a). *Coverage problems in wireless ad-hoc sensor networks.* IEEE INFOCOM 2001, 1380-1387.
- S. Meguerdichian, F. Koushanfar, G. Qu, M. Potkonjak (2001b). *Exposure in wireless ad-hoc sensor networks.* ACM MobiCom 2001, 139-150.
- E. Israeli, R. K. Wood (2002). *Shortest-path network interdiction.* Networks 40(2), 97-111.
- S. Tabik, A. R. Cervilla, E. Zapata, L. F. Romero (2015). *Efficient data structure and highly scalable algorithm for total-viewshed computation.* IEEE JSTARS 8(1), 304-310.
- O. Ronneberger, P. Fischer, T. Brox (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015, 234-241.
- T. Cohen, M. Welling (2016). *Group equivariant convolutional networks.* ICML 2016, 2990-2999.
- D. R. Roberts et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40(8), 913-929.

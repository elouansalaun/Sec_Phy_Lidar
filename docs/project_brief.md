# Original project brief

This is the brief the project started from. The README describes what was
actually done, including where the plan was changed by measurements (for example, the surrogate model was moved away from the per-position viewshed, which turned out not to be a bottleneck).

## The problem

Protect a site (an area to watch, with more or less critical zones) with a limited number of sensors. Each
sensor has:

- a candidate position (roof, mast, ground);
- a height;
- a detection model: a camera has a range, a field of view and an orientation; a radar has a range, a
  360° coverage and an elevation mask.

The objective is to maximise the weighted coverage of the area under a budget constraint. Formally, it is
a Maximal Covering Location Problem (MCLP), NP-hard and well documented in operations research.

## What makes the project interesting

**1. Use the 3D, not only a DSM.** With a plain digital surface model, a tree blocks the view like a wall.
With the classified point cloud, buildings can be treated as opaque and vegetation as semi-transparent,
with an attenuation that depends on the density of points crossed. The result is a detection probability
rather than a plain visible / invisible. This is a real contribution of the LiDAR.

**2. Evaluate against an intruder, not only as a coverage percentage.** Covering 95 % of the area is
useless if one corridor is left unwatched. To test each configuration, compute the least detected
intrusion path: a Dijkstra on a cost map equal to the detection probability, from the outside to the
critical zone. The key metric becomes the detection probability on the worst path. Then iterate: the
optimiser places the sensors, the intruder finds the flaw, the optimiser corrects. This is exactly the
reasoning of a security engineer.

**3. Put deep learning where it really adds something.** The expected bottleneck is the viewshed
computation: one candidate position × several heights × several orientations gives thousands of
computations. Train a surrogate model (for example a U-Net) that takes a local patch of the DSM, the
vegetation density and the observer height, and predicts the visibility map. Then measure the trade-off
between speed-up and accuracy (IoU against the exact viewshed). This gives a real ML argument, justified by
an engineering constraint.

## Optimisation methods to compare

- **Greedy**: coverage is submodular, so the greedy algorithm has a (1 − 1/e) approximation guarantee. An
  excellent baseline.
- **Integer linear programming**: gives the exact optimum on small instances, useful to measure the gap of
  the heuristics.
- **NSGA-II** (pymoo): multi-objective optimisation (cost, coverage, k-coverage redundancy, detection on the
  worst path). The Pareto front speaks directly to a decision maker.

## Data and tools

- **Data**: classified LiDAR point clouds, DTM and DSM, building footprints, roads and paths (likely
  approach axes), candidate positions.
- **Stack**: PDAL and laspy, rasterio, numpy and numba for a home-made viewshed engine (with
  `gdal_viewshed` to validate), graph tools for the paths, pymoo, PyTorch.
- **Demo**: a Streamlit application with a 2D map, where the budget and the sensor types can be set, and
  the coverage and the intrusion path update live.

## Choice of site

We choose a non-sensitive arbitrary site.

## Plan over about 8 phases

| Phase | Content |
|---|---|
| 1 | Site, tiles, DTM / DSM / vegetation density rasters, candidate positions |
| 2 | Exact viewshed engine + sensor models, validation against GDAL |
| 3 | Greedy and MILP baselines, coverage metrics |
| 4 | Evaluation by the intruder (worst path) + multi-objective NSGA-II |
| 5-6 | Deep learning surrogate: dataset, training, speed-up / accuracy study |
| 7 | Adversarial optimiser ↔ intruder loop, sensitivity analysis (season, mast height) |
| 8 | Streamlit demo |

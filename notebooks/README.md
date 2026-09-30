# Notebooks

One notebook per phase of the project, with the equations, short code examples that call the project modules,
and figures. They are stored without outputs.

| Notebook | Phase |
|---|---|
| [01_site_and_lidar_rasters](01_site_and_lidar_rasters.ipynb) | the site drawn in QGIS, point cloud, DTM / DSM / buildings / vegetation, gap fraction, candidate positions | 
| [02_visibility_engine](02_visibility_engine.ipynb) | exact ray traversal (Amanatides & Woo), optical depth τ, Beer-Lambert, analytical checks | 
| [03_sensor_models](03_sensor_models.ipynb) | pinhole camera + TTPF, radar equation + Swerling I, detection maps, coverage matrix | 
| [04_coverage_optimisation](04_coverage_optimisation.ipynb) | MCLP, submodularity, ratio greedy, certified optimum by submodular cuts | 
| [05_intruder](05_intruder.ipynb) | intruder graph, corridor threshold p\* (union-find), exposure Λ, NSGA-II front | 
| [06_timely_detection_and_adversarial_loop](06_timely_detection_and_adversarial_loop.ipynb) | remaining time, T_min, linearisation, constraint generation, certified minimal cost | 
| [07_surrogate_model](07_surrogate_model.ipynb) | potential V, D4 symmetries, log-polar pyramid, why selection is a set problem | 

## Running them

The examples run on the site of `data/site`. Its outputs are not versioned; to run the notebooks, first run
the computation chain of the main README (the LiDAR tiles must be in `data/raw/`), then:

```bash
conda activate lidar-project
python -m src.chain
jupyter lab notebooks/
```

"""
src/chain.py: the whole computation chain, in order, on the site of data/site.

Each step is `python -m <module> [args]`, run unchanged; the chain stops at the first failure.
The main chain goes from the site to the sensitivity analysis; the DTM / viewshed validation and
the surrogate model (PyTorch) are optional groups.

Usage : python -m src.chain                          main chain
        python -m src.chain --from milp              from a step (included)
        python -m src.chain --until coverage         up to a step (included)
        python -m src.chain --with validation surrogate
        python -m src.chain --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.site import ROOT

# (name, group, module and arguments), in the order of their dependencies
STEPS = (
    ("site", "main", ["src.site"]),
    ("rasters", "main", ["src.rasters"]),
    ("validation", "validation", ["src.validation", "compare"]),
    ("candidates", "main", ["src.candidates"]),
    ("validation_viewshed", "validation", ["src.validation_viewshed"]),
    ("coverage", "main", ["src.coverage"]),
    ("greedy", "main", ["src.greedy"]),
    ("milp", "main", ["src.milp"]),
    ("pareto", "main", ["src.pareto"]),
    ("analysis", "main", ["src.analysis"]),
    ("intruder_domain", "main", ["src.intruder_domain"]),
    ("intruder", "main", ["src.intruder"]),
    ("intruder_evaluation", "main", ["src.intruder_evaluation"]),
    ("nsga", "main", ["src.nsga"]),
    ("nsga_analysis", "main", ["src.nsga_analysis"]),
    ("timely", "main", ["src.timely"]),
    ("adversarial_loop", "main", ["src.adversarial_loop"]),
    ("sensitivity", "main", ["src.sensitivity"]),
    ("surrogate_dataset", "surrogate", ["src.surrogate_dataset"]),
    ("surrogate_baselines", "surrogate", ["src.surrogate_baselines"]),
    ("surrogate_train", "surrogate", ["src.surrogate", "train"]),
    ("surrogate_evaluate", "surrogate", ["src.surrogate", "evaluate"]),
    ("physics_train", "surrogate", ["src.surrogate_physics", "train"]),
    ("physics_evaluate", "surrogate", ["src.surrogate_physics", "evaluate"]),
    ("surrogate_catalog", "surrogate", ["src.surrogate_catalog"]),
    ("reduced_catalog", "surrogate", ["src.reduced_catalog"]),
    ("surrogate_metrics", "surrogate", ["src.surrogate_metrics"]),
)
NAMES = [s[0] for s in STEPS]


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.chain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=NAMES)
    ap.add_argument("--until", choices=NAMES)
    ap.add_argument("--with", dest="groups", nargs="+", default=[],
                    choices=["validation", "surrogate"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    i0 = NAMES.index(a.start) if a.start else 0
    i1 = NAMES.index(a.until) if a.until else len(STEPS) - 1
    groups = {"main", *a.groups}
    steps = [s for s in STEPS[i0: i1 + 1] if s[1] in groups or s[0] in (a.start, a.until)]
    if a.list:
        for name, group, cmd in STEPS:
            print(f"{'*' if (name, group, cmd) in steps else ' '} {name:<20} {group:<11} "
                  f"python -m {' '.join(cmd)}")
        return

    t_all = time.time()
    for k, (name, _, cmd) in enumerate(steps, 1):
        print(f"\n=== [{k}/{len(steps)}] {name}: python -m {' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"=== {name} FAILED after {time.time() - t0:.0f} s; "
                     f"resume with: python -m src.chain --from {name}")
        print(f"=== {name}: {time.time() - t0:.0f} s", flush=True)
    print(f"\nChain done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Full reproducibility pipeline (spec §40).

    python run_all.py --config configs/default.yaml [--quick]

Runs, in order:
    stationary  -> drift  -> ablation  -> sensitivity  -> application
and regenerates all tables and figures from the raw trajectories.
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--benchmarks", default="configs/benchmarks.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--output", default="results")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--skip", default="",
        help="comma-separated studies to skip, e.g. 'ablation,sensitivity'",
    )
    args = ap.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    from experiments import (
        run_ablation,
        run_application,
        run_drift,
        run_sensitivity,
        run_stationary,
    )

    argv = [
        "--config", args.config,
        "--benchmarks", args.benchmarks,
        "--experiments", args.experiments,
        "--output", args.output,
    ]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.quick:
        argv.append("--quick")

    t0 = time.time()
    for name, mod in [
        ("stationary", run_stationary),
        ("drift", run_drift),
        ("ablation", run_ablation),
        ("sensitivity", run_sensitivity),
        ("application", run_application),
    ]:
        if name in skip:
            print(f"=== {name}: skipped ===")
            continue
        print(f"=== {name} ===", flush=True)
        rc = mod.main(argv)
        if rc != 0:
            print(f"=== {name}: FAILED (rc={rc}) ===")
            return rc

    print("=== analysis (tables + figures) ===", flush=True)
    from analysis.make_outputs import main as make_outputs

    aargv = ["--config", args.config, "--benchmarks", args.benchmarks,
             "--experiments", args.experiments, "--output", args.output]
    rc = make_outputs(aargv)
    if rc != 0:
        return rc

    print(f"ALL DONE in {time.time() - t0:.0f}s. Results in: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

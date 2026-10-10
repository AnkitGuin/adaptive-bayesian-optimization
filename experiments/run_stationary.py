"""Stationary study (RQ1, RQ2): how well do the methods do when the objective
does not move?  1D benchmarks at three noise levels (all 7 methods); the 2D
benchmark at medium noise (the 2D grid is the expensive case).

    python -m experiments.run_stationary [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import (
    parse_common_args,
    load_all_configs,
    seed_ids_for,
    run_matrix,
)


def main(argv=None) -> int:
    args = parse_common_args("stationary BO study").parse_args(argv)
    cfg, bench_cfgs, exp_cfg = load_all_configs(args)
    spec = exp_cfg["stationary"]

    combos_1d = [(b, "stationary") for b in spec["benchmarks"] if b != "peak2d"]
    combos_2d = [(b, "stationary") for b in spec["benchmarks"] if b == "peak2d"]
    methods = list(spec["methods"])
    seeds = [args.seed] if args.seed is not None else seed_ids_for(cfg, args.quick)

    out = Path(args.output)
    run_matrix(
        study="stationary", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
        combos=combos_1d, noises=list(spec["noises"]), methods=methods,
        seeds=seeds, quick=args.quick,
    )
    run_matrix(
        study="stationary", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
        combos=combos_2d, noises=["medium"], methods=methods,
        seeds=seeds, quick=args.quick,
    )
    print("[stationary] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

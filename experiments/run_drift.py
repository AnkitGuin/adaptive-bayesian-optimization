"""Non-stationary study (RQ3-RQ7): slow drift, abrupt change, random-walk
optimum, observation-noise shift, multimodal amplitude hand-over.

    python -m experiments.run_drift [--quick]
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
    args = parse_common_args("drift / non-stationarity study").parse_args(argv)
    cfg, bench_cfgs, exp_cfg = load_all_configs(args)
    spec = exp_cfg["drift"]
    seeds = [args.seed] if args.seed is not None else seed_ids_for(cfg, args.quick)
    out = Path(args.output)

    for study_name, study_cfg in spec["studies"].items():
        combos = [(c["benchmark"], c["scenario"]) for c in study_cfg["combos"]]
        noises = list(study_cfg["noises"])
        methods = list(study_cfg["methods"])
        run_matrix(
            study=f"drift_{study_name}", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=combos, noises=noises, methods=methods, seeds=seeds,
            quick=args.quick, label="",
        )
    print("[drift] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

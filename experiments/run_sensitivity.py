"""Sensitivity analysis (spec §30): exploration beta, evaluation budget,
initial design size, noise level, drift magnitude, GP kernel.

Hyperparameters were *not* tuned on these trajectories; the frozen DA-BO
configuration comes from the development run documented in the report.

    python -m experiments.run_sensitivity [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.optimization import dabo_ablation, method_config

from .common import (
    parse_common_args,
    load_all_configs,
    seed_ids_for,
    run_matrix,
)


def main(argv=None) -> int:
    args = parse_common_args("sensitivity analysis").parse_args(argv)
    cfg, bench_cfgs, exp_cfg = load_all_configs(args)
    spec = exp_cfg["sensitivity"]
    quick = args.quick
    if args.seed is not None:
        seeds = [args.seed]
    elif quick:
        seeds = seed_ids_for(cfg, True)
    else:
        seeds = seed_ids_for(cfg, False)[:5]
    out = Path(args.output)

    # ---- beta sweep (slow drift, 1D peak) ---------------------------------
    for beta in spec["beta_sweep"]:
        def builder_beta(method, d, b=beta):
            d = type(d)(**{**d.__dict__, "beta_base": float(b)})
            return method_config(method, d)
        run_matrix(
            study="sensitivity", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=[("peak", "slow")], noises=["medium"],
            methods=list(spec["methods_beta"]), seeds=seeds, quick=quick,
            method_builder=builder_beta, label=f"beta_{beta}",
        )

    # ---- budget sweep (abrupt, 1D + 2D) ------------------------------------
    # quick mode scales the budgets down so the smoke test stays fast.
    budget_list = spec["budget_sweep"]
    if quick:
        budget_list = [int(b * 0.4) for b in budget_list]  # 20/40/60
    for budget in budget_list:
        def builder_budget(method, d, b=budget):
            return method_config(method, d)
        run_matrix(
            study="sensitivity", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=[("peak", "abrupt"), ("peak2d", "abrupt")], noises=["medium"],
            methods=list(spec["methods_budget"]), seeds=seeds, quick=quick,
            method_builder=builder_budget,
            label=f"budget_{budget}", budget_override=int(budget),
        )

    # ---- initial design size (slow drift, 1D peak) -------------------------
    # n_initial must drive the initial-design size, so it is passed as an
    # extra_default (applied before x0 is generated), not via method_builder.
    for n_init in spec["n_initial_sweep"]:
        run_matrix(
            study="sensitivity", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=[("peak", "slow")], noises=["medium"],
            methods=["static_ucb", "dabo"], seeds=seeds, quick=quick,
            label=f"ninitial_{n_init}", extra_defaults={"n_initial": int(n_init)},
        )

    # ---- noise sweep (abrupt, 1D peak) --------------------------------------
    for noise in spec["noise_sweep"]:
        run_matrix(
            study="sensitivity", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=[("peak", "abrupt")], noises=[noise],
            methods=list(spec["methods_noise"]), seeds=seeds, quick=quick,
            label=f"noise_{noise}",
        )

    # ---- drift magnitude sweep (random walk, 1D peak) -----------------------
    for i, drift in enumerate(spec["drift_sweep"]):
        sigma = drift["sigma_drift"]
        # build a dedicated scenario entry
        bench_cfg = dict(bench_cfgs["peak"])
        bench_cfg = dict(bench_cfg)
        bench_cfg["scenarios"] = dict(bench_cfgs["peak"]["scenarios"])
        bench_cfg["scenarios"][f"rw_{i}"] = drift
        def builder_rw(method, d):
            return method_config(method, d)
        run_matrix(
            study="sensitivity", outdir=out, cfg=cfg,
            bench_cfgs={"peak": bench_cfg},
            combos=[("peak", f"rw_{i}")], noises=["medium"],
            methods=["static_ucb", "dabo"], seeds=seeds, quick=quick,
            label=f"drift_{sigma}",
        )

    # ---- kernel sweep (slow drift, 1D peak) ---------------------------------
    for kernel in spec["kernel_sweep"]:
        def builder_kernel(method, d, k=kernel):
            d = type(d)(**{**d.__dict__, "kernel": k})
            return method_config(method, d)
        run_matrix(
            study="sensitivity", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=[("peak", "slow")], noises=["medium"],
            methods=list(spec["methods_kernel"]), seeds=seeds, quick=quick,
            method_builder=builder_kernel, label=f"kernel_{kernel}",
        )

    print("[sensitivity] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

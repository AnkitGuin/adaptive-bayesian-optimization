# Adaptive Bayesian Optimization under Non-Stationary and Noisy Objectives

*A study of exploration–exploitation strategies for sequential optimization
under changing conditions.*

A compact, reproducible research project in probabilistic machine learning +
optimization + sequential decision-making. It investigates how Bayesian
Optimization (BO) behaves when the objective is **expensive to evaluate,
noisy, and changing over time**, and develops **DA-BO**, a drift-aware
adaptive BO strategy that tracks moving optima.

> **Research question.** *How should Bayesian optimization adapt its
> exploration–exploitation strategy when the objective is noisy and changes
> over time?* — answered empirically, with controlled synthetic benchmarks
> and a real-world-inspired engineering application, under multi-seed
> statistical analysis.

---

## What is implemented

| Component | Where |
|---|---|
| Gaussian-process regression (Matérn 5/2 default; Matérn 3/2, RBF), pure numpy, cross-validated against scikit-learn | `src/gp/` |
| **Weighted GP**: time-decay weights `w_i = exp(-ln 2 · age_i / h)` converted to heteroscedastic observation noise `σ_i² = σ_base² / w_i` (exact GP regression with diagonal heteroscedastic noise — no approximation) | `src/gp/adaptive_gp.py` |
| Acquisition functions: **EI, UCB, PI**, plus pure exploitation (`μ`) | `src/acquisition/` |
| Deterministic acquisition maximisation: dense grid + local refinement (1D) / multi-start L-BFGS-B (d≥2) | `src/optimization/acquisition_optimizer.py` |
| Single sequential BO loop shared by **all** methods (fair comparison by construction) | `src/optimization/bayesian_optimizer.py` |
| **Page–Hinkley** drift detector on standardized GP residuals (global-mean statistic with post-detection restart) | `src/drift/page_hinkley.py` |
| **DA-BO** = time-decayed GP + drift detection + recovery window (exploration boost `β_high`, temporarily increased forgetting, shortened lengthscale) | `src/optimization/adaptive_optimizer.py` |
| Benchmarks: 1D moving peak, 2D moving peak, 1D three-peak multimodal, 1D drifting sinusoid; scenarios: stationary / slow drift / abrupt / random walk / noise shift / amplitude switch; exact precomputed noise-free optima | `src/benchmarks/` |
| 3D process-efficiency simulator (temperature / pressure / speed) under equipment degradation + demand cycle | `src/benchmarks/process.py` |
| Metrics: instantaneous & cumulative regret (noise-free), tracking error, recovery time, adaptation regret, exploration rate; paired bootstrap + Wilcoxon + Holm correction | `src/evaluation/` |
| Experiments: stationary, drift, ablation, sensitivity, application | `experiments/` |
| Tables (9) + figures (12) regenerated from raw trajectories | `analysis/` |

**Methods compared (spec):** Random search · static GP+EI · static GP+UCB ·
static GP+PI · pure exploitation · **ABO** (time-decayed GP + UCB) ·
**DA-BO** (drift-aware ABO).

### DA-BO in one paragraph

At iteration `t` the optimizer holds `(x_i, y_i, t_i)`. Each observation
receives a time-decay weight `w_i = exp(-λ(t - t_i))` with
`λ = ln 2 / h` (half-life `h`); the weight is converted into an *observation
noise* `σ_i² = σ_base² / w_i`, so the GP posterior is exact and stale
observations are down-weighted rather than deleted. After each evaluation a
standardized predictive residual `r_t = |y_t - μ_t(x_t)| /
(sqrt(σ_f,t(x_t)^2 + σ_noise^2) + ε)` is fed to a Page–Hinkley detector;
the denominator includes both latent GP uncertainty and observation noise.
On drift, for a recovery window of `L` iterations the
optimizer (i) boosts exploration to `β_high` in the UCB, (ii) halves the
effective half-life, and (iii) halves the lengthscale (shorter effective
memory). All three responses are individually ablated.

---

## Quickstart

```bash
cd adaptive-bayesian-optimization
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

# 1) Unit tests (69 tests, a few seconds)
pytest

# 2) Smoke test of the whole pipeline (~4 min, reduced budget/seeds)
python run_all.py --quick

# 3) Full pipeline (~40 min on a 2-core laptop; ~80 min on 1 core)
python run_all.py

# 4) Regenerate tables + figures from the raw trajectories (fast)
python -m analysis.make_outputs
```

Individual studies:

```bash
python -m experiments.run_stationary   [--quick] [--seed 20260923]
python -m experiments.run_drift        [--quick]
python -m experiments.run_ablation     [--quick]
python -m experiments.run_sensitivity  [--quick]
python -m experiments.run_application  [--quick]
```

All scripts accept `--config configs/default.yaml --benchmarks
configs/benchmarks.yaml --experiments configs/experiments.yaml --output
results/`.

---

## Repository layout

```
adaptive-bayesian-optimization/
├── README.md  requirements.txt  pyproject.toml  LICENSE  run_all.py
├── configs/               default.yaml · benchmarks.yaml · experiments.yaml
├── src/
│   ├── base.py            Observation / Trajectory / SearchSpace / LHS
│   ├── gp/                gaussian_process.py · adaptive_gp.py · kernels.py
│   │                      · sklearn_ref.py (validation only)
│   ├── acquisition/       ei.py · ucb.py · pi.py · mean.py
│   ├── optimization/      bayesian_optimizer.py (shared loop)
│   │                      · adaptive_optimizer.py (method factory)
│   │                      · acquisition_optimizer.py
│   ├── drift/             page_hinkley.py
│   ├── benchmarks/        moving_peak.py · multimodal.py · sinusoidal.py
│   │                      · scenarios.py · environment.py · process.py
│   ├── evaluation/        regret.py · tracking.py · statistics.py
│   └── utils/             seeds.py · config.py · logging.py
├── experiments/           run_stationary.py · run_drift.py · run_ablation.py
│                          · run_sensitivity.py · run_application.py
├── analysis/              load.py · tables.py · figures.py · make_outputs.py
├── tests/                 69 unit/integration tests
├── results/               raw/ · tables/ · figures/ · summary.md (generated)
└── report/                report.md · figures/ (generated copies)
```

## Reproducibility

* Every experiment stores `config.json`, per-seed per-method
  `trajectory.csv` (x, noisy y, noise-free f, true optimum, residuals,
  acquisition/beta/half-life/lengthscale used, drift flags) and
  `drift_events.csv` under `results/raw/<study>/<slug>/`.
* All randomness is derived deterministically from
  `(study, seed, component)` (SHA-256 → PCG64), never from wall-clock or
  global numpy state.
* **Fair comparison (spec §22):** within one (study, seed), all methods
  share the same objective state path, the same initial design, and — via an
  identically seeded environment per method — the *same noise draw at every
  iteration*. Verified by `tests/test_optimizer.py`.
* **No lookahead:** decisions at time `t` use only data with index `< t`;
  verified by a perturbation test (`test_no_future_information_leaks...`).
* Regret and tracking are computed **only from noise-free values**
  (spec §33); the optimizer never sees `f_true`.
* `python run_all.py --quick` is a faithful reduced-scale replica
  (budget 40, 3 seeds, smaller grids) — same code paths, same slugs.

## Headline design choices (and why)

1. **Custom numpy GP instead of scikit-learn as the production path.**
   The adaptive GP needs per-iteration, per-observation noise and
   per-iteration lengthscale; scikit-learn is used as an independent
   *reference* in the test suite (posterior mean agrees to 1e-7).
2. **Weights → heteroscedastic noise, not a "weighted GP" fudge.** The
   posterior is exact GP regression with observation covariance
   `K + diag(σ_i²)`; documented in `src/gp/adaptive_gp.py` (spec §16/§17).
3. **UCB for the adaptive methods.** Its exploration knob (β) is the natural
   place to inject the post-drift exploration boost (spec §20). EI/PI are
   studied as static baselines.
4. **Global-mean Page–Hinkley with restart.** A *windowed* baseline adapts to
   a sustained residual level shift and can never fire on it; the
   post-reset restart re-baselines after each event (spec §19).
5. **Frozen configuration.** DA-BO hyperparameters
   (`h=32`, `β_base=2`, `β_high=6`, `L=20`, PH `δ=0.5`, `T=25`) are set in
   `configs/default.yaml` *before* the final runs and never tuned on the
   evaluation trajectories (spec §30); sensitivity studies report their
   neighbourhood.

## Empirical findings (from the full 20-seed runs)

The checked-in result artifacts and headline numbers below were generated
before the drift-residual calibration fix described above. Re-run
`python run_all.py` before using them as current quantitative results.

Headline cumulative-regret results (budget 100, medium noise; mean over 20
seeds):

| Regime (1D peak) | static UCB | ABO | **DA-BO** |
|---|---|---|---|
| stationary | **7.27** | 7.59 | 10.80 (+48%) |
| slow drift (v=0.006) | 74.16 | 66.75 | **26.66 (−64%)** |
| abrupt jump @ t=40 | 67.12 (never recovers) | 62.79 (never) | **16.37 (−76%)**, recovery in ~10 it. |
| amplitude switch | 63.68 | 67.46 | **24.78 (−61%)** |
| noise-only shift | **7.35** | 7.52 | 11.47 (+56%) |
| 2D slow drift | **46.59** | 46.14 | 54.73 (+17%) |
| 2D abrupt jump | 49.95 | 45.61 | 53.20 (n.s.) |

* DA-BO's gains are ablation-attributable: removing **drift detection**
  costs +243% regret (1D abrupt), the **exploration boost** +100%, memory
  shortening +185%, forgetting +89%.
* The benefit scales with the ratio of optimum-motion speed to
  (noise + kernel width); below that ratio adaptation is overhead
  (stationary, slow random walk, 2D slow drift, 3D application).
* Full analysis, statistics (paired bootstrap CIs, Wilcoxon, Holm
  correction), sensitivity analysis and failure analysis:
  **`report/report.md`**; machine-readable tables in `results/tables/`,
  auto-generated summary in `results/summary.md`.

## Reproducibility checklist (Definition of Done)

GP ✓ · acquisitions ✓ · static BO ✓ · random search ✓ · adaptive
forgetting ✓ · drift detection ✓ · DA-BO ✓ · synthetic benchmarks ✓ ·
true moving optimum (noise-free) ✓ · regret from noise-free values ✓ ·
identical evaluation trajectories for all methods ✓ (unit-tested) · no
lookahead ✓ (unit-tested) · ablations ✓ · sensitivity ✓ · multi-seed
(20) ✓ · raw trajectories saved ✓ · reproducible figures/tables ✓
(`run_all.py`, `analysis.make_outputs`) · tests pass (69) ✓ · report
complete ✓ · limitations documented ✓ · exact reproduction commands ✓.

# instructions.md
# Adaptive Bayesian Optimization under Noisy and Non-Stationary Objectives
## A research-oriented classical ML / probabilistic optimization project

---

# 0. Role and objective

You are a senior machine-learning research engineer, optimization researcher, and scientific-software developer.

Your task is to **fully implement, test, experiment with, document, and package** the research project described in this file.

Do not build a generic Bayesian Optimization demo.

Build a compact, reproducible research project investigating how Bayesian Optimization behaves when the objective function is:

- expensive to evaluate;
- noisy;
- non-stationary;
- potentially changing over time.

The project should use classical probabilistic ML and optimization techniques.

The entire project must run on a normal laptop with CPU-only execution.

The final repository should resemble a small research project suitable for discussion in a master's application.

The student must be able to explain every major algorithmic component.

Do NOT use:

- LLMs;
- RAG;
- transformers;
- CNNs;
- large deep-learning models;
- cloud-scale computation.

The central tools should be:

- Gaussian Processes;
- Bayesian optimization;
- acquisition functions;
- uncertainty estimation;
- online/sequential optimization;
- classical statistical analysis.

---

# 1. Project title

Use:

> **Adaptive Bayesian Optimization under Non-Stationary and Noisy Objectives**

Optional subtitle:

> **A Study of Exploration–Exploitation Strategies for Sequential Optimization under Changing Conditions**

Do not claim that the project invents a fundamentally new Bayesian Optimization algorithm unless the experiments and analysis genuinely justify such a claim.

A defensible contribution is:

> developing and empirically evaluating an adaptive acquisition strategy for Bayesian Optimization when the objective changes over time.

---

# 2. Core research problem

Bayesian Optimization (BO) is useful when evaluating an objective function is expensive.

Suppose:

    y = f(x) + epsilon

where evaluating f(x) may represent:

- running a physical experiment;
- simulating a complex system;
- testing a manufacturing configuration;
- tuning a costly model;
- measuring a system response.

Instead of evaluating thousands of candidates, Bayesian Optimization builds a probabilistic surrogate.

The standard loop is:

    observations
         ↓
    Gaussian Process
         ↓
    acquisition function
         ↓
    select next x
         ↓
    expensive evaluation
         ↓
    update GP
         ↓
       repeat

Classical Bayesian Optimization generally assumes that the underlying objective is reasonably stationary during the optimization process.

But many real systems change.

For example:

    f_t(x) != f_{t+1}(x)

because of:

- environmental changes;
- machine degradation;
- changing demand;
- changing operating conditions;
- sensor drift;
- temporal effects.

The central research problem is therefore:

> **How should Bayesian Optimization adapt its exploration–exploitation strategy when the objective is noisy and changes over time?**

---

# 3. Main research hypothesis

Primary hypothesis:

> A time-aware/adaptive Bayesian Optimization strategy can achieve lower cumulative optimization regret and track a moving optimum more effectively than a standard stationary Bayesian Optimization strategy under non-stationary objectives, while avoiding excessive exploration.

This is an empirical hypothesis.

Do not assume it will always be true.

The experiments must determine:

- when adaptation helps;
- when it hurts;
- how much noise can be tolerated;
- how rapidly the objective may change before BO fails.

---

# 4. Main research questions

The final project must answer:

### RQ1 — Stationary baseline

How well do standard Bayesian Optimization methods work when the objective is stationary?

### RQ2 — Noise

How does observation noise affect acquisition-function performance?

### RQ3 — Non-stationarity

What happens when the objective function changes during optimization?

### RQ4 — Adaptation

Can time-aware weighting of historical observations improve optimization of a changing objective?

### RQ5 — Exploration vs exploitation

Does an adaptive acquisition strategy choose better trade-offs between:

    explore uncertain regions
    vs
    exploit promising regions

under changing conditions?

### RQ6 — Forgetting

How quickly should old observations become irrelevant?

### RQ7 — Robustness

How does the method behave under:

- abrupt objective changes;
- gradual objective drift;
- high observation noise;
- changing optimum location;
- changing objective amplitude?

### RQ8 — Failure modes

When does adaptive BO perform worse than standard BO?

---

# 5. Important conceptual distinction

Do not confuse:

### Optimization of a fixed function

    maximize f(x)

with:

### Tracking a changing optimum

    maximize f_t(x)

The second problem is fundamentally different.

For a changing objective, define:

    x*_t = argmax_x f_t(x)

The optimizer must balance:

1. finding good regions;
2. identifying when the optimum moved;
3. adapting to the new objective;
4. not wasting too many evaluations relearning stale information.

This tracking perspective is central to the project.

---

# 6. Recommended project scope

Use two levels of experiments.

## Level 1 — Controlled synthetic benchmarks

These are the main research experiments.

They allow precise control over:

- noise;
- drift;
- optimum movement;
- dimensionality;
- function shape.

## Level 2 — One real-world-inspired application

Use a low-dimensional engineering or process optimization dataset/simulation.

The real-world component is for demonstrating applicability.

Do not let dataset imperfections dominate the research.

The synthetic experiments are where the algorithmic conclusions should come from.

---

# 7. Synthetic benchmark functions

Implement several classical optimization functions.

At minimum:

### Function A — 1D sinusoidal objective

Example:

    f_t(x) =
        sin(x + phi_t)

where:

    phi_t

changes with time.

This gives an easily visualized moving optimum.

---

### Function B — Moving Gaussian peak

Example:

    f_t(x) =
        exp(-(x - mu_t)^2 / (2*sigma^2))

where:

    mu_t

moves over time.

This is especially useful because the true optimum is known.

---

### Function C — Non-stationary multimodal function

Construct a function such as:

    f_t(x) =
        a1(t) * exp(-(x-mu1(t))^2/s1^2)
        +
        a2(t) * exp(-(x-mu2(t))^2/s2^2)

This creates multiple competing regions.

The optimizer must decide whether to:

- continue exploiting the current best peak;
- investigate another region.

---

### Function D — 2D moving objective

Use a two-dimensional objective with:

    x = [x1, x2]

and a moving optimum:

    mu_t = [mu1(t), mu2(t)]

This tests whether the method scales beyond trivial 1D visualization.

---

# 8. Ground-truth optimum

For synthetic functions, calculate or approximate the true optimum:

    x*_t = argmax_x f_t(x)

at every time step.

This is extremely important.

It allows calculation of:

    instantaneous regret =
        f_t(x*_t) - f_t(x_t)

and:

    cumulative regret =
        sum_t [f_t(x*_t) - f_t(x_t)]

Do NOT use the noisy observed reward to define the true optimum.

The optimum must come from the underlying noise-free function.

---

# 9. Baseline Bayesian Optimization

Implement a standard Bayesian Optimization pipeline.

Use a Gaussian Process surrogate.

The GP should model:

    y = f(x) + epsilon

with:

    f ~ GP(m(x), k(x,x'))

Use a sensible kernel.

Start with:

> Matérn 5/2

Optionally compare with:

- RBF;
- Matérn 3/2.

Do not perform excessive kernel experimentation.

The purpose is to study adaptive optimization, not kernel benchmarking.

---

# 10. Gaussian Process implementation

Use either:

- scikit-learn GaussianProcessRegressor;
- GPyTorch if there is a strong reason.

Prefer scikit-learn for simplicity unless the implementation requires functionality unavailable there.

The implementation should expose:

- kernel;
- noise level;
- normalization;
- optimizer settings;
- random seed.

The GP must provide:

    posterior mean μ(x)

and:

    posterior standard deviation σ(x)

These are required by the acquisition functions.

---

# 11. Acquisition functions

Implement at least:

## Acquisition A — Expected Improvement (EI)

For maximization:

    EI(x) =
        (μ(x) - f_best - ξ) Φ(Z)
        +
        σ(x) φ(Z)

where:

    Z = (μ(x) - f_best - ξ) / σ(x)

Handle:

    σ(x) ≈ 0

carefully.

Do not allow NaNs.

---

## Acquisition B — Upper Confidence Bound (UCB)

Use:

    UCB(x) = μ(x) + β σ(x)

where β controls exploration.

For maximization.

Study at least two β values.

---

## Acquisition C — Probability of Improvement (PI)

Use:

    PI(x) = Φ(Z)

where:

    Z = (μ(x) - f_best - ξ) / σ(x)

PI is useful as a baseline because it can become overly exploitative.

---

# 12. Optimization of the acquisition function

The acquisition function itself must be optimized.

For low-dimensional synthetic problems, use:

- dense grid evaluation;
- local refinement;
- scipy.optimize.

Do not simply evaluate 10 random points and call the best one the acquisition optimum.

For 1D/2D problems, a dense grid plus local optimization is acceptable.

For higher dimensions, use multistart optimization.

The acquisition optimizer must be deterministic when a seed is supplied.

---

# 13. Initial design

Bayesian Optimization requires initial observations.

Use a configurable initial design.

Options:

- random uniform;
- Latin Hypercube Sampling.

Prefer Latin Hypercube Sampling for the main experiments.

The initial design should be generated before sequential optimization.

Do not use information from future objective values to construct it.

---

# 14. Standard stationary BO baseline

Implement:

> **Static BO**

Procedure:

    1. collect initial observations
    2. fit GP
    3. optimize acquisition
    4. evaluate selected x
    5. append observation
    6. refit GP
    7. repeat

All historical observations remain equally relevant.

This is the key baseline.

---

# 15. Main proposed method

Develop:

> **Adaptive Bayesian Optimization (ABO)**

The core idea is:

> Old observations should become progressively less influential when the objective is changing.

A simple and interpretable first implementation should use **time-decayed observation weights**.

For observation i at current time t:

    age_i = t - t_i

and:

    w_i = exp(-lambda * age_i)

where λ controls the forgetting rate.

Recent observations therefore receive larger effective importance.

---

# 16. Important implementation issue

Standard Gaussian Process regression does not directly accept arbitrary observation weights in the simplest scikit-learn interface.

Therefore, do NOT pretend that passing weights to the GP automatically implements weighted GP regression.

Use one of these defensible strategies:

### Strategy A — Likelihood/noise weighting

Convert weights into observation-specific noise:

    sigma_i^2 ∝ 1 / w_i

Recent points receive lower effective observation noise.

### Strategy B — Replication/approximation

Use weighted pseudo-observations carefully.

This is less elegant and should not be the default.

### Strategy C — Implement a custom weighted GP

If practical, implement the GP posterior equations directly.

This is mathematically clean but more work.

Preferred default:

> Implement a custom weighted GP or use heteroscedastic observation noise derived transparently from the decay weights.

Document the exact mathematical formulation.

Do not hide this detail.

---

# 17. Recommended adaptive GP formulation

A practical formulation is:

    K_w = K + Σ_w

where:

    Σ_w = diag(σ_noise,i²)

and:

    σ_noise,i² = σ_base² / max(w_i, epsilon)

Interpretation:

- recent observations → high weight → lower effective noise;
- old observations → low weight → higher effective noise.

This makes stale observations less influential while retaining them.

The exact formula may be modified if a more principled weighted GP formulation is implemented.

The final report must state the exact equation.

---

# 18. Forgetting-rate parameter

Introduce:

    lambda

as the forgetting rate.

Interpretation:

- λ = 0 → no forgetting;
- small λ → slow forgetting;
- large λ → rapid forgetting.

Study:

    lambda ∈ {0, small, medium, large}

Use values appropriate to the number of optimization iterations rather than arbitrary physical units.

For example, define a half-life:

    h

such that:

    w(age=h) = 0.5

Then:

    lambda = ln(2) / h

This is easier to interpret.

---

# 19. Main proposed extension: Drift-Aware Adaptive BO

After implementing time-decayed BO, add a second layer:

> **Drift-Aware Adaptive Bayesian Optimization (DA-BO)**

The optimizer should detect evidence that the objective has changed.

Possible signals:

- degradation in recent residual likelihood;
- standardized GP residuals;
- rolling prediction error;
- Page-Hinkley detector on residuals.

Use one primary drift detector.

Recommended:

> Page-Hinkley on standardized GP residuals.

At time t:

    residual_t =
        |y_t - μ_t(x_t)| / (σ_t(x_t) + epsilon)

Feed residual_t into the detector.

When drift is detected:

- temporarily increase forgetting;
- shorten effective memory;
- increase exploration;
- optionally reset part of the surrogate history.

Do not change all mechanisms simultaneously without an ablation study.

---

# 20. Exploration adaptation

The optimizer should optionally adapt exploration after drift.

For UCB:

    UCB(x) = μ(x) + β_t σ(x)

Set:

    β_t = β_base

during stable periods.

After drift:

    β_t = β_high

for a configurable recovery period.

This reflects the intuition:

> after the environment changes, uncertainty about the optimum should temporarily matter more.

For EI/PI, equivalent exploration parameters can be modified.

The main implementation may use UCB for the adaptive method if this makes the design cleaner.

---

# 21. Proposed final algorithm

The main algorithm should be:

> **DA-BO: Drift-Aware Adaptive Bayesian Optimization**

At each iteration t:

### Step 1 — Observe current state

The optimizer has historical:

    (x_i, y_i, t_i)

for:

    i < t

### Step 2 — Compute time weights

    w_i = exp(-lambda_t * (t - t_i))

### Step 3 — Fit/update adaptive GP

Recent observations receive greater effective importance.

### Step 4 — Determine exploration level

If no drift:

    beta_t = beta_base

If drift detected:

    beta_t = beta_high

### Step 5 — Optimize acquisition

For example:

    a_t(x) = μ_t(x) + beta_t σ_t(x)

### Step 6 — Select next experiment

    x_t = argmax_x a_t(x)

### Step 7 — Evaluate true objective

    y_t = f_t(x_t) + epsilon_t

### Step 8 — Calculate standardized residual

    r_t =
        |y_t - μ_t(x_t)|
        /
        (σ_t(x_t) + epsilon)

### Step 9 — Update drift detector

### Step 10 — Store observation

### Step 11 — Repeat

The current target value must never be used to choose x_t before x_t is evaluated.

---

# 22. Fair comparison requirement

All algorithms must receive:

- the same initial design;
- the same objective realization;
- the same noise realization where possible;
- the same evaluation budget;
- the same domain;
- the same random seeds.

Otherwise comparisons are not fair.

For example:

    Static BO
    Adaptive BO
    DA-BO

must all optimize the same:

    f_t(x) + epsilon_t

trajectory.

---

# 23. Required baseline algorithms

At minimum implement:

1. Random Search
2. Static GP + EI
3. Static GP + UCB
4. Static GP + PI
5. Time-decayed GP + acquisition
6. DA-BO

Random Search is important because Bayesian Optimization should demonstrate benefit over a simple baseline.

---

# 24. Optional baseline

Add:

> Pure exploitation

where:

    x_t = argmax μ_t(x)

This shows what happens when exploration is removed.

This baseline is especially useful in changing environments because it may become trapped in stale regions.

---

# 25. Non-stationarity scenarios

Create controlled scenarios.

## Scenario A — Stationary

    f_t(x) = f(x)

Purpose:

Establish baseline BO performance.

---

## Scenario B — Slow drift

The optimum moves slowly:

    μ_t = μ_0 + v*t

Purpose:

Test whether adaptation helps under gradual change.

---

## Scenario C — Abrupt change

At:

    t = T_change

switch:

    f_A(x) → f_B(x)

Purpose:

Measure adaptation/recovery.

---

## Scenario D — Random walk optimum

Let:

    μ_{t+1} = μ_t + η_t

where:

    η_t ~ N(0, σ_drift²)

Purpose:

Model continuously changing environments.

---

## Scenario E — Changing noise

The underlying function remains fixed but:

    σ_noise(t)

changes.

Purpose:

Separate objective drift from observation-noise changes.

---

## Scenario F — Changing multimodal landscape

Peak amplitudes or locations change.

Example:

    peak 1 initially dominates
    ↓
    peak 2 becomes dominant
    ↓
    peak 3 becomes dominant

Purpose:

Test whether the optimizer can abandon obsolete regions.

---

# 26. Noise levels

Use multiple noise levels.

For example:

    low
    medium
    high

Define them relative to the objective's scale.

Do not blindly use arbitrary absolute noise values.

Normalize the objective where appropriate.

Document the signal-to-noise ratio.

---

# 27. Evaluation metrics

Do NOT evaluate only the best function value.

Use multiple metrics.

## Metric 1 — Instantaneous regret

    R_t =
        f_t(x*_t) - f_t(x_t)

Lower is better.

---

## Metric 2 — Cumulative regret

    CR_T =
        Σ_t R_t

This is one of the main metrics.

---

## Metric 3 — Best-so-far objective

Track:

    max_{i <= t} f_i(x_i)

For stationary problems.

For non-stationary problems, be careful because old best values may no longer be relevant.

---

## Metric 4 — Tracking error

For synthetic functions with known optimum:

    ||x_t - x*_t||

This directly measures whether the optimizer follows the moving optimum.

---

## Metric 5 — Recovery time

After an abrupt change, measure:

> How many evaluations are required before the optimizer reaches a specified fraction of the new optimum?

For example:

    f_t(x_t) >=
        f_t(x*_t) - tolerance

---

## Metric 6 — Adaptation regret

Calculate regret specifically over the post-change recovery window.

This isolates adaptation performance.

---

## Metric 7 — Exploration rate

Measure how often the optimizer selects points outside the current high-value region.

This is optional but useful for explaining behavior.

---

# 28. Acquisition-function analysis

The report must investigate why acquisition functions behave differently.

Compare:

### EI

Typically balances expected improvement against uncertainty.

### UCB

Explicitly controls exploration through β.

### PI

Can become highly exploitative.

The analysis should explain observed differences rather than merely reporting scores.

---

# 29. Ablation studies

This is essential.

The final adaptive method has several components.

Remove them individually.

## Ablation A

DA-BO

vs

DA-BO without forgetting.

This becomes essentially stationary BO.

---

## Ablation B

DA-BO

vs

DA-BO without drift detection.

---

## Ablation C

DA-BO

vs

DA-BO without exploration boost after drift.

---

## Ablation D

Different forgetting half-lives.

---

## Ablation E

Different drift thresholds.

---

## Ablation F

Different recovery durations.

---

# 30. Sensitivity analysis

Study sensitivity to:

- forgetting half-life;
- GP kernel;
- observation noise;
- drift magnitude;
- drift frequency;
- exploration parameter;
- evaluation budget;
- initial design size.

Do not tune parameters on the final test trajectory.

Use a development configuration for tuning.

Freeze the configuration before final evaluation.

---

# 31. Statistical rigor

Do not report a single optimization trajectory.

Run multiple seeds.

Recommended:

    20 independent seeds

for the main synthetic experiments if computationally feasible.

For every metric report:

- mean;
- standard deviation;
- median where useful;
- confidence interval where appropriate.

Example:

    cumulative regret:
    12.4 ± 2.1

Do not fabricate values.

The code must calculate these from actual runs.

---

# 32. Statistical comparisons

Where appropriate, compare algorithms using paired evaluations because all methods can be run on the same objective/noise trajectories.

Possible tests:

- paired bootstrap;
- Wilcoxon signed-rank test.

Use effect sizes where practical.

Do not run dozens of hypothesis tests without correction or explanation.

The goal is scientific interpretation, not p-value hunting.

---

# 33. Important issue: noisy objective vs true objective

The optimizer observes:

    y_t = f_t(x_t) + epsilon_t

but the researcher can evaluate:

    f_t(x_t)

for synthetic benchmarks.

Use:

### Noisy value

to simulate what the optimizer sees.

### Noise-free value

to calculate regret.

This distinction must be maintained throughout the code.

Otherwise the optimization metrics become contaminated by observation noise.

---

# 34. Real-world-inspired application

After synthetic experiments, implement one practical case.

Recommended:

> **Adaptive optimization of an engineering process under changing operating conditions**

Possible objective:

- maximize simulated process efficiency;
- minimize energy consumption;
- maximize output quality.

A good approach is to construct a low-dimensional simulator or use an existing open dataset.

For example:

    x1 = operating temperature
    x2 = pressure
    x3 = speed

and:

    y = process performance

Then introduce controlled environmental drift.

The goal is not to pretend that the simulator is a real factory.

Clearly label it as:

> a controlled engineering benchmark inspired by process optimization.

---

# 35. Optional application: battery/energy optimization

A second possible application is:

> optimize charging/control parameters under changing battery conditions.

Possible variables:

- charging current;
- temperature;
- charging duration.

Objective:

- efficiency;
- charging time;
- degradation-aware reward.

Keep the simulator simple.

Do not turn this into a giant battery-modeling project.

---

# 36. Visualizations

Generate publication-style figures.

At minimum:

## Figure 1

Example of a stationary objective.

---

## Figure 2

Moving optimum over time.

Show:

    x*_t

---

## Figure 3

Optimization trajectories.

Plot:

    x_t

for:

- static BO;
- adaptive BO;
- DA-BO.

Overlay the true moving optimum.

This is one of the most important figures.

---

## Figure 4

Cumulative regret vs evaluation budget.

Compare all algorithms.

---

## Figure 5

Instantaneous regret vs time.

---

## Figure 6

Objective value vs time.

---

## Figure 7

Recovery after abrupt objective change.

Mark:

    true change point
    drift detection point
    recovery point

---

## Figure 8

Effect of forgetting half-life.

---

## Figure 9

Effect of observation noise.

---

## Figure 10

Acquisition functions over the search domain at selected iterations.

This helps explain exploration vs exploitation.

---

## Figure 11

Ablation study.

---

## Figure 12

Coverage/behavior across different drift rates.

---

# 37. Required tables

Generate CSV and report-ready tables.

### Table 1

Synthetic benchmark definitions.

### Table 2

Stationary optimization results.

### Table 3

Slow-drift results.

### Table 4

Abrupt-shift results.

### Table 5

Random-walk optimum results.

### Table 6

Noise robustness.

### Table 7

Ablation results.

### Table 8

Sensitivity analysis.

### Table 9

Real-world-inspired application.

---

# 38. Repository structure

Use approximately:

    adaptive-bayesian-optimization/
    │
    ├── README.md
    ├── instructions.md
    ├── LICENSE
    ├── requirements.txt
    ├── pyproject.toml
    ├── .gitignore
    │
    ├── configs/
    │   ├── default.yaml
    │   ├── benchmarks.yaml
    │   └── experiments.yaml
    │
    ├── src/
    │   ├── gp/
    │   │   ├── gaussian_process.py
    │   │   ├── kernels.py
    │   │   └── adaptive_gp.py
    │   │
    │   ├── acquisition/
    │   │   ├── ei.py
    │   │   ├── ucb.py
    │   │   └── pi.py
    │   │
    │   ├── optimization/
    │   │   ├── bayesian_optimizer.py
    │   │   ├── adaptive_optimizer.py
    │   │   └── acquisition_optimizer.py
    │   │
    │   ├── drift/
    │   │   └── page_hinkley.py
    │   │
    │   ├── benchmarks/
    │   │   ├── stationary.py
    │   │   ├── moving_peak.py
    │   │   ├── multimodal.py
    │   │   └── nonstationary.py
    │   │
    │   ├── evaluation/
    │   │   ├── regret.py
    │   │   ├── tracking.py
    │   │   └── statistics.py
    │   │
    │   └── utils/
    │       ├── seeds.py
    │       ├── config.py
    │       └── logging.py
    │
    ├── experiments/
    │   ├── run_stationary.py
    │   ├── run_drift.py
    │   ├── run_ablation.py
    │   └── run_sensitivity.py
    │
    ├── tests/
    │   ├── test_gp.py
    │   ├── test_acquisition.py
    │   ├── test_optimizer.py
    │   ├── test_benchmarks.py
    │   ├── test_drift.py
    │   └── test_metrics.py
    │
    ├── results/
    │   ├── raw/
    │   ├── tables/
    │   └── figures/
    │
    └── report/
        ├── report.md
        └── figures/

The exact structure may be improved if there is a scientifically justified reason.

---

# 39. Software architecture

Keep these responsibilities separate.

### GP

Responsible for:

    fit
    predict
    posterior mean
    posterior variance

### Acquisition

Responsible for:

    EI
    UCB
    PI

### Acquisition optimizer

Responsible for:

    argmax_x acquisition(x)

### Objective

Responsible for:

    f_t(x)
    true optimum

### Observation model

Responsible for:

    f_t(x) + noise

### Drift detector

Responsible for:

    residual stream
    drift state

### Bayesian optimizer

Responsible for:

    sequential loop

### Evaluation

Responsible for:

    regret
    tracking
    recovery
    statistics

Do not put all logic into one class.

---

# 40. CLI requirements

Provide commands such as:

    python -m experiments.run_stationary

    python -m experiments.run_drift

    python -m experiments.run_ablation

    python -m experiments.run_sensitivity

Ideally support:

    --config configs/default.yaml
    --seed 42
    --output results/

A full reproducibility command should exist.

For example:

    python run_all.py --config configs/default.yaml

---

# 41. Testing requirements

Write unit tests for mathematical components.

## GP tests

Verify:

- posterior shapes;
- variance is non-negative;
- deterministic output under fixed configuration;
- predictions behave sensibly on toy data.

## Acquisition tests

Verify:

- EI is non-negative;
- UCB increases with uncertainty;
- PI remains in [0,1];
- zero-variance cases do not produce NaN.

## Benchmark tests

Verify:

- known optimum for simple functions;
- moving optimum is correctly calculated;
- deterministic output with seed.

## Drift detector

Construct:

    stable residuals
    →
    shifted residuals

and verify detection.

## Regret tests

Use a hand-created objective where the optimum is known.

## Optimizer tests

Verify:

- no duplicate logic errors;
- budget respected;
- selected points remain inside bounds;
- objective queried exactly once per iteration.

---

# 42. Reproducibility requirements

Every experiment must save:

- configuration;
- seed;
- objective parameters;
- initial design;
- noise realization where relevant;
- selected points;
- noisy observations;
- noise-free objective values;
- true optimum;
- acquisition values where useful;
- drift detector state;
- detected drift events.

Example:

    results/raw/experiment_001/
        config.json
        trajectory.csv
        summary.json
        drift_events.csv

This allows all figures to be regenerated.

---

# 43. Experiment configuration

A central YAML configuration should include:

    search_space
    initial_design_size
    evaluation_budget
    kernel
    acquisition
    noise_level
    forgetting_half_life
    drift_threshold
    exploration_beta
    number_of_seeds
    benchmark
    shift_type

Avoid hard-coded experimental constants.

---

# 44. Research report

Create a proper report.

## Abstract

150–250 words.

## 1. Introduction

Explain:

- expensive black-box optimization;
- Bayesian Optimization;
- exploration vs exploitation;
- non-stationarity.

## 2. Research Questions

State RQs and hypothesis.

## 3. Background

Explain:

- Gaussian Processes;
- posterior mean/variance;
- acquisition functions;
- Bayesian Optimization;
- regret;
- non-stationary optimization.

## 4. Methodology

Describe:

- static BO;
- adaptive GP;
- time decay;
- drift detection;
- exploration adaptation;
- DA-BO.

Include equations.

## 5. Experimental Setup

Describe:

- benchmark functions;
- noise;
- drift scenarios;
- evaluation budget;
- seeds;
- baselines.

## 6. Results

Use tables and figures.

## 7. Ablation

Determine which components actually matter.

## 8. Sensitivity

Analyze hyperparameters.

## 9. Failure Analysis

Discuss when the adaptive approach performs poorly.

## 10. Limitations

Be honest about:

- low-dimensional experiments;
- synthetic benchmarks;
- assumptions;
- computational constraints;
- absence of theoretical guarantees.

## 11. Conclusion

Only state conclusions supported by experiments.

## 12. Future Work

Potential directions:

- contextual Bayesian Optimization;
- multi-fidelity BO;
- constrained BO;
- multi-objective BO;
- theoretical regret bounds;
- contextual/non-stationary Gaussian Processes.

---

# 45. Important scientific caveat

Do not claim:

> "Adaptive BO always performs better."

The correct research framing is:

> "We investigate whether adaptive forgetting and drift-aware exploration improve optimization performance under controlled non-stationary objectives."

Some settings may favor static BO.

That is scientifically useful.

---

# 46. Contextual Bayesian Optimization extension

Only after the main DA-BO implementation works.

Introduce an observed context:

    c_t

such as:

- temperature;
- operating regime;
- time;
- environmental condition.

Then model:

    f(x, c)

rather than only:

    f(x)

This turns the project toward:

> **Contextual Bayesian Optimization**

The optimizer observes context c_t and chooses x_t accordingly.

Example:

    context:
        temperature = 30°C
        demand = high

    decision:
        operating parameter = x

    reward:
        efficiency

This is an excellent future-work direction.

Do not implement it before the core project is stable.

---

# 47. Optional contextual extension architecture

If implemented, use:

    z = [x, c]

as GP input.

Then learn:

    f(x,c)

The objective becomes:

    x*_t =
        argmax_x f(x, c_t)

The optimizer must not choose the context.

The environment provides it.

This is conceptually different from ordinary BO.

Document the distinction carefully.

---

# 48. Optional connection to contextual bandits

Do not mix Bayesian Optimization and contextual bandits carelessly.

The conceptual distinction is:

### Bayesian Optimization

Usually:

- expensive continuous action space;
- relatively small evaluation budget;
- exploit structure of the objective using a surrogate.

### Contextual Bandits

Usually:

- repeated decisions;
- observed context;
- actions;
- rewards;
- explicit exploration/exploitation over time.

If the contextual extension is implemented, explain why the project is still framed primarily as Bayesian Optimization.

---

# 49. Expected final resume entry

After the experiments actually produce results, the resume entry can resemble:

> **Adaptive Bayesian Optimization under Distribution Shift**  
> *Python, Gaussian Processes, Bayesian Optimization, Statistical ML*
>
> - Developed a drift-aware Bayesian Optimization framework using time-decayed Gaussian-process observations and adaptive exploration to track moving optima under noisy, non-stationary objectives.
> - Compared EI, UCB, PI, static BO, and adaptive BO across controlled drift/noise regimes using cumulative regret, tracking error, recovery time, and multi-seed statistical analysis.

Only include numerical results after they are actually measured.

For example, if the final experiments show a particular percentage reduction in regret, that number may be added later.

---

# 50. What NOT to do

Do not turn this into:

- a generic hyperparameter tuner;
- "Bayesian optimization of XGBoost hyperparameters";
- a single toy function;
- a single optimization trajectory;
- a dashboard project;
- a Kaggle project;
- a deep-learning project;
- a huge reinforcement-learning system.

The research contribution is:

> **sequential decision-making under an evolving objective**

not merely:

> "I know how to call a Bayesian Optimization library."

Do not use Optuna as the core implementation and claim that you implemented Bayesian Optimization.

Libraries may be used for validation, but the core GP/acquisition/optimization logic should be implemented or clearly exposed by your own code.

---

# 51. Implementation order

Follow this order.

## Phase 1 — Repository

Create:

- project structure;
- environment;
- configuration;
- logging;
- seed management.

## Phase 2 — Benchmark functions

Implement:

- stationary;
- moving peak;
- multimodal;
- abrupt shift;
- random walk optimum.

Verify true optima.

## Phase 3 — Gaussian Process

Implement:

- GP;
- kernels;
- posterior prediction.

Test thoroughly.

## Phase 4 — Acquisition functions

Implement:

- EI;
- UCB;
- PI.

Test edge cases.

## Phase 5 — Static Bayesian Optimization

Implement the standard sequential loop.

Verify performance on stationary functions.

## Phase 6 — Adaptive GP

Implement:

- time decay;
- effective observation noise;
- configurable half-life.

Compare against static BO.

## Phase 7 — Drift detection

Implement:

- standardized residual stream;
- Page-Hinkley;
- detection logging.

## Phase 8 — DA-BO

Combine:

- forgetting;
- drift detection;
- exploration adaptation.

## Phase 9 — Controlled experiments

Run:

- stationary;
- gradual drift;
- abrupt drift;
- random walk;
- noise shift;
- multimodal shift.

## Phase 10 — Ablations

Remove components one at a time.

## Phase 11 — Sensitivity

Vary:

- half-life;
- noise;
- drift;
- β;
- evaluation budget.

## Phase 12 — Real-world-inspired benchmark

Only after the synthetic experiments are stable.

## Phase 13 — Analysis

Generate:

- figures;
- tables;
- statistical summaries.

## Phase 14 — Report

Write the report based on actual results.

## Phase 15 — Reproducibility

Delete generated results.

Run the entire pipeline from scratch.

Verify that results can be regenerated.

---

# 52. Definition of done

The project is complete only when:

- [ ] GP implementation works;
- [ ] acquisition functions work;
- [ ] static BO works;
- [ ] random search baseline works;
- [ ] adaptive forgetting works;
- [ ] drift detection works;
- [ ] DA-BO works;
- [ ] synthetic benchmarks work;
- [ ] true moving optimum is calculated;
- [ ] regret is calculated from noise-free objective values;
- [ ] all methods use identical evaluation trajectories;
- [ ] no future information leaks into decisions;
- [ ] ablations are complete;
- [ ] sensitivity analysis is complete;
- [ ] multiple seeds are supported;
- [ ] raw trajectories are saved;
- [ ] figures are reproducible;
- [ ] tables are reproducible;
- [ ] tests pass;
- [ ] report is complete;
- [ ] limitations are documented;
- [ ] README contains exact reproduction commands.

---

# 53. Final instruction to the coding LLM

You are responsible for implementing the entire project described in this file.

Treat this document as the complete project specification.

Do not ask the student to manually implement individual modules unless a genuine external decision is required.

Proceed autonomously through:

1. repository architecture;
2. implementation;
3. unit tests;
4. benchmark generation;
5. optimization experiments;
6. ablation studies;
7. sensitivity analysis;
8. visualization;
9. statistical analysis;
10. documentation;
11. research report.

Use the simplest scientifically defensible implementation.

If there are multiple reasonable approaches, choose one and document why.

Do not silently alter the research question.

Do not replace the project with reinforcement learning or deep learning.

Do not fabricate results.

Do not fabricate citations.

Do not claim a theoretical regret guarantee unless it has actually been established.

Do not claim novelty merely because a combination of existing techniques has been implemented.

Do not optimize hyperparameters against the final evaluation trajectories.

Do not use future objective values to select current actions.

Do not compare algorithms using different random trajectories unless this is explicitly part of an experiment.

For every experiment, preserve enough raw data to reproduce the analysis.

If an adaptive method performs worse than a baseline, report that result and investigate why.

If an implementation choice is mathematically approximate, state the approximation clearly.

At the end, provide:

1. complete source code;
2. tests;
3. configuration files;
4. reproducible experiment commands;
5. raw experiment outputs;
6. plots;
7. tables;
8. research report;
9. README;
10. concise explanation of DA-BO;
11. summary of empirical findings;
12. limitations and future-work directions.

The final product should look like a compact research project in **probabilistic machine learning + optimization + sequential decision-making**, not like a software demonstration.

# safe_BO Hyperparameters

Source script: safe_BO.py
Export date: 2026-03-23

## 1) Main experiment switches

| Name | Current value | Notes |
|---|---:|---|
| introductory_example | True | True: runs toy comparison branch; False: runs iterative PACSBO loop |
| noise_type | "uniform" | Comment lists options: Student-t, Gaussian, uniform, heteroscedastic |
| iterations | 30 | Used when introductory_example=False |

## 2) Core optimization and confidence settings

| Name | Current value | Notes |
|---|---:|---|
| eta | 1e-4 | PACSBO parameter |
| R | 5e-3 | Noise/radius parameter passed through multiple calls |
| delta_confidence | 0.1 | PACSBO confidence level |
| kappa_confidence | 1e-3 | Scenario confidence parameter |
| gamma_confidence | 0.1 | Scenario confidence parameter |
| exploration_threshold | 0.1 | PACSBO exploration threshold |
| RKHS_norm | 1 | RKHS norm bound |

## 3) Function generation / prior settings

| Name | Current value | Notes |
|---|---:|---|
| coeff_distribution | "Gaussian" | Coefficient distribution for random function generation |
| basis_functions | "BSpline" | Basis used by generator |
| kernel | gpytorch.kernels.RBFKernel() | Kernel object |
| kernel.lengthscale | 0.1 | RBF kernel lengthscale |
| lengthscale | 0.1 | Also passed separately to ground_truth and PACSBO |
| Gaussian_std | 1e-2 | Std used in random function generation |

## 4) Reproducibility and discretization

| Name | Current value | Notes |
|---|---:|---|
| random_seed_number | 10 | Used for NumPy and Torch seeds |
| n_dimensions | 1 | Used in compute_X_plot |
| points_per_axis | 1000 | Used in compute_X_plot |
| num_safe_points | 1 | Used in initial_safe_samples |
| safety_threshold_quantile | 0.4 | safety_threshold = quantile(gt.fX, 0.4) |

## 5) Scenario-noise model constants (inside noise_scenario)

| Name | Current value | Notes |
|---|---:|---|
| pi_t | pi^2 * t^2 / 6 | Time-varying factor |
| kappa_t | kappa / pi_t | Adjusted confidence |
| N_scenario | int(log(kappa_t)/log(1-gamma_confidence)) + 1 | Number of scenarios |
| epsilon_t rule | (1/5) * standard_t(df=10, size=N_scenario) * abs(x) | Current synthetic noise sampling rule |
| bar_epsilon_t | max(epsilon_t) | Used for confidence widening |

## 6) Plot/export constants

| Name | Current value | Notes |
|---|---:|---|
| figure.figsize | (16, 12) | Matplotlib default |
| font.size | 30 | Matplotlib default |
| save_plot_step | 8 | Used in plot(..., save=True) |
| toy_plot_step | 5 | Used in introductory_example branch |

## 7) Quick edit template for a new experiment

Fill this block, then mirror values into safe_BO.py:

- experiment_name: <name>
- introductory_example: <True/False>
- noise_type: <uniform/Gaussian/Student-t/heteroscedastic>
- iterations: <int>
- eta: <float>
- R: <float>
- delta_confidence: <float>
- kappa_confidence: <float>
- gamma_confidence: <float>
- exploration_threshold: <float>
- RKHS_norm: <float>
- coeff_distribution: <string>
- basis_functions: <string>
- kernel: <kernel class>
- kernel.lengthscale: <float>
- lengthscale: <float>
- Gaussian_std: <float>
- random_seed_number: <int>
- n_dimensions: <int>
- points_per_axis: <int>
- num_safe_points: <int>
- safety_threshold_quantile: <float>

## 8) Note

There is one hardcoded call using noise_type="uniform" in the ground_truth(...) construction. If you change noise_type above for a new experiment, also update that call to keep behavior consistent.

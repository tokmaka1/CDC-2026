# Spline Experiment Hyperparameters (Reproducibility)

Source files:
- safe_BO.py
- enveloped.py

Export date: 2026-03-23

## 1) Run mode and random seeds

- script_seed: 10
- numpy_seed: 10
- torch_seed: 10
- introductory_example: True

## 2) Core experiment settings (safe_BO.py)

- basis_functions: "BSpline"
- coeff_distribution: "Gaussian"
- noise_type: "uniform"
- iterations: 20
- eta: 1e-4
- R: 5e-3
- delta_confidence: 0.1
- kappa_confidence: 1e-3
- gamma_confidence: 0.1
- exploration_threshold: 0.1
- lengthscale: 0.1
- kernel: RBFKernel
- kernel.lengthscale: 0.1
- Gaussian_std: 1e-2
- RKHS_norm: 1

## 3) Discretization and initialization

- n_dimensions: 1
- points_per_axis: 1000
- number_of_KL_terms: len(X_plot) = 1000
- safety_threshold: quantile(gt.fX, 0.4)
- initial_safe_samples: num_safe_points = 1 (via initial_safe_samples(...))

## 4) Basis construction details (enveloped.py)

For BSpline basis, current code path is:
- ONB = bspline_basis_repeated(X_plot, N, degree=3)

Inside bspline_basis_repeated:
- default M: 30
- degree: 3
- repeats: N // M
- remainder: N % M
- columns repeated to reach N total basis columns

For this experiment (N = 1000):
- repeats = 1000 // 30 = 33
- remainder = 1000 % 30 = 10
- effective basis: 33 full repeats of 30 spline columns + first 10 columns once more

## 5) Coefficient/noise model details (enveloped.py)

Coefficient sampling:
- coeff_distribution = "Gaussian"
- xi ~ Normal(0, Gaussian_std^2), Gaussian_std = 1e-2

Observation noise model in ground_truth.noise(type, x):
- Gaussian: Normal(0, R)
- uniform: Uniform(-R, R)
- Student-t: R * standard_t(df=10)

Scenario random-function noise in generate_noise(noise_type, R, size):
- Gaussian: torch.randn(size) * R
- uniform: -R + 2R * torch.rand(size)
- Student-t: t(df=10) * R

## 6) Introductory plotting branch settings

- step: 1
- wj=True and wj=False are both evaluated for comparison plots

## 7) Notes

- This export records the currently active spline experiment configuration exactly as found in the code on export date.
- If you change basis, coefficient distribution, or M/degree in bspline_basis_repeated, update this file for full reproducibility.

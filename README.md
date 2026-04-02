# Safe Learning-Based Control via Function-Based Uncertainty Quantification

This repository contains the implementation accompanying the paper:

**Safe learning-based control via function-based uncertainty quantification**  
Abdullah Tokmak, Toni Karvonen, Thomas B. Schön, Dominik Baumann  

> Under submission to CDC 2026.

---

## Overview

This project introduces **function-based uncertainty quantification (UQ)** for safe learning-based control. The core idea is to model the unknown function as a random function from which i.i.d. realizations can be sampled, and to construct **high-probability uncertainty tubes** using the scenario approach. :contentReference[oaicite:0]{index=0}

These uncertainty tubes:
- do **not** require RKHS norm bounds or Lipschitz constants
- accommodate **general noise models**, including heavy-tailed and heteroscedastic noise 
- support **flexible function classes**, including discontinuous functions 

The resulting framework is integrated into a **safe Bayesian optimization (BO)** algorithm for control parameter tuning. :contentReference[oaicite:1]{index=1}

---

## Repository Structure

- `main`  
  Contains the **hardware experiments** on the Furuta pendulum, demonstrating safe control parameter tuning.

- `toy_example`  
  Contains a **synthetic example** illustrating the construction of uncertainty tubes and the scenario-based methodology.

---

## Key Idea

We construct uncertainty tubes of the form  
\(
\mathbb{P}\big[h(a) \in [\ell_t(a), u_t(a)] \ \forall a \in \mathcal{A}\big] \geq 1 - \nu
\)  
by sampling function realizations and applying the **scenario approach**. :contentReference[oaicite:2]{index=2}

These tubes are:
- **data-consistent** via projection onto observed data  
- **uniform over the domain**  
- **computationally tractable** via convex programs or closed-form extrema  

---

## Contributions

- Function-based probabilistic modeling of unknown functions  
- Scenario-based construction of uncertainty tubes  
- Wait-and-judge refinement for tighter bounds  
- Safe BO algorithm with **provable safety guarantees**  

---

## Getting Started

Clone the repository:
```bash
git clone https://github.com/tokmaka1/CDC-2026.git
cd CDC-2026
```

## Cite as follows
@misc{tokmak2026function,
  title={Safe learning-based control via function-based uncertainty quantification},
  author={Tokmak, Abdullah and Karvonen, Toni and Schön, Thomas B. and Baumann, Dominik},
  note={Under submission to IEEE Conference on Decision and Control (CDC)},
  year={2026}
}

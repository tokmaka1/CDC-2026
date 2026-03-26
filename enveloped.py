# KL expansion
# From very Gaussian to non-Gaussian

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
# import tikzplotlib
# import cvxpy as cp
import math
import warnings
from tqdm import tqdm
from scipy.stats import binom


def compute_X_plot(n_dimensions, points_per_axis):
    X_plot_per_domain = torch.linspace(0, 1, points_per_axis)
    X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
    X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    return X_plot


def epsilon_wait_and_judge(s, num_scenario, beta, tol=1e-10, max_iter=200):
    k = int(s)
    m = np.arange(k, num_scenario + 1, dtype=int)
    # log binomials
    logCmk = np.array([
        math.lgamma(mi + 1) - math.lgamma(k + 1) - math.lgamma(mi - k + 1)
        for mi in m
    ])
    logCNk = math.lgamma(num_scenario + 1) - math.lgamma(k + 1) - math.lgamma(num_scenario - k + 1)
    # def f(t):
    #     lt = math.log(t)
    #     a = np.max(logCmk + (m - k) * lt)
    #     sum_term = np.exp(logCmk + (m - k) * lt - a).sum()
    #     term1 = (beta / (num_scenario + 1)) * math.exp(a) * float(sum_term)
    #     term2 = math.exp(logCNk + (num_scenario - k) * lt)
    #     return term1 - term2

    def f(t):
        lt = math.log(t)

        vals = logCmk + (m - k) * lt
        a = np.max(vals)
        sum_term = np.exp(vals - a).sum()

        log_term1 = math.log(beta) - math.log(num_scenario + 1) + a + math.log(sum_term)
        log_term2 = logCNk + (num_scenario - k) * lt

        b = max(log_term1, log_term2)
        return math.exp(log_term1 - b) - math.exp(log_term2 - b)

    lo, hi = 1e-16, 1.0 - 1e-16
    flo, fhi = f(lo), f(hi)
    
    # Handle case where bracketing fails (e.g., when s is very small)
    if not (flo > 0 and fhi < 0):
        # Return a reasonable default: either 0 (no epsilon needed) or 1 (full epsilon)
        # If both same sign, likely degenerate case - return minimal epsilon
        # warnings.warn("Returning default epsilon=0.0 due to lack of proper bracketing in bisection method.")
        return 0.0
    
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        # CORRECT bisection update
        if fmid > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    t_star = 0.5 * (lo + hi)
    return 1.0 - t_star


def m_wait_and_judge(s, epsilon, beta, max_m=20000, tol=1e-10, max_iter=200):
    """
    Find the minimal number of scenarios m such that epsilon_wait_and_judge(s, m, beta) <= epsilon.
    """
    k = int(s)
    lo = k + 1  # minimal num_scenario is k+1
    hi = max_m
    # Check if at hi it satisfies
    eps_hi = epsilon_wait_and_judge(s, hi, beta)
    if eps_hi > epsilon:
        raise ValueError(f"Even at max_m={max_m}, epsilon={eps_hi} > {epsilon}. Increase max_m.")
    # At lo, eps_lo = epsilon_wait_and_judge(s, lo, beta), which should be > epsilon usually
    for _ in range(max_iter):
        mid = int(0.5 * (lo + hi))
        eps_mid = epsilon_wait_and_judge(s, mid, beta)
        if eps_mid <= epsilon:
            hi = mid
        else:
            lo = mid + 1
        if hi - lo <= 1:
            break
    # Now hi is the smallest m where eps <= epsilon
    return hi


class ground_truth():
    def __init__(self, coeff_distribution, Gaussian_std, X_plot, kernel, noise_type="uniform", R=1e-1):
        N = len(X_plot)
        self.X_plot = X_plot
        self.R = R
        self.noise_type = noise_type
        self.kernel = kernel

        # Compute kernel matrix efficiently without storing full matrix
        K = kernel(X_plot, X_plot).evaluate()
        ONB = K

        # Sample coefficients xi the same way as in create_random_functions
        xi = sample_coefficients(coeff_distribution, ONB, Gaussian_std)
        self.xi = xi

        # Compute fX for plotting
        self.fX = (ONB * xi).sum(dim=1)
        # self.fX = torch.tensor([
            # ONB[i] * xi for i in range(N)
        # ])

        # Define the function evaluator for arbitrary x
        def f_eval(x):
            ONB = kernel(x.reshape(-1, self.X_plot.shape[1]), self.X_plot).to_dense()
            return (ONB * xi).sum(dim=1)
        self.f = f_eval

    def noise(self, type):  # let us just consider uniform noise here

        if type == "uniform":
            return -self.R + 2*self.R*np.random.uniform()
    def conduct_experiment(self, x):
        return torch.tensor(self.f(x)  + self.noise(self.noise_type), dtype=torch.float32)

def generate_noise(noise_type, R, size, x=None):
    if noise_type == "uniform":
        return -R + 2 * R * torch.rand(size)


def sample_coefficients(coeff_distribution, ONB, Gaussian_std):
    if coeff_distribution == "Gaussian":
        return torch.randn(ONB.shape[1], dtype=ONB.dtype, device=ONB.device) * Gaussian_std
    raise ValueError(f"Unknown coeff_distribution: {coeff_distribution}")


def create_random_functions(coeff_distribution, Gaussian_std, X_plot, kernel, X_sample,
                             Y_sample, gamma_confidence, kappa_confidence, wj=True, noise_type="uniform", R=1e-1, t=1):
    N = len(X_plot)
    # list_random_RKHS_norms.append(torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha))
    K = kernel(X_plot, X_plot).to_dense()  # just this!
    ONB = K


        

    y_interpol = Y_sample  # still interpolation, no noise
    # Compare each sample point against each grid point across all coordinates.
    dist = torch.cdist(X_plot, X_sample)  # (10000, 2)
    interpolation_indices = dist.argmin(dim=0).tolist()                          
    # WJ is True
    s = 1  # init
    epsilon_iterative = np.inf
    while epsilon_iterative >= gamma_confidence:
        m_functions = m_wait_and_judge(s=s, epsilon=gamma_confidence, beta=kappa_confidence, max_m=20000, tol=1e-10, max_iter=200)
        tensor_random_functions = torch.zeros([m_functions, len(X_plot)])
        A = ONB[interpolation_indices, :]
        G = A @ A.T
        noise_matrix = torch.zeros([len(y_interpol), m_functions])
        xi0_matrix = torch.zeros([ONB.shape[1], m_functions], dtype=ONB.dtype, device=ONB.device)
        
        for j in range(m_functions):
            noise = generate_noise(noise_type, R, len(y_interpol), X_sample if noise_type == "heteroscedastic" else None)
            noise_matrix[:, j] = noise
            xi0 = sample_coefficients(coeff_distribution, ONB, Gaussian_std)
            xi0_matrix[:, j] = xi0
        y = y_interpol.unsqueeze(1) - noise_matrix  
        jitter = 5e-2  # needs to go up if xi goes up

        
        res = y - (A @ xi0_matrix)
        alpha = torch.linalg.solve(G + jitter*torch.eye(G.shape[0], device=G.device, dtype=G.dtype), res)
        xi0_matrix = xi0_matrix + A.T @ alpha
        # random_function = (ONB * xi).sum(dim=1)
        # tensor_random_functions[j, :] = random_function.flatten()
        tensor_random_functions = (ONB @ xi0_matrix).T
        ub, argmax = torch.max(tensor_random_functions, dim=0)
        lb, argmin = torch.min(tensor_random_functions, dim=0)
        support = torch.unique(torch.cat([argmax, argmin], dim=0))
        s = support.numel()
        epsilon_iterative = epsilon_wait_and_judge(s, m_functions, beta=6*kappa_confidence/(np.pi**2*t**2), tol=1e-10)

    return lb, ub, argmin, argmax, tensor_random_functions, support







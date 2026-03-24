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

def bspline_basis_torch(x, N, degree=3):
    x = x.view(-1)

    # knot vector (clamped, uniform)
    n_knots = N + degree + 1
    knots = torch.zeros(n_knots, dtype=x.dtype, device=x.device)

    n_interior = n_knots - 2*(degree+1)
    if n_interior > 0:
        interior = torch.linspace(0, 1, n_interior + 2, dtype=x.dtype, device=x.device)[1:-1]
        knots[degree+1:degree+1+n_interior] = interior
    knots[-(degree+1):] = 1.0

    # avoid boundary issue at x = 1
    x = torch.clamp(x, max=1.0 - 1e-12)

    # degree 0
    B = torch.zeros((N, len(x)), dtype=x.dtype, device=x.device)
    for i in range(N):
        # B[i] = ((x >= knots[i]) & (x < knots[i+1])).to(x.dtype)
        if i == N - 1:
            B[i] = ((x >= knots[i]) & (x <= knots[i+1])).to(x.dtype)  # end point consistency
        else:
            B[i] = ((x >= knots[i]) & (x < knots[i+1])).to(x.dtype)

    # recursion
    for k in range(1, degree+1):
        B_new = torch.zeros_like(B)
        for i in range(N):
            denom1 = knots[i+k] - knots[i]
            denom2 = knots[i+k+1] - knots[i+1]

            term1 = 0.0
            term2 = 0.0

            if denom1 > 0:
                term1 = (x - knots[i]) / denom1 * B[i]

            if denom2 > 0 and i+1 < N:
                term2 = (knots[i+k+1] - x) / denom2 * B[i+1]

            B_new[i] = term1 + term2
        B = B_new

    return B.T  # shape: [len(x), N]

def bspline_basis_repeated(x, N, M=1000, degree=3):
    # generate true spline basis (size: len(x) x M)
    Phi_small = bspline_basis_torch(x, M, degree)  # (n, M)

    # repeat columns to reach N
    repeats = N // M
    remainder = N % M

    Phi = Phi_small.repeat(1, repeats)  # (n, M*repeats)

    if remainder > 0:
        Phi = torch.cat([Phi, Phi_small[:, :remainder]], dim=1)

    return Phi  # shape: (len(x), N)


def polynomial_basis_1d_scaled(x, p):
    x = x.view(-1, 1)
    return torch.cat([x**k for k in range(p)], dim=1)


def haar_wavelet(n, x):
    """Evaluate the n-th Haar wavelet at points x ∈ [0,1]."""
    x = x.reshape(-1)
    if n == 0:
        return torch.ones_like(x)

    k = int(torch.floor(torch.log2(torch.tensor(n, dtype=torch.float32))))
    p = n - 2**k
    width = 1 / 2**k
    start = p * width
    mid = start + width / 2
    end = start + width

    h = torch.zeros_like(x)
    h[(x >= start) & (x < mid)] = 1.0
    h[(x >= mid) & (x < end)] = -1.0
    return h * (2**(k / 2))

def generate_haar_basis(x, N):
    """Return a [len(x) x N] matrix of Haar basis functions evaluated at x."""
    x = x.reshape(-1)  # keep singletons as length-1 vectors
    basis = [haar_wavelet(n, x) for n in range(N)]  # list of 1D tensors
    return torch.stack(basis, dim=1)  # shape [len(x), N]

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
        warnings.warn("Returning default epsilon=0.0 due to lack of proper bracketing in bisection method.")
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


def m_wait_and_judge(s, epsilon, beta, max_m=10000, tol=1e-10, max_iter=200):
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
    def __init__(self, coeff_distribution, Gaussian_std, X_plot, basis_functions, kernel, lengthscale, noise_type="uniform", R=1e-1):
        N = len(X_plot)
        self.X_plot = X_plot
        self.R = R
        self.noise_type = noise_type
        self.basis_functions = basis_functions
        self.kernel = kernel
        self.lengthscale = lengthscale

        # Build the basis matrix for X_plot
        if basis_functions == "Haar":
            ONB = generate_haar_basis(X_plot, N)
        elif basis_functions == "BSpline":
            ONB = bspline_basis_repeated(X_plot, N, degree=3)
        elif basis_functions == "RKHS":
            K = kernel(X_plot, X_plot).to_dense()
            ONB = K
        elif basis_functions == "polynomial":
            ONB = polynomial_basis_1d_scaled(X_plot, p=N)  # c is a scaling factor to control the norm of the basis functions
        else:
            raise ValueError(f"Unknown basis_functions: {basis_functions}")

        # Sample coefficients xi the same way as in create_random_functions
        xi = sample_coefficients(coeff_distribution, ONB, Gaussian_std)
        self.xi = xi

        # Compute fX for plotting
        self.fX = (ONB * xi).sum(dim=1)

        # Define the function evaluator for arbitrary x
        def f_eval(x):
            if basis_functions == "Haar":
                phi = generate_haar_basis(x, N)
                return (phi * xi).sum(dim=1)
            elif basis_functions == "BSpline":
                ONB = bspline_basis_repeated(x, N, degree=3)
                return (ONB * xi).sum(dim=1)
            elif basis_functions == "RKHS":
                return kernel(x.reshape(-1, self.X_plot.shape[1]), self.X_plot) @ xi
            elif basis_functions == "polynomial":
                ONB = polynomial_basis_1d_scaled(x, p=N)
                return (ONB * xi).sum(dim=1)
            else:
                raise ValueError(f"Unknown basis_functions: {basis_functions}")

        self.f = f_eval

    def noise(self, type, x):
        if type == "Gaussian":
            return np.random.normal(loc=0.0, scale=self.R)
        if type == "uniform":
            return -self.R + 2*self.R*np.random.uniform()
        if type == "Student-t":
            return self.R*np.random.standard_t(df=10)
    def conduct_experiment(self, x):
        return torch.tensor(self.f(x))  #  + self.noise(self.noise_type, x), dtype=torch.float32)

def generate_noise(noise_type, R, size, x=None):
    from scipy.stats import t
    if noise_type == "Gaussian":
        return torch.randn(size) * R
    elif noise_type == "uniform":
        return -R + 2 * R * torch.rand(size)
    elif noise_type == "Student-t":
        return torch.tensor(t.rvs(df=10, size=size)) * R
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def sample_coefficients(coeff_distribution, ONB, Gaussian_std):
    if coeff_distribution == "Gaussian":
        return torch.randn(ONB.shape[1], dtype=ONB.dtype, device=ONB.device) * Gaussian_std
    if coeff_distribution == "uniform":
        return (2 * Gaussian_std) * torch.rand(ONB.shape[1], dtype=ONB.dtype, device=ONB.device) - Gaussian_std
    if coeff_distribution == "Student-t":
        dist = torch.distributions.StudentT(df=10.0)
        return dist.sample((ONB.shape[1],)).to(dtype=ONB.dtype, device=ONB.device) * Gaussian_std
    raise ValueError(f"Unknown coeff_distribution: {coeff_distribution}")


def create_random_functions(coeff_distribution, Gaussian_std, X_plot, basis_functions, kernel, X_sample,
                             Y_sample, gamma_confidence, kappa_confidence, wj=True, noise_type="uniform", R=1e-1, t=1):
    N = len(X_plot)
    # list_random_RKHS_norms.append(torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha))
    if basis_functions == "Haar":
        phi_n = generate_haar_basis(X_plot, N)
        ONB = phi_n  # Hadamard product, we will multiply xi with this; scaled eigenvectors
    elif basis_functions == "BSpline":
            ONB = bspline_basis_repeated(X_plot, N, degree=3)
    elif basis_functions == "RKHS":
        K = kernel(X_plot, X_plot).to_dense()
        ONB = K
    elif basis_functions == "polynomial":
        ONB = polynomial_basis_1d_scaled(X_plot, p=N)


        

    y_interpol = Y_sample  # still interpolation, no noise
    mask = (X_plot == X_sample.T)          # shape [n, m]
    interpolation_indices = mask.int().argmax(dim=0).tolist()

    if wj is True:
        s = 1  # init
        epsilon_iterative = np.inf
        while epsilon_iterative >= gamma_confidence:
            m_functions = m_wait_and_judge(s=s, epsilon=gamma_confidence, beta=kappa_confidence, max_m=10000, tol=1e-10, max_iter=200)
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

    elif wj is False:
        # Determine m_functions directly via classic scenario approach
        high = 100  # init
        while True:
            lhs = binom.cdf(2*N-1, high, gamma_confidence)
            if lhs > 6*kappa_confidence/(np.pi**2*t**2):
                low = high
                high *= 2
            else:  # m_function_new is large enough, m_functions_old is not, so we can do binary search between them
                break
        while low + 1 < high:
            mid = (low + high) // 2
            lhs = binom.cdf(2*N-1, mid, gamma_confidence)
            if lhs > 6*kappa_confidence/(np.pi**2*t**2):
                low = mid
            else:
                high = mid
        m_functions = high
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



    return lb, ub, argmin, argmax, tensor_random_functions, support






if __name__ == '__main__':

    random_seed_number = 11
    np.random.seed(random_seed_number)
    torch.manual_seed(random_seed_number)

    number_of_samples = 3
    gamma_confidence = 0.1
    kappa_confidence = 1e-3
    coeff_distribution = "Gaussian"  # Options: Gaussian, uniform, Student-t, Fourier_decay
    basis_functions = "RKHS"  # Haar wavelet basis functions, not in RKHS, or "RKHS" for standard
    number_of_KL_terms = 1000  # Number of KL terms to keep
    X_plot = compute_X_plot(n_dimensions=1, points_per_axis=1000)
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = 0.1
    nugget_factor = 0  # 1e-5
    X_sample = X_plot[500]
    Y_sample = torch.tensor([0])
    lb, ub, argmin, argmax, tensor_random_functions, support = \
    create_random_functions(coeff_distribution, X_plot, basis_functions, kernel, X_sample,
                             Y_sample, gamma_confidence, kappa_confidence)
    plt.figure()
    plt.plot(X_plot, tensor_random_functions[support[0], :].detach().numpy(), 'magenta', alpha=0.1, label="Scenario")
    for i in range(1, len(support)):
        plt.plot(X_plot, tensor_random_functions[support[i], :].detach().numpy(), 'magenta', alpha=0.1)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.2)  # , 'gray', alpha=0.2
    plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
    plt.plot(X_plot, lb.detach().numpy(), 'blue', label="lb, ub")
    plt.plot(X_plot, ub.detach().numpy(), 'blue')
    plt.plot(X_plot, tensor_random_functions[0, :].detach().numpy(), 'black', label="Truth")
    plt.legend()
    plt.title("Wait and judge bounds")

    # Running example: \phi_i(x) = 1,x,x^2,x^3,x^4, ..., x^100 or x^1000


    plt.figure()
    plt.plot(X_plot, tensor_random_functions[support[0], :].detach().numpy(), 'magenta', alpha=0.1, label="Support scenarios")
    for i in range(1, len(support)):
        plt.plot(X_plot, tensor_random_functions[support[i], :].detach().numpy(), 'magenta', alpha=0.1)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.2)  # , 'gray', alpha=0.2
    plt.plot(X_plot, lb.detach().numpy(), 'blue', label="lb, ub")
    plt.plot(X_plot, ub.detach().numpy(), 'blue')
    plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
    plt.plot(X_plot, tensor_random_functions[0, :].detach().numpy(), 'black', label="Truth")
    plt.legend()
    plt.title("Theorem 1 (support constraints)")
    # plt.savefig("Theorem_1_support.pdf")

    '''

    plt.figure()
    plt.plot(X_plot, tensor_random_functions[support[0], :].detach().numpy(), 'magenta', alpha=0.1, label="Scenario")
    for i in range(1, len(support)):
        plt.plot(X_plot, tensor_random_functions[support[i], :].detach().numpy(), 'magenta', alpha=0.1)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.2, label='Scenario bounds')  # , 'gray', alpha=0.2
    plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
    plt.plot(X_plot, tensor_random_functions[0, :].detach().numpy(), 'blue', alpha=0.05, label="Truth")
    plt.legend()
    # plt.savefig("Students_t.png")
    plt.show()

    # 
    t=1
    kappa=1e-3
    kappa_t = 6*kappa/(t**2*np.pi**2)
    nu=1e-1
    N=1000
    m_start = 1
    m=m_start
    while binom.cdf(2*N-1, m, nu)>kappa_t:
        m+=100
    while binom.cdf(2*N-1, m, nu)< kappa_t:
        m-=1
    print(m+1)
    '''


    '''
    support = torch.unique(torch.cat([argmax, argmin], dim=0))
    s = support.numel()

    print("support scenarios (two-sided tube):", s)
    # Now get the wait and judge epsilon
    epsilon_wj = epsilon_wait_and_judge(s, m_functions, beta=1e-3, tol=1e-10)


    # Optimization problem
    x = X_plot.detach().cpu().numpy().reshape(-1)
    n = x.shape[0]

    # Scenario values on grid: G[j,i] = g_j(x_i)  -> shape (m, n)
    G = tensor_random_functions.detach().cpu().numpy()
    m = G.shape[0]
    W = np.column_stack([np.ones(n), x, x**2, x**3, x**4, x**5, x**6])

    # Interpolation (sample) constraints
    idx = np.array(interpolation_indices, dtype=int)
    y = y_interpol.detach().cpu().numpy().reshape(-1)

    c_u = cp.Variable(7)
    c_l = cp.Variable(7)

    u = W @ c_u   # (n,)
    l = W @ c_l   # (n,)


    constraints = [
        l[None, :] <= G,
        G <= u[None, :],
        u[idx] == y,
        l[idx] == y,
    ]

    objective = cp.Minimize(cp.sum(u - l))
    prob = cp.Problem(objective, constraints)

    prob.solve(
        solver=cp.SCS,
        verbose=True,        # prints progress
        max_iters=1000,      # hard cap (default is much larger)
        eps=1e-3,            # accuracy; 1e-3–1e-4 is plenty here
        alpha=1.5            # often speeds convergence
    )
    # if prob.status not in ("optimal", "optimal_inaccurate"):
    #    prob.solve(solver=cp.SCS, verbose=False)
    print(prob.status)

    c_u_hat = c_u.value
    c_l_hat = c_l.value
    u_hat = (W @ c_u_hat)  # (n,)
    l_hat = (W @ c_l_hat)  # (n,)
    '''


    # Scenario/pointwise approach
    # Plots
    '''
    plt.figure()
    plt.plot(X_plot, tensor_random_functions[0, :].detach().numpy(), 'magenta', alpha=0.05, label="Scenario")
    for i in range(1, m_functions):
        plt.plot(X_plot, tensor_random_functions[i, :].detach().numpy(), 'magenta', alpha=0.05)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.2, label='Scenario bounds')  # , 'gray', alpha=0.2
    plt.plot(X_plot, ground_truth.detach().numpy(), '-b', label='Ground truth')
    plt.scatter(x_interpol.detach().numpy(), y_interpol.detach().numpy(), color='k', s=100, label='Samples')
    plt.legend()
    # plt.savefig("Students_t.png")
    plt.show()


    # Convex approach
    plt.figure()
    plt.plot(X_plot, tensor_random_functions[0, :].detach().numpy(), 'magenta', alpha=0.05, label="Scenario")
    for i in range(1, m_functions):
        plt.plot(X_plot, tensor_random_functions[i, :].detach().numpy(), 'magenta', alpha=0.05)
    plt.plot(X_plot, u_hat, 'red')
    plt.plot(X_plot, l_hat, 'red', label="Convex bounds")
    # plt.fill_between(X_plot.flatten().detach().numpy(), l_hat, u_hat, alpha=0.2, label='Convex bounds')  # , 'gray', alpha=0.2
    plt.plot(X_plot, ground_truth.detach().numpy(), '-b', label='Ground truth')
    plt.scatter(x_interpol.detach().numpy(), y_interpol.detach().numpy(), color='k', s=100, label='Samples')
    plt.legend()
    # plt.savefig("Students_t.png")
    plt.show()

    # Plotting both
    plt.figure()
    plt.plot(X_plot, tensor_random_functions[0, :].detach().numpy(), 'magenta', alpha=0.05, label="Scenario")
    for i in range(1, m_functions):
        plt.plot(X_plot, tensor_random_functions[i, :].detach().numpy(), 'magenta', alpha=0.01)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.5, label='Scenario bounds')  # , 'gray', alpha=0.2
    plt.plot(X_plot, ground_truth.detach().numpy(), '-b', label='Ground truth')
    plt.scatter(x_interpol.detach().numpy(), y_interpol.detach().numpy(), color='k', s=100, label='Samples')
    plt.plot(X_plot, u_hat, 'red', label="Parametrized bounds")
    plt.plot(X_plot, l_hat, 'red')
    plt.legend()
    # plt.savefig("Students_t.png")
    plt.show()





    # Support constraints
    plt.figure()
    plt.plot(X_plot, tensor_random_functions[support[0], :].detach().numpy(), 'magenta', alpha=0.1, label="Scenario")
    for i in range(1, len(support)):
        plt.plot(X_plot, tensor_random_functions[support[i], :].detach().numpy(), 'magenta', alpha=0.1)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.2, label='Scenario bounds')  # , 'gray', alpha=0.2
    plt.plot(X_plot, ground_truth.detach().numpy(), '-b', label='Ground truth')
    plt.scatter(x_interpol.detach().numpy(), y_interpol.detach().numpy(), color='k', s=100, label='Samples')
    plt.legend()
    # plt.savefig("Students_t.png")
    plt.show()
    '''
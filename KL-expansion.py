# KL expansion
# From very Gaussian to non-Gaussian

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

def compute_X_plot(n_dimensions, points_per_axis):
    X_plot_per_domain = torch.linspace(0, 1, points_per_axis)
    X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
    X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    return X_plot


def haar_wavelet(n, x):
    """Evaluate the n-th Haar wavelet at points x ∈ [0,1]."""
    if n == 0:
        return torch.ones_like(x)

    k = int(torch.floor(torch.log2(torch.tensor(n, dtype=torch.float32))))
    p = n - 2**k
    width = 1 / 2**k
    start = p * width
    mid = start + width / 2
    end = start + width

    h = torch.zeros_like(x)
    h = h.squeeze()  # remove any [1000, 1] shape
    h[(x >= start) & (x < mid)] = 1.0
    h[(x >= mid) & (x < end)] = -1.0
    return h * (2**(k / 2))

def generate_haar_basis(x, N):
    """Return a [len(x) x N] matrix of Haar basis functions evaluated at x."""
    x = x.squeeze()  # ensure 1D input
    basis = [haar_wavelet(n, x) for n in range(N)]  # list of 1D tensors
    return torch.stack(basis, dim=1)  # shape [len(x), N]


if __name__ == '__main__':
    # random_seed_number = 42
    # np.random.seed(random_seed_number)
    # torch.manual_seed(random_seed_number)

    coeff_distribution = "Gaussian"  # Options: Students_t, Gaussian
    basis_functions = "RKHS"  # Haar wavelet basis functions, not in RKHS, or "RKHS" for standard
    iterations = 5
    N = 250  # Number of KL terms to keep
    m_functions = 100  # number of random functions
    X_plot = compute_X_plot(n_dimensions=1, points_per_axis=1000)
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = 0.1
    if coeff_distribution == "Gaussian":
        xi = torch.randn(N)  # mean 0, std 1!
    elif coeff_distribution == "Students_t":
        df = 1  # degrees of freedom, smaller = heavier tail
        xi = torch.distributions.StudentT(df).rsample((N,))     
    if basis_functions == "RKHS":   
        K = kernel(X_plot, X_plot).to_dense()
        # eigh returns ascending eigenvalues — flip them
        lambda_n, phi_n = torch.linalg.eigh(K)
        lambda_n = lambda_n.flip(dims=[-1])[:N]      # Top-N eigenvalues
        phi_n = phi_n.flip(dims=[-1])[:, :N]         # Corresponding eigenfunctions
    elif basis_functions == "Haar":
        lambda_n = torch.ones(N)
        phi_n = generate_haar_basis(X_plot, N)
    ground_truth =  (phi_n * (torch.sqrt(lambda_n) * xi)).sum(dim=1)
    interpolation_indices = []
    tensor_random_functions = torch.zeros([m_functions, len(X_plot)])
    # When we discretize the domain (as in the PyTorch code), the covariance operator becomes a matrix K, and we compute its eigendecomposition
    for t in range(iterations):  # we go with random.choice
        random_index = torch.randint(low=0, high=1000, size=[]).item()
        interpolation_indices.append(random_index)
        x_interpol = X_plot[interpolation_indices]
        y_interpol = ground_truth[interpolation_indices]
        for j in range(m_functions):
            if coeff_distribution == "Gaussian":
                xi_tail = torch.randn(N-len(x_interpol))
            elif coeff_distribution == "Students_t":
                df = 2  # degrees of freedom, smaller = heavier tail
                xi_tail = torch.distributions.StudentT(df).rsample((N-len(x_interpol),))
            phi_n_tail = phi_n[interpolation_indices, len(x_interpol):]  # only at interpolation rows, and only the last columns
            phi_n_head = phi_n[interpolation_indices, :len(x_interpol)]
            # y_tail = phi_n_tail @ xi_tail
            y_tail = phi_n_tail * torch.sqrt(lambda_n[len(x_interpol):]) @ xi_tail
            y_head = y_interpol - y_tail
            xi_head = torch.linalg.inv(phi_n_head * torch.sqrt(lambda_n[:len(x_interpol)])) @ y_head
            xi = torch.cat((xi_head, xi_tail))
            random_function = (phi_n * (torch.sqrt(lambda_n) * xi)).sum(dim=1)
            tensor_random_functions[j, :] = random_function.flatten()
            ub, _ = torch.max(tensor_random_functions, axis=0)  # We can have this as a concentrating upper and lower bound! But not that important now
            lb, _ = torch.min(tensor_random_functions, axis=0)

    plt.figure()
    for i in range(m_functions):
        plt.plot(X_plot, tensor_random_functions[i, :].detach().numpy(), 'magenta', alpha=0.15)
    plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.1, label='Scenario bounds')  # , 'gray', alpha=0.2
    plt.plot(X_plot, ground_truth.detach().numpy(), '-b', label='Ground truth')
    plt.scatter(x_interpol.detach().numpy(), y_interpol.detach().numpy(), color='k', s=50, label='Samples')
    plt.legend()
    # plt.savefig('bounds_GP.png')
    plt.show()


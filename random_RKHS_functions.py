# Random RKHS functions
# Is this still "Gaussian"? Maybe yes, but the random function generation can be completely different
# See Figure 1 in Latex document


import torch
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
import tikzplotlib

class ground_truth():
    def __init__(self, num_center_points, X_plot, RKHS_norm):
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        # For ground truth
        self.X_plot = X_plot
        self.RKHS_norm = RKHS_norm
        random_indices_center = torch.randint(high=self.X_plot.shape[0], size=(num_center_points,))
        self.X_center = self.X_plot[random_indices_center]
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = 0.1  # used in all runs except for Furuta hardware. There, this will get over-written.
        RKHS_norm_squared = alpha.T @ self.kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.f = fun(self.kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)

    def conduct_experiment(self, x):
        return torch.tensor(self.f(x), dtype=x.dtype)  # no noise!

def compute_X_plot(n_dimensions, points_per_axis):
    X_plot_per_domain = torch.linspace(0, 1, points_per_axis)
    X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
    X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    return X_plot





if __name__ == '__main__':
    RKHS_norm = 1
    X_plot = compute_X_plot(n_dimensions=1, points_per_axis=1000)
    gt = ground_truth(num_center_points=500, X_plot=X_plot, RKHS_norm=RKHS_norm)
    m_RKHS_functions = 100
    iterations = 5
    alpha_bar = 0.1
    nugget_factor = 1e-6
    kernel = gt.kernel
    tensor_random_RKHS_functions = torch.zeros([m_RKHS_functions, len(X_plot)])
    x_interpol = torch.tensor([])
    y_interpol = gt.conduct_experiment(x_interpol)
    N_hat = 1000  # 1000 center points, just like the ground truth. We can work on this

    for t in range(iterations):
        list_random_RKHS_norms = []
        for j in range(m_RKHS_functions):
            X_c = (torch.min(X_plot) - torch.max(X_plot))*torch.rand(N_hat, X_plot.shape[1]) + torch.max((X_plot))
            X_c_tail = X_c[len(x_interpol):]
            X_c[:len(x_interpol)] = x_interpol.unsqueeze(1)
            alpha_tail = -2*alpha_bar*torch.rand(N_hat-len(y_interpol), 1) + alpha_bar
            y_tail = kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
            y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
            alpha_head = torch.inverse(kernel(x_interpol, x_interpol).evaluate()+torch.eye(len(y_interpol))*nugget_factor) @ y_head
            alpha = torch.cat((alpha_head, alpha_tail))
            # We want the functions, not the RKHS norms!
            random_RKHS_function = kernel(X_plot, X_c).evaluate() @ alpha
            tensor_random_RKHS_functions[j, :] = random_RKHS_function.flatten()
            list_random_RKHS_norms.append(torch.sqrt(alpha.T @ kernel(X_c, X_c).evaluate() @ alpha))
        K_interpol_inverse = torch.inverse(kernel(x_interpol, x_interpol).to_dense()+torch.eye(len(y_interpol))*nugget_factor)  # torch.inverse(kernel(x_interpol, x_interpol).to_dense())
        interpolating_function = y_interpol.T @ K_interpol_inverse @ kernel(x_interpol, X_plot)
        interpolating_function_RKHS_norm = y_interpol.T @ K_interpol_inverse @ y_interpol
        # power_function_squared = (kernel(X_plot, X_plot) - kernel(X_plot, x_interpol) @ K_interpol_inverse @ kernel(x_interpol, X_plot)).to_dense()
        product = kernel(X_plot, x_interpol) @ K_interpol_inverse
        tensor_dot_product = torch.sum(product * kernel(X_plot, x_interpol).to_dense(), dim=1, keepdim=True)
        power_function_squared = 1 - tensor_dot_product
        power_function = torch.sqrt(power_function_squared).flatten()
        interpolating_function_ub = interpolating_function + power_function*torch.sqrt((RKHS_norm**2 - interpolating_function_RKHS_norm**2))
        interpolating_function_lb = interpolating_function - power_function*torch.sqrt((RKHS_norm**2 - interpolating_function_RKHS_norm**2))
        # Credible intervals
        ub, _ = torch.max(tensor_random_RKHS_functions, axis=0)  # We can have this as a concentrating upper and lower bound! But not that important now
        lb, _ = torch.min(tensor_random_RKHS_functions, axis=0)
        numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
        numpy_list.sort()
        r_final = 0  # potentially with discarding
        B = numpy_list[-1-r_final]
        interpolating_function_ub_AISTATS = interpolating_function + power_function*torch.sqrt((B**2 - interpolating_function_RKHS_norm**2))
        interpolating_function_lb_AISTATS = interpolating_function - power_function*torch.sqrt((B**2 - interpolating_function_RKHS_norm**2))

        # plt.figure()
        # for i in range(m_RKHS_functions):
        #     plt.plot(X_plot, tensor_random_RKHS_functions[i, :].detach().numpy(), 'magenta', alpha=0.15)
        # plt.fill_between(X_plot.flatten().detach().numpy(), lb.detach().numpy(), ub.detach().numpy(), alpha=0.1, label='Scenario bounds')  # , 'gray', alpha=0.2
        # plt.fill_between(X_plot.flatten().detach().numpy(), interpolating_function_lb.flatten().detach().numpy(), interpolating_function_ub.flatten().detach().numpy(), alpha=0.3, label='Kernel interpolation bounds')  # , 'gray', alpha=0.2)
        # plt.plot(X_plot, gt.conduct_experiment(X_plot), '-b', label='Ground truth')
        # plt.plot(X_plot, interpolating_function.detach().numpy(), '-g', label='Interpolation')
        # plt.legend()
        # plt.show()
        x = torch.rand(1)  # new sample; random
        y = gt.conduct_experiment(x)
        x_interpol = torch.cat((x_interpol, x), dim=0)  # torch.cat((x_interpol, x.unsqueeze(0)), dim=0)
        y_interpol = torch.cat((y_interpol, y), dim=0)
    plt.figure()
    for i in range(m_RKHS_functions):
        if True:
            plt.plot(X_plot[::25], tensor_random_RKHS_functions[i, :].detach().numpy()[::25], 'magenta', alpha=0.05)
    plt.fill_between(X_plot.flatten().detach().numpy()[::25], interpolating_function_lb_AISTATS.flatten().detach().numpy()[::25],
                      interpolating_function_ub_AISTATS.flatten().detach().numpy()[::25],
                      alpha=0.1, label='Kernel interpolation AISTATS')  # , 'gray', alpha=0.2)
    plt.fill_between(X_plot.flatten().detach().numpy()[::25], lb.detach().numpy()[::25], ub.detach().numpy()[::25], alpha=0.2, label='Scenario bounds')  # , 'gray', alpha=0.2
    # plt.fill_between(X_plot.flatten().detach().numpy(), interpolating_function_lb.flatten().detach().numpy(), interpolating_function_ub.flatten().detach().numpy(), alpha=0.3, label='Kernel interpolation bounds')  # , 'gray', alpha=0.2)
    plt.plot(X_plot[::25], gt.conduct_experiment(X_plot[::25]), '-b', label='Ground truth')
    plt.plot(X_plot[::25], interpolating_function.detach().numpy()[::25], '-g', label='Interpolation')
    # plt.legend()
    # plt.savefig('bound_comparison_pre_RKHS.png')
    # tikzplotlib.save("AISTATS_scenario.tex", float_format="%.3f")
    # tikzplotlib.save("AISTATS_scenario.tex", float_format=".3f")
    tikzplotlib.save("AISTATS_scenario.tex")
    #plt.show()






# How does kernel interpolation compare?
# Additional assumption: We know the function space of the ground truth. We know the RKHS norm...
# Is there an RKHS norm that we need such that these bounds may become tighter?
# But then there is the RKHS frequentist Bayesian question again right...
# It has to be tighter because we have more structural assumptions on the function!

# BO with general noise
import warnings
import torch
import numpy as np
import gpytorch
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import copy
import tikzplotlib
import os

random_seed_number = 10
np.random.seed(random_seed_number)
torch.manual_seed(random_seed_number)



# Add the relative path to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print to verify
print("Changed working directory to:", os.getcwd())

plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams.update({'font.size': 30})


from pacsbo.pacsbo_main import compute_X_plot, initial_safe_samples, PACSBO
from enveloped import create_random_functions, ground_truth



def acquisition_function(cube):
    # if sum(torch.logical_or(cube.M, cube.G)) != 0:
    #     max_indices = torch.nonzero(cube.ucb[torch.logical_or(cube.M, cube.G)] == torch.max(cube.ucb[torch.logical_or(cube.M, cube.G)]), as_tuple=True)[0]
    #     random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
    #     x_new = cube.discr_domain[torch.logical_or(cube.M, cube.G)][random_max_index, :]
    # else:
    #      x_new = None
    # return x_new  # we can return cube, no parallelization needed
    interesting_points = torch.logical_or(cube.M, cube.G)
    if sum(interesting_points) != 0:
        x_new = cube.discr_domain[interesting_points][torch.argmax(cube.uncertainty[interesting_points])]
    else: 
        x_new = None
    return x_new  # we can return cube, no parallelization needed





def update_model(cube, lb, ub):
    # cube.compute_model(gpr=GPRegressionModel)  # compute confidence intervals?
    # cube.compute_mean_var()
    # We do not need any of the above! 
    # We just need to update the confidence intervals, which are given by the scenario approach.
    cube.lcb = lb
    cube.ucb = ub
    cube.uncertainty = ub - lb
    # cube.compute_confidence_intervals() old script!
    


def compute_sets(cube):
    cube.compute_safe_set()
    cube.maximizer_routine()
    cube.expander_routine()
    



def noise_scenario(kappa, gamma_confidence, t, R_scenario, x):  # just Gaussian now
    pi_t = np.pi**2*t**2/6
    kappa_t = kappa/pi_t
    # How much N do we need?
    # Some log thing here which Lemma 2; scenario approach, we can get N directly! Then take the max.
    N_scenario = int(np.log(kappa_t)/np.log(1-gamma_confidence)) + 1  # log base change, add 1
    # epsilon_t = np.abs(np.random.normal(loc=0.0, scale=R_scenario, size=N_scenario))  # taking abs value okay?
    # epsilon_t = np.abs(-R_scenario + 2*R_scenario*np.abs(np.random.uniform(size=N_scenario)))
    # epsilon_t = 1/10*np.random.standard_t(df=10, size=N_scenario)
    epsilon_t = 1/5*np.random.standard_t(df=10, size=N_scenario)*np.abs(x.item())
    bar_epsilon_t = max(epsilon_t)
    '''
    t = np.array(range(1,1000))
    pi_t = np.pi**2*t**2/6
    kappa_1 = 1e-6
    kappa_t_1 = kappa_1/pi_t
    N_scenario_11 = np.log(kappa_t_1)/np.log(1-0.01) + 1
    N_scenario_12 = np.log(kappa_t_1)/np.log(1-0.001) + 1
    kappa_2 = 1e-3
    kappa_t_2 = kappa_2/pi_t
    N_scenario_21 = np.log(kappa_t_2)/np.log(1-0.01) + 1
    N_scenario_22 = np.log(kappa_t_2)/np.log(1-0.001) + 1
    plt.figure()
    plt.plot(t, N_scenario_11)
    plt.plot(t, N_scenario_12)
    plt.plot(t, N_scenario_21)
    plt.plot(t, N_scenario_22)
    plt.yscale("log")
    tikzplotlib.save("N_development.tex")
    '''
    return torch.tensor(bar_epsilon_t, dtype=torch.float32)


def update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube, x):
    bar_epsilon_t = noise_scenario(kappa_confidence, gamma_confidence, t, R, x)
    cube.bar_epsilon_tensor = torch.cat((cube.bar_epsilon_tensor, bar_epsilon_t.unsqueeze(0)))
    return cube





def plot(cube, gt, tensor_random_functions, support, plot_support=True, save=False, title=""):
    safety_threshold = cube.safety_threshold
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_sample = cube.y_sample.detach().numpy()
    ucb = cube.ucb.detach().numpy()
    lcb = cube.lcb.detach().numpy()
    # Ensure gt.fX is a NumPy array for plotting
    if hasattr(gt.fX, "detach"):
        gt_fX = gt.fX.detach().numpy()
    else:
        gt_fX = np.asarray(gt.fX)


    if not save:
        plt.figure()
        if plot_support:
            for i in range(1, len(support)):
                plt.plot(X_plot, tensor_random_functions[support[i], :].detach().numpy(), 'gray', alpha=0.1)
        plt.plot(X_plot, gt_fX, color='blue')
        plt.scatter(X_sample[1:], Y_sample[1:], color='black')
        # plt.scatter(X_plot[np.argmax(ucb[cube.S])], np.max(cube.ucb[cube.S]), color='black'); plot maximum
        failures = gt.f(cube.x_sample) < safety_threshold
        if sum(failures) > 0:
            plt.plot(X_sample[failures], Y_sample[failures], 'xr', markersize=25, markeredgewidth=3)
        plt.plot(X_plot, [safety_threshold] * len(X_plot), '-r')
        plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
        plt.fill_between(X_plot.flatten(), lcb, ucb, color="gray", alpha=0.2)
        plt.plot(X_plot, lcb, 'gray', label="lb, ub")
        plt.plot(X_plot, ucb, 'gray')
        plt.xlabel('$a$')
        plt.ylabel('$y$')
        plt.show()
    elif save:
        step = 5
        plt.figure()
        if plot_support:
                for i in range(1, len(support)):
                    y = tensor_random_functions[support[i], :].detach().numpy()
                    plt.plot(X_plot[::step], y[::step], color='gray', alpha=0.1)
        plt.plot(X_plot[::step], gt_fX[::step], color='blue')
        plt.scatter(X_sample[1:], Y_sample[1:], color='black')
        failures = gt.f(cube.x_sample) < safety_threshold
        if sum(failures) > 0:
            plt.plot(X_sample[failures], Y_sample[failures], 'xr', markersize=25, markeredgewidth=3)
        plt.plot(X_plot[::step], np.full(len(X_plot[::step]), safety_threshold), '-r')
        plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
        plt.plot(X_plot[::step], lcb[::step], 'gray', label="lb, ub")
        plt.plot(X_plot[::step], ucb[::step], 'gray')

        plt.fill_between(X_plot[::step].flatten(), lcb[::step], ucb[::step], color="gray", alpha=0.2)
        tikzplotlib.save(title)



if __name__ == '__main__':
        introductory_example = False  # Section 4; toy example to compare classic scenario theory with wait and judge
        noise_type = "uniform"  # Student-t, Gaussian, uniform, heteroscedastic
        iterations = 30
        eta = 1e-4
        R = 5e-3
        delta_confidence = 0.1      
        bar_epsilon_tensor = torch.tensor([])
        coeff_distribution = "Gaussian"
        basis_functions = "BSpline"
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = 0.1

        kappa_confidence = 1e-3   # 0.01  
        gamma_confidence = 0.1  
        exploration_threshold = 0.1  #  0.2
        lengthscale = 0.1
        Gaussian_std = 0.01
        RKHS_norm = 1
        beta_list = []
        beta_list_ours = []

        X_plot = compute_X_plot(n_dimensions=1, points_per_axis=1000)
        number_of_KL_terms = len(X_plot)
        gt = ground_truth(coeff_distribution, Gaussian_std=Gaussian_std, X_plot=X_plot, basis_functions=basis_functions, kernel=kernel, lengthscale=lengthscale, noise_type="uniform", R=R)   
        safety_threshold = torch.quantile(gt.fX, 0.4).item() 
        X_sample_init, Y_sample_init = initial_safe_samples(gt, num_safe_points=1, X_plot=X_plot, R=R, safety_threshold=safety_threshold)
        X_sample = X_sample_init.clone()
        Y_sample = Y_sample_init.clone()


        if not introductory_example:

            cube = PACSBO(delta_confidence=delta_confidence, eta=eta, R=R, X_plot=X_plot, X_sample=X_sample,
                        Y_sample=Y_sample, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold,
                        RKHS_norm=RKHS_norm, lengthscale=lengthscale)
            list_cubes = []
            for t in tqdm(range(1, iterations + 1)):
                # list_cubes.append(copy.deepcopy(cube))  # store the cube before updating it
                # cube = update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube)
                lb, ub, argmin, argmax, tensor_random_functions, support = create_random_functions(
                    coeff_distribution, Gaussian_std, X_plot, basis_functions, kernel, X_sample,
                    Y_sample, gamma_confidence, kappa_confidence, wj=True, noise_type=noise_type, R=R, t=t)

                update_model(cube, lb, ub)
                compute_sets(cube)
                if t == 1:
                    plot(cube, gt, tensor_random_functions, support, plot_support=True, save=True, title="numerical_example_beginning.tex") 
                x_new = acquisition_function(cube=cube)
                if x_new != None:
                    y_new = torch.tensor(gt.conduct_experiment(x=x_new), dtype=torch.float32)
                else:
                    # print('Done!')
                    break
                Y_sample = torch.cat((Y_sample, y_new), dim=0)
                X_sample = torch.cat((X_sample, x_new.unsqueeze(0)), dim=0)
                cube.x_sample = X_sample
                cube.y_sample = Y_sample
                list_cubes.append(cube)
            plot(cube, gt, tensor_random_functions, support, plot_support=True, save=False, title="numerical_example_end.tex") 

        elif introductory_example:
                lb, ub, argmin, argmax, tensor_random_functions, support = create_random_functions(
                    coeff_distribution, Gaussian_std, X_plot, basis_functions, kernel, X_sample,
                    Y_sample, gamma_confidence, kappa_confidence, wj=True, noise_type=noise_type, R=R, t=1)
                lb_scenario, ub_scenario, argmin_scenario, argmax_scenario, tensor_random_functions_scenario, support_scenario = create_random_functions(
                coeff_distribution, Gaussian_std, X_plot, basis_functions, kernel, X_sample, Y_sample, gamma_confidence, kappa_confidence, wj=False, noise_type=noise_type, R=R, t=1)                

                # Plots
                step = 5
                # First: wait and judge plot
                plt.figure()
                for i in range(len(support)):
                    plt.plot(X_plot[::step], tensor_random_functions[support[i], :].detach().numpy()[::step], 'gray', alpha=0.1)
                plt.fill_between(X_plot.flatten().detach().numpy()[::step], lb.detach().numpy()[::step], ub.detach().numpy()[::step], color="gray", alpha=0.2)
                plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
                plt.plot(X_plot[::step], lb.detach().numpy()[::step], 'gray', label="lb, ub")
                plt.plot(X_plot[::step], ub.detach().numpy()[::step], 'gray')
                plt.plot(X_plot[::step], gt.fX.detach().numpy()[::step], 'blue', label="Truth")
                plt.title("Wait and judge bounds")

                # Second: scenario approach plot with plotting supports
                plt.figure()
                for i in range(len(support_scenario)):
                    plt.plot(X_plot[::step], tensor_random_functions_scenario[support_scenario[i], :].detach().numpy()[::step], 'gray', alpha=0.1)
                plt.fill_between(X_plot.flatten().detach().numpy()[::step], lb_scenario.detach().numpy()[::step], ub_scenario.detach().numpy()[::step], color="gray", alpha=0.2)
                plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
                plt.plot(X_plot[::step], lb_scenario.detach().numpy()[::step], 'gray', label="lb, ub")
                plt.plot(X_plot[::step], ub_scenario.detach().numpy()[::step], 'gray')
                plt.plot(X_plot[::step], gt.fX.detach().numpy()[::step], 'blue', label="Truth")
                plt.title("Scneario approach (only support)")                
                
                plt.figure()
                for i in range(tensor_random_functions_scenario.shape[0]):
                    plt.plot(X_plot[::step], tensor_random_functions_scenario[i, :].detach().numpy()[::step], 'gray', alpha=0.1)
                plt.fill_between(X_plot.flatten().detach().numpy()[::step], lb_scenario.detach().numpy()[::step], ub_scenario.detach().numpy()[::step], color="gray", alpha=0.2)
                plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
                plt.plot(X_plot[::step], lb_scenario.detach().numpy()[::step], 'gray', label="lb, ub")
                plt.plot(X_plot[::step], ub_scenario.detach().numpy()[::step], 'gray')
                plt.plot(X_plot[::step], gt.fX.detach().numpy()[::step], 'blue', label="Truth")
                plt.title("Scneario approach (all scenarios)")                


                # Compare the bounds
                plt.figure()
                plt.plot(X_plot[::step], gt.fX.detach().numpy()[::step], 'blue', label="Truth")
                plt.scatter(X_sample.detach().numpy(), Y_sample.detach().numpy(), color='k', s=100, label='Samples')
                plt.plot(X_plot[::step], lb_scenario.detach().numpy()[::step], 'gray', label="lb, ub scenario")
                plt.plot(X_plot[::step], ub_scenario.detach().numpy()[::step], 'gray')
                plt.plot(X_plot[::step], lb.detach().numpy()[::step], 'magenta', label="lb, ub wait and judge")
                plt.plot(X_plot[::step], ub.detach().numpy()[::step], 'magenta')
                plt.fill_between(X_plot.flatten().detach().numpy()[::step], lb_scenario.detach().numpy()[::step], ub_scenario.detach().numpy()[::step], color="gray", alpha=0.1)
                plt.fill_between(X_plot.flatten().detach().numpy()[::step], lb.detach().numpy()[::step], ub.detach().numpy()[::step], color="magenta", alpha=0.2)
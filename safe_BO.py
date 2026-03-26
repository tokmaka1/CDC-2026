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
from plot_furuta import plot_constraints, plot_rewards, plot_ucb, plot_lcb
import sys 
import dill

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
from furuta import ground_truth_Furuta

# Uncomment the following and clone repo https://git.rwth-aachen.de/quanser-vision/vision-based-furuta-pendulum to conduct Furuta pendulum experiments

repo_path = os.path.join(script_dir, "vision-based-furuta-pendulum")
print("Repo exists:", os.path.exists(repo_path))

sys.path.insert(0, repo_path)
# from vision_based_furuta_pendulum_master.gym_brt.envs import QubeBalanceEnv, QubeSwingupEnv
# from vision_based_furuta_pendulum_master.gym_brt.control.control import QubeHoldControl, QubeFlipUpControl

from gym_brt.envs import QubeBalanceEnv, QubeSwingupEnv
from gym_brt.control.control import QubeHoldControl, QubeFlipUpControl
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from IPython import embed as IPS





def acquisition_function(cube):
    interesting_points = torch.logical_or(cube.M, cube.G)
    if sum(interesting_points) != 0:
        x_new = cube.discr_domain[interesting_points][torch.argmax(cube.uncertainty[interesting_points])]
    else: 
        x_new = None
    return x_new  # we can return cube, no parallelization needed





def update_model(cube, lb_rew, ub_rew, lb_con_1, ub_con_1, lb_con_2, ub_con_2):
    # cube.compute_model(gpr=GPRegressionModel)  # compute confidence intervals?
    # cube.compute_mean_var()
    # We do not need any of the above! 
    # We just need to update the confidence intervals, which are given by the scenario approach.
        # Reward 
    # self.lcb_rew = self.mean_rew - self.beta_rew*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
    # self.ucb_rew = self.mean_rew + self.beta_rew*torch.sqrt(self.var)
    # self.lcb_rew = torch.max(self.lcb_rew_old, self.lcb_rew)  # pointwise!
    # self.ucb_rew = torch.min(self.ucb_rew_old, self.ucb_rew)
    # self.lcb_rew_old = self.lcb_rew.clone().detach()
    # self.ucb_rew_old = self.ucb_rew.clone().detach()

    # # Constraint
    # self.lcb_con = self.mean_con - self.beta_con*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
    # self.ucb_con = self.mean_con + self.beta_con*torch.sqrt(self.var)
    # self.lcb_con = torch.max(self.lcb_con_old, self.lcb_con)  # pointwise!
    # self.ucb_con = torch.min(self.ucb_con_old, self.ucb_con)
    # self.lcb_con_old = self.lcb_con.clone().detach()
    # self.ucb_con_old = self.ucb_con.clone().detach()




    cube.lcb_rew = lb_rew
    cube.ucb_rew = ub_rew
    cube.uncertainty_rew = ub_rew - lb_rew
    cube.lcb_con_1 = lb_con_1
    cube.ucb_con_1 = ub_con_1
    cube.uncertainty_con_1 = ub_con_1 - lb_con_1
    cube.lcb_con_2 = lb_con_2
    cube.ucb_con_2 = ub_con_2
    cube.uncertainty_con_2 = ub_con_2 - lb_con_2
    cube.uncertainty = torch.max(cube.uncertainty_rew, torch.max(cube.uncertainty_con_1, cube.uncertainty_con_2))


def compute_sets(cube):
    cube.compute_safe_set()
    cube.maximizer_routine()
    cube.expander_routine()
    



def plot(cube, gt, tensor_random_functions, support, plot_support=True, save=False, title="", t=None):
    safety_threshold = cube.safety_threshold
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_sample = cube.y_sample.detach().numpy()
    ucb = cube.ucb.detach().numpy()
    lcb = cube.lcb.detach().numpy()
    # Support both torch and numpy storage for the safe set indices/mask.
    if hasattr(cube.S, "detach"):
        safe_set = cube.S.detach().cpu().numpy()
    else:
        safe_set = np.asarray(cube.S)

    safe_set = np.asarray(safe_set)
    if safe_set.dtype == np.bool_:
        safe_idx = np.where(safe_set)[0]
    else:
        safe_idx = safe_set.astype(int).reshape(-1)

    if safe_idx.size > 0:
        best_safe_idx = safe_idx[np.argmax(lcb[safe_idx])]
        best_safe_x = X_plot[best_safe_idx]
        best_safe_x_tensor = cube.discr_domain[best_safe_idx].unsqueeze(0)
        best_safe_gt = gt.f(best_safe_x_tensor).detach().cpu().numpy().reshape(-1)[0]
    else:
        best_safe_x = None
        best_safe_gt = None

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
        failures = gt.f(cube.x_sample) < safety_threshold
        if sum(failures) > 0:
            plt.plot(X_sample[failures], Y_sample[failures], 'xr', markersize=25, markeredgewidth=3)
        plt.plot(X_plot, [safety_threshold] * len(X_plot), '-r')
        plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
        plt.fill_between(X_plot.flatten(), lcb, ucb, color="gray", alpha=0.2)
        plt.plot(X_plot, lcb, 'gray', label="lb, ub")
        plt.plot(X_plot, ucb, 'gray')
        if best_safe_x is not None and t != 1:
            plt.plot(best_safe_x, best_safe_gt, marker='*', color='cyan', markersize=18, linestyle='None')
        plt.xlabel('$a$')
        plt.ylabel('$y$')
        plt.show()
    elif save:
        step = 8  # Plot every 8th point to reduce file size; adjust as needed
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
        if best_safe_x is not None and t != 1:
            plt.plot(best_safe_x, best_safe_gt, marker='*', color='cyan', markersize=18, linestyle='None')

        plt.fill_between(X_plot[::step].flatten(), lcb[::step], ucb[::step], color="gray", alpha=0.2)
        tikzplotlib.save(title)



if __name__ == '__main__':
        try_best_param = True  # Section 4; toy example to compare classic scenario theory with wait and judge
        noise_type = "uniform"  # Student-t, Gaussian, uniform, heteroscedastic
        iterations = 20  # 20 iterations should be enough
        R = 5e-2  # 5e-3  # 1e-2  #  # 5e-3  # for noise in the observations
        bar_epsilon_tensor = torch.tensor([])
        coeff_distribution = "Gaussian"  # Options: Gaussian 
        kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        kernel.lengthscale = 0.01  # Increasing smoothness

        kappa_confidence = 1e-3    
        gamma_confidence = 0.1  
        Gaussian_std = 1e-1  # 1e-1 for 1 sample
        X_plot = compute_X_plot(n_dimensions=2, points_per_axis=101)
        gt_furuta = ground_truth_Furuta(safety_threshold=0, use_simulator=False)  # set to False to run on the real system; make sure to have the vision-based-furuta-pendulum repo in place and adjust the import statements at the top of this file accordingly.
        X_sample_init = torch.tensor([[0.23, 0.40]])  # , [0.9, 0.32]])  # start from here, safe but low-performing      
        # Maximizer we found:

        # X_sample_init = torch.tensor([[0.19, 0.35]])
        if try_best_param:
            # Starting parameter first
            X_sample_init = torch.tensor([[0.23, 0.40]]) 
            _, _, _, alpha_list_init, theta_list_init = gt_furuta.conduct_experiment(X_sample_init)
            X_sample_best = torch.tensor([[0.19, 0.35]])
            _, _, _, alpha_list_best, theta_list_best = gt_furuta.conduct_experiment(X_sample_best)
            dict_angles = {
                "alpha_init": alpha_list_init,
                "theta_init": theta_list_init,
                "alpha_best": alpha_list_best,
                "theta_best": theta_list_best
            }
            dill.dump(dict_angles, open("angles.pkl", "wb"))
            raise Exception("Done with angles")
        X_sample = X_sample_init.clone()
        Y_sample_rew_init, Y_sample_con_1_init, Y_sample_con_2_init, _, _ = gt_furuta.conduct_experiment(X_sample_init)
        list_lb_rew = []  # putting max S
        list_lb_con_1 = []
        list_lb_con_2 = []
        list_ub_rew = []
        list_ub_con_1 = []
        list_ub_con_2 = []
        Y_sample_reward = Y_sample_rew_init.clone()
        Y_sample_constraint_1 = Y_sample_con_1_init.clone()
        Y_sample_constraint_2 = Y_sample_con_2_init.clone()
        cube = PACSBO(X_plot, X_sample, Y_sample_reward, Y_sample_constraint_1, Y_sample_constraint_2, safety_threshold=0)
        for t in tqdm(range(1, iterations + 1)):
            # Constraints
            lb_con_1, ub_con_1, argmin_con_1, argmax_con_1, tensor_random_functions_con_1, support_con_1 = create_random_functions(
                coeff_distribution, Gaussian_std, X_plot, kernel, X_sample, Y_sample_constraint_1, gamma_confidence, kappa_confidence, wj=True, noise_type=noise_type, R=R, t=t)
            
            lb_con_2, ub_con_2, argmin_con_2, argmax_con_2, tensor_random_functions_con_2, support_con_2 = create_random_functions(
                coeff_distribution, Gaussian_std, X_plot, kernel, X_sample, Y_sample_constraint_2, gamma_confidence, kappa_confidence, wj=True, noise_type=noise_type, R=R, t=t)
            # Rewards    
            lb_rew, ub_rew, argmin_rew, argmax_rew, tensor_random_functions_rew, support_rew = create_random_functions(
                coeff_distribution, Gaussian_std, X_plot, kernel, X_sample, Y_sample_reward, gamma_confidence, kappa_confidence, wj=True, noise_type=noise_type, R=R, t=t)


            update_model(cube, lb_rew, ub_rew, lb_con_1, ub_con_1, lb_con_2, ub_con_2)
            compute_sets(cube)
            # Update lists
            list_lb_rew.append(cube.lcb_rew[cube.S].max().item())  # max over the safe set
            list_lb_con_1.append(cube.lcb_con_1[cube.S].max().item())  # max over the safe set
            list_ub_rew.append(cube.ucb_rew[cube.S].max().item())  # max over the safe set
            list_ub_con_1.append(cube.ucb_con_1[cube.S].max().item())  # max over the safe set
            if t==iterations:
                break  # no need to do the last. Have 30 in total.
            x_new = acquisition_function(cube=cube)
            if x_new != None:
                # Stop if the acquisition proposes a point that has already been sampled.
                already_sampled = torch.any(torch.all(torch.isclose(X_sample, x_new.unsqueeze(0)), dim=1))
                if already_sampled:
                    print('Stopping: acquisition proposed an already sampled point.')
                    break
                y_new_reward, y_new_constraint_1, y_new_constraint_2, _, _ = gt_furuta.conduct_experiment(x_new)
            else:
                print('Done!')
                break
            Y_sample_reward = torch.cat((Y_sample_reward, y_new_reward), dim=0)
            Y_sample_constraint_1 = torch.cat((Y_sample_constraint_1, y_new_constraint_1), dim=0)
            Y_sample_constraint_2 = torch.cat((Y_sample_constraint_2, y_new_constraint_2), dim=0)
            X_sample = torch.cat((X_sample, x_new.unsqueeze(0)), dim=0)
            cube.x_sample = X_sample
            cube.Y_sample_reward = Y_sample_reward
            cube.Y_sample_constraint_1 = Y_sample_constraint_1
            cube.Y_sample_constraint_2 = Y_sample_constraint_2
            if t==1:
                cube.lcb_rew_init = lb_rew
                cube.ucb_rew_init = ub_rew
                cube.lcb_con_1_init = lb_con_1
                cube.ucb_con_1_init = ub_con_1
                cube.lcb_con_2_init = lb_con_2
                cube.ucb_con_2_init = ub_con_2

        cube.list_lb_rew = list_lb_rew
        cube.list_lb_con_1 = list_lb_con_1
        cube.list_lb_con_2 = list_lb_con_2
        cube.list_ub_rew = list_ub_rew
        cube.list_ub_con_1 = list_ub_con_1
        cube.list_ub_con_2 = list_ub_con_2
        print(123)
        # dill.dump(cube, open("cube.pkl", "wb"))
        # Plot alpha and theta





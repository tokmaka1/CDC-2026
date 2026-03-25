# Plotting Furuta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation




def plot_constraints(cube, Y_sample_constraints):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_constraints = Y_sample_constraints.detach().numpy()
    plt.figure()
    plt.plot(range(len(Y_constraints)), Y_constraints)
    plt.plot(range(len(Y_constraints)), cube.safety_threshold * np.ones_like(Y_constraints), 'r--')


def plot_rewards(cube, Y_sample_reward):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_rewards = Y_sample_reward.detach().numpy()
    plt.figure()
    plt.plot(range(len(Y_rewards)), Y_rewards)
    plt.plot(range(len(Y_rewards)), cube.safety_threshold * np.ones_like(Y_rewards), 'r--')


def plot_ucb(cube):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    ub_reward = cube.ucb_rew.detach().numpy()
    plt.figure()
    sc = plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=ub_reward,
        cmap='plasma'
    )
    plt.colorbar(sc)  # Add a colorbar to show the mapping
    plt.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    plt.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)


def plot_lcb(cube):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    lb_reward = cube.lcb_rew.detach().numpy()
    plt.figure()
    sc = plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=lb_reward,
        cmap='plasma'
    )
    plt.colorbar(sc)  # Add a colorbar to show the mapping
    plt.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    plt.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)


def plot_constraint_surface(X_plot, constraint_list, title='Constraint surface'):
    """Plot a 2D surface-style map: x=X[:,0], y=X[:,1], color=constraint."""
    X_plot = np.asarray(X_plot)
    constraint_array = np.asarray(constraint_list).reshape(-1)

    if X_plot.ndim != 2 or X_plot.shape[1] != 2:
        raise ValueError('X_plot must have shape (n_points, 2).')
    if X_plot.shape[0] != constraint_array.shape[0]:
        raise ValueError('X_plot and constraint_list must have the same length.')

    x = X_plot[:, 0]
    y = X_plot[:, 1]

    plt.figure()
    tri = Triangulation(x, y)
    surface = plt.tripcolor(tri, constraint_array, shading='gouraud', cmap='viridis')
    plt.colorbar(surface, label='Constraint value')
    plt.scatter(x, y, s=6, c='k', alpha=0.2)
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title("Constraint value")
    plt.tight_layout()


def plot_reward_surface(X_plot, reward_list, title='Reward surface'):
    """Plot a 2D surface-style map: x=X[:,0], y=X[:,1], color=reward."""
    X_plot = np.asarray(X_plot)
    reward_array = np.asarray(reward_list).reshape(-1)

    if X_plot.ndim != 2 or X_plot.shape[1] != 2:
        raise ValueError('X_plot must have shape (n_points, 2).')
    if X_plot.shape[0] != reward_array.shape[0]:
        raise ValueError('X_plot and reward_list must have the same length.')

    x = X_plot[:, 0]
    y = X_plot[:, 1]

    plt.figure()
    tri = Triangulation(x, y)
    surface = plt.tripcolor(tri, reward_array, shading='gouraud', cmap='plasma')
    plt.colorbar(surface, label='Reward value')
    plt.scatter(x, y, s=6, c='k', alpha=0.2)
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title("Reward value")
    plt.tight_layout()




# Plotting Furuta

import matplotlib.pyplot as plt
import numpy as np




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




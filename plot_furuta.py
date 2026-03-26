# Plotting Furuta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
import dill
import torch
from scipy.spatial import ConvexHull, QhullError
import tikzplotlib


def plot_constraints(cube):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_constraints = cube.Y_sample_constraint.detach().numpy()
    plt.figure()
    plt.plot(range(len(Y_constraints)), Y_constraints)
    plt.plot(range(len(Y_constraints)), cube.safety_threshold * np.ones_like(Y_constraints), 'r--')


def plot_rewards(cube):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_rewards = cube.Y_sample_reward.detach().numpy()
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


def plot_lcb_constraints(cube):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    lb_con = cube.lcb_con.detach().numpy()
    plt.figure()
    sc = plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=lb_con,
        cmap='plasma'
    )
    plt.colorbar(sc)  # Add a colorbar to show the mapping
    plt.scatter(cube.discr_domain[cube.S][:, 0], cube.discr_domain[cube.S][:, 1], color='white')  #, label='Safe set')
    plt.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    # plt.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)

def plot_just_reward(cube):
    X_sample = cube.x_sample.detach().numpy()
    Y_rewards = cube.Y_sample_reward.detach().numpy()
    Y_rewards_max = []
    for y in Y_rewards:
        if len(Y_rewards_max) == 0:
            Y_rewards_max.append(y)
        else:
            Y_rewards_max.append(max(y, Y_rewards_max[-1]))
    plt.figure()
    plt.plot(range(len(Y_rewards)), Y_rewards**10, '*')  # , '*')
    plt.plot(range(len(Y_rewards_max)), np.array(Y_rewards_max)**10, 'b-')  # , '*')
    plt.ylim(0, 1)

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





def plot_uncertainty_3d(cube):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # --- data ---
    X_plot = cube.discr_domain.detach().cpu().numpy()
    X_sample = cube.x_sample.detach().cpu().numpy()

    # you need these in cube
    u = cube.ucb_rew.detach().cpu().numpy()   # or u_t
    l = cube.lcb_rew.detach().cpu().numpy()   # or l_t

    W = u - l  # uncertainty

    x = X_plot[:, 0]
    y = X_plot[:, 1]
    z = W

    # --- figure ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # surface (triangular, works with arbitrary discretization)
    tri_surf = ax.plot_trisurf(
        x, y, z,
        cmap='viridis',
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    # --- samples ---
    # find nearest grid points (same logic as before)
    dist = np.linalg.norm(
        X_plot[:, None, :] - X_sample[None, :, :],
        axis=2
    )
    nearest_idx = np.argmin(dist, axis=0)

    ax.scatter(
        X_plot[nearest_idx, 0],
        X_plot[nearest_idx, 1],
        z[nearest_idx],
        color='red',
        s=50
    )

    # --- colorbar ---
    fig.colorbar(tri_surf, ax=ax, shrink=0.6, label='u(x) - l(x)')

    # --- labels ---
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Uncertainty width')

    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()
    plt.savefig("uncertainty_surface.png", dpi=300)


def plot_intro_init(cube):
    lcb = cube.lcb_rew_init.detach().numpy()
    X_sample = cube.x_sample[0].detach().numpy()
    X_plot = cube.discr_domain.detach().numpy()
    S = torch.min(cube.lcb_con_1_init, cube.lcb_con_2_init) > 0
    safe_points = cube.discr_domain[S].detach().numpy()
    plt.figure()
    tri = Triangulation(X_plot[:, 0], X_plot[:, 1])
    surface = plt.tripcolor(tri, lcb, shading='gouraud', cmap='plasma')
    plt.colorbar(surface)  # Add a colorbar to show the mapping
    if safe_points.shape[0] >= 3:
        try:
            hull = ConvexHull(safe_points)
            hull_cycle = np.append(hull.vertices, hull.vertices[0])
            plt.plot(safe_points[hull_cycle, 0], safe_points[hull_cycle, 1], color='green', linewidth=2, label='Safe set hull')
            plt.fill(safe_points[hull.vertices, 0], safe_points[hull.vertices, 1], color='green', alpha=0.10)
        except QhullError:
            pass
    elif safe_points.shape[0] == 2:
        plt.plot(safe_points[:, 0], safe_points[:, 1], color='green', linewidth=2, label='Safe set hull')
    elif safe_points.shape[0] == 1:
        circle = plt.Circle((safe_points[0, 0], safe_points[0, 1]), 0.01, color='green', fill=False, linewidth=2, label='Safe set hull')
        plt.gca().add_patch(circle)
    plt.scatter(X_sample[0], X_sample[1], color='red', label='Initial sample')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Initial safe set and sample')
    plt.xlim(0.15, 0.35)
    plt.ylim(0.25, 0.60)
    plt.legend()
    plt.show()


def plot_intro_end(cube):
    lcb = cube.lcb_rew.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    X_plot = cube.discr_domain.detach().numpy()
    S = torch.min(cube.lcb_con_1, cube.lcb_con_2) > 0
    safe_points = cube.discr_domain[S].detach().numpy()
    best_parameter = cube.discr_domain[cube.S][torch.argmax(cube.lcb_rew[cube.S])].detach().numpy()
    print(f"Best parameter found: {best_parameter}")
    plt.figure()
    tri = Triangulation(X_plot[:, 0], X_plot[:, 1])
    surface = plt.tripcolor(tri, lcb, shading='gouraud', cmap='plasma')
    plt.colorbar(surface)  # Add a colorbar to show the mapping
    if safe_points.shape[0] >= 3:
        try:
            hull = ConvexHull(safe_points)
            hull_cycle = np.append(hull.vertices, hull.vertices[0])
            plt.plot(safe_points[hull_cycle, 0], safe_points[hull_cycle, 1], color='green', linewidth=2, label='Safe set hull')
            plt.fill(safe_points[hull.vertices, 0], safe_points[hull.vertices, 1], color='green', alpha=0.10)
        except QhullError:
            pass
    elif safe_points.shape[0] == 2:
        plt.plot(safe_points[:, 0], safe_points[:, 1], color='green', linewidth=2, label='Safe set hull')
    elif safe_points.shape[0] == 1:
        circle = plt.Circle((safe_points[0, 0], safe_points[0, 1]), 0.01, color='green', fill=False, linewidth=2, label='Safe set hull')
        plt.gca().add_patch(circle)
    plt.scatter(X_sample[1:, 0], X_sample[1:, 1], color='black', label='All samples')
    plt.scatter(
        X_sample[0, 0], X_sample[0, 1],
        color='magenta',
        marker='D',   # diamond
        s=70,         # optional size
        label='Initial sample'
    )
    plt.scatter(
        best_parameter[0], best_parameter[1],
        color='cyan',
        marker='s',   # Square
        s=70,         # optional size
        label='Best parameter'
    )
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Initial safe set and sample')
    plt.xlim(0.15, 0.35)
    plt.ylim(0.25, 0.60)
    plt.legend()
    plt.show()




def plot_max_development_twin(cube, save=False):
    
    # Left y-axis: best parameter reward
    X_sample = cube.x_sample.detach().numpy()
    Y_rewards = cube.Y_sample_reward.detach().numpy()
    Y_rewards_max = []
    for y in Y_rewards:
        if len(Y_rewards_max) == 0:
            Y_rewards_max.append(y**10)
        else:
            Y_rewards_max.append(max(y**10, Y_rewards_max[-1]))

    
    fig, ax1 = plt.subplots()

    ax1.plot(range(len(Y_rewards)), np.array(Y_rewards)**10, '*', color="tab:blue")
    ax1.plot(range(len(Y_rewards)), Y_rewards_max, '-', color="tab:blue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best parameter reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)
    # Right y-axis: constraint values
    ax2 = ax1.twinx()
    ax2.plot(range(len(cube.Y_sample_constraint_1)), torch.min(cube.Y_sample_constraint_1, cube.Y_sample_constraint_2).detach().numpy(), '*', color="tab:red", label="Constraint value")
    # ax2.plot(range(len(cube.Y_sample_constraint_2)), cube.Y_sample_constraint_2.detach().numpy(), '*', color="tab:red", label="Constraint value")
    ax2.plot(range(len(cube.Y_sample_constraint_1)), np.zeros(len(cube.Y_sample_constraint_1)), '-', color="tab:red")
    ax2.set_ylabel("Constraint value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.1, 1.5)
    # Optional legend
    # fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.show()
    if save:
        tikzplotlib.save("reward_constraint_development.tex")


def plot_angles(angles, save=False):
    theta_init = angles["theta_init"]
    alpha_init = angles["alpha_init"]
    theta_best = angles["theta_best"]
    alpha_best = angles["alpha_best"]
    plt.figure()
    # plt.plot(theta_init, label='Theta (initial)')
    plt.plot(alpha_init, label='Alpha (initial)')
    # plt.plot(theta_best, label='Theta (best)')
    plt.plot(alpha_best, label='Alpha (best)')
    plt.xlabel('Time step')
    plt.ylabel('Angle (radians)')
    plt.title('Pendulum angles over time')
    # plt.legend()
    if save:
        tikzplotlib.save("angles_over_time.tex")
    plt.show()




if __name__ == '__main__':
    cube = dill.load(open("cube_final.pkl", "rb"))
    angles = dill.load(open("angles.pkl", "rb"))
    print(123)
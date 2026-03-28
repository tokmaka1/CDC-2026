# Plotting Furuta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
import dill
import torch
from scipy.spatial import ConvexHull, QhullError
import tikzplotlib
from scipy.ndimage import gaussian_filter



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


def plot_intro_blue_manual_tikz(cube, tex_path="exploration_beginning_manual.tex", bg_path="exploration_beginning_manual_bg.png"):
    # --- extract ---
    lcb = cube.lcb_rew_init.detach().cpu().numpy()
    X_plot = cube.discr_domain.detach().cpu().numpy()

    # --- triangulation ---
    tri = Triangulation(X_plot[:, 0], X_plot[:, 1])

    # --- clip (flatten background) ---
    z_min = np.percentile(lcb, 85)
    lcb_clipped = np.clip(lcb, z_min, lcb.max())
    norm = Normalize(vmin=z_min, vmax=lcb.max())

    # --- safe set ---
    S = (torch.min(cube.lcb_con_1_init, cube.lcb_con_2_init) > 0)
    safe_points = cube.discr_domain[S].detach().cpu().numpy()

    levels = np.linspace(z_min, lcb.max(), 12)
    x_limits = (0.125, 0.30)
    y_limits = (0.275, 0.45)

    # Save contour+colorbar as raster; overlay hull/markers in TikZ.
    fig_bg, ax_bg = plt.subplots(figsize=(6, 4))
    cf_bg = ax_bg.tricontourf(tri, lcb_clipped, levels=levels, cmap='coolwarm', norm=norm)
    ax_bg.tricontour(tri, lcb_clipped, levels=levels, colors='black', linewidths=0.5)
    cbar_bg = fig_bg.colorbar(cf_bg)
    cbar_bg.set_ticks([z_min, lcb.max()])
    cbar_bg.set_ticklabels(['Low', 'High'])
    ax_bg.set_xlabel("Parameter 1")
    ax_bg.set_ylabel("Parameter 2")
    ax_bg.set_xlim(*x_limits)
    ax_bg.set_ylim(*y_limits)
    fig_bg.tight_layout()
    fig_bg.savefig(bg_path, dpi=350)
    plt.close(fig_bg)

    # Keep the usual matplotlib preview plot.
    plt.figure(figsize=(6, 4))
    cf = plt.tricontourf(tri, lcb_clipped, levels=levels, cmap='coolwarm', norm=norm)
    plt.tricontour(tri, lcb_clipped, levels=levels, colors='black', linewidths=0.5)
    cbar = plt.colorbar(cf)
    cbar.set_ticks([z_min, lcb.max()])
    cbar.set_ticklabels(['Low', 'High'])

    if safe_points.shape[0] >= 3:
        try:
            hull = ConvexHull(safe_points)
            hull_pts = safe_points[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            plt.plot(hull_pts[:, 0], hull_pts[:, 1], color='green', linewidth=2)
            plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='green', alpha=0.1)
        except QhullError:
            pass
    elif safe_points.shape[0] == 2:
        plt.plot(safe_points[:, 0], safe_points[:, 1], color='green', linewidth=2)
    elif safe_points.shape[0] == 1:
        circle = plt.Circle((safe_points[0, 0], safe_points[0, 1]), 0.01, color='green', fill=False, linewidth=2)
        plt.gca().add_patch(circle)

    init_x = cube.x_sample[0, 0].item()
    init_y = cube.x_sample[0, 1].item()
    plt.scatter(init_x, init_y, color='magenta', marker='D', s=70)
    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.tight_layout()
    plt.show()

    def _coords_string(points):
        return " ".join(f"({x:.8f},{y:.8f})" for x, y in points)

    tikz_lines = [
        "% !TEX root = ../root.tex",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "width = 0.45\\textwidth,",
        "height = 0.25\\textheight,",
        "label style={font=\\scriptsize},",
        "tick align=outside,",
        "tick pos=left,",
        f"xmin={x_limits[0]:.8f}, xmax={x_limits[1]:.8f},",
        f"ymin={y_limits[0]:.8f}, ymax={y_limits[1]:.8f},",
        "xlabel={$f_\\alpha$}, ylabel={$f_\\theta$},",
        "axis on top,",
        "enlargelimits=false",
        "]",
        "\\addplot graphics ["
        f"xmin={x_limits[0]:.8f}, xmax={x_limits[1]:.8f}, ymin={y_limits[0]:.8f}, ymax={y_limits[1]:.8f}] "
        f"{{{bg_path}}};",
    ]

    if safe_points.shape[0] >= 3:
        try:
            hull = ConvexHull(safe_points)
            hull_pts = safe_points[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            tikz_lines.append(
                "\\addplot [draw=green!60!black, semithick, fill=green, fill opacity=0.10] "
                f"coordinates {{{_coords_string(hull_pts)}}};"
            )
        except QhullError:
            pass
    elif safe_points.shape[0] == 2:
        tikz_lines.append(
            "\\addplot [draw=green!60!black, semithick] "
            f"coordinates {{{_coords_string(safe_points)}}};"
        )
    elif safe_points.shape[0] == 1:
        x0, y0 = safe_points[0]
        tikz_lines.append(
            "\\draw[green!60!black, semithick] "
            f"(axis cs:{x0:.8f},{y0:.8f}) circle [radius=0.01];"
        )

    tikz_lines.append(
        "\\addplot [only marks, mark=diamond*, mark size=2.5pt, color=magenta] "
        f"coordinates {{({init_x:.8f},{init_y:.8f})}};"
    )
    tikz_lines.append("\\end{axis}")
    tikz_lines.append("\\end{tikzpicture}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tikz_lines) + "\n")

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

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError


from scipy.spatial import ConvexHull, QhullError
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_intro_blue(cube):
    # --- extract ---
    lcb = cube.lcb_rew_init.detach().cpu().numpy()
    X_plot = cube.discr_domain.detach().cpu().numpy()

    # --- triangulation ---
    tri = Triangulation(X_plot[:, 0], X_plot[:, 1])

    # --- clip (flatten background) ---
    z_min = np.percentile(lcb, 85)  # tune if needed (0.95)
    lcb_clipped = np.clip(lcb, z_min, lcb.max())  # (1.00)
    norm = Normalize(vmin=z_min, vmax=lcb.max())  # (1.00)

    # --- safe set ---
    S = (torch.min(cube.lcb_con_1_init, cube.lcb_con_2_init) > 0)
    safe_points = cube.discr_domain[S].detach().cpu().numpy()

    plt.figure(figsize=(5,5))

    # --- contours ---
    levels = np.linspace(z_min, lcb.max(), 12)  # (0.95)
    cf = plt.tricontourf(tri, lcb_clipped, levels=levels, cmap='coolwarm', norm=norm)
    plt.tricontour(tri, lcb_clipped, levels=levels, colors='black', linewidths=0.5)

# --- correct colorbar ---
    # cbar = plt.colorbar(cf)  # (1.00)
    # cbar.set_ticks([z_min, lcb.max()])  # (1.00)
    # cbar.set_ticklabels(['Low', 'High'])  # (1.00)


    # --- ONE green hull ---
    if safe_points.shape[0] >= 3:
        try:
            hull = ConvexHull(safe_points)
            hull_pts = safe_points[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            plt.plot(hull_pts[:, 0], hull_pts[:, 1], color='green', linewidth=2)
            plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='green', alpha=0.1)
        except QhullError:
            pass
    elif safe_points.shape[0] == 2:
        plt.plot(safe_points[:, 0], safe_points[:, 1], color='green', linewidth=2)
    elif safe_points.shape[0] == 1:
        circle = plt.Circle((safe_points[0,0], safe_points[0,1]),
                            0.01, color='green', fill=False, linewidth=2)
        plt.gca().add_patch(circle)

    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")

    plt.tight_layout()
    plt.scatter(cube.x_sample[0, 0].item(), cube.x_sample[0, 1].item(), color='magenta', marker='D', s=70, label='Initial sample')
    plt.xlim(0.125, 0.30)  # (1.00)
    plt.ylim(0.275, 0.45)  # (1.00)
    plt.show()
    plt.axis('off')  # (1.00)
    ax = plt.gca()  # (1.00)
    ax.set_position([0, 0, 1, 1])  # (1.00)
    plt.savefig("intro_exploration.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig("intro_exploration.pdf", bbox_inches="tight")


def plot_intro_blue_end(cube):
    # --- extract ---
    lcb = cube.lcb_rew.detach().cpu().numpy()
    X_plot = cube.discr_domain.detach().cpu().numpy()

    # --- triangulation ---
    tri = Triangulation(X_plot[:, 0], X_plot[:, 1])

    # --- clip (flatten background) ---
    z_min = np.percentile(lcb, 85)  # tune if needed (0.95)
    lcb_clipped = np.clip(lcb, z_min, lcb.max())  # (1.00)
    norm = Normalize(vmin=z_min, vmax=lcb.max())  # (1.00)

    # --- safe set ---
    S = (torch.min(cube.lcb_con_1, cube.lcb_con_2) > 0)
    safe_points = cube.discr_domain[S].detach().cpu().numpy()

    plt.figure(figsize=(5,5))

    # --- contours ---
    levels = np.linspace(z_min, lcb.max(), 12)  # (0.95)
    cf = plt.tricontourf(tri, lcb_clipped, levels=levels, cmap='coolwarm', norm=norm)
    plt.tricontour(tri, lcb_clipped, levels=levels, colors='black', linewidths=0.5)

# --- correct colorbar ---
    # cbar = plt.colorbar(cf)  # (1.00)
    # cbar.set_ticks([z_min, lcb.max()])  # (1.00)
    # cbar.set_ticklabels(['Low', 'High'])  # (1.00)


    # --- ONE green hull ---
    hull = ConvexHull(safe_points)
    hull_pts = safe_points[hull.vertices]
    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    plt.plot(hull_pts[:, 0], hull_pts[:, 1], color='green', linewidth=2)
    plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='green', alpha=0.1)


    best_parameter = cube.discr_domain[cube.S][torch.argmax(cube.lcb_rew[cube.S])].detach().numpy()

    plt.tight_layout()
    plt.scatter(cube.x_sample[0, 0].item(), cube.x_sample[0, 1].item(), color='magenta', marker='D', s=70, label='Initial sample')
    plt.scatter(cube.x_sample[1:, 0].detach().cpu().numpy(), cube.x_sample[1:, 1].detach().cpu().numpy(), color='black', marker='o', s=70, label='Samples')
    plt.scatter(best_parameter[0], best_parameter[1], color='cyan', marker='s', s=70, label='Best parameter')
    plt.xlim(0.125, 0.30)  # (1.00)
    plt.ylim(0.275, 0.45)  # (1.00)
    plt.show()
    plt.axis('off')  # (1.00)
    ax = plt.gca()  # (1.00)
    ax.set_position([0, 0, 1, 1])  # (1.00)
    plt.savefig("intro_exploration_end.pdf", dpi=300, bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    cube = dill.load(open("cube_final.pkl", "rb"))
    angles = dill.load(open("angles.pkl", "rb"))
    plot_intro_blue(cube)
    plot_intro_blue_end(cube)
    print(123)
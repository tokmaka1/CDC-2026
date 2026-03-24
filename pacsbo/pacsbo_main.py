import torch
import torch.nn as nn
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tqdm import tqdm
import time
import warnings
import copy
# import concurrent
import multiprocessing
#import tikzplotlib
import pickle
import dill
from matplotlib.patches import Ellipse
import torch.multiprocessing as mp
from scipy.special import comb
# from plot import plot_1D, plot_2Dtour, plot_1D_SafeOpt_with_sets, plot_gym, plot_gym_together
import sys
import os
# from IPython import embed as IPS



def compute_X_plot(n_dimensions, points_per_axis, beginning=0, end=1):
    if type(beginning) == int or type(beginning) == float:
        X_plot_per_domain = torch.linspace(beginning, end, points_per_axis)
        X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
        X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    elif type(beginning) == list:
        X_plot = torch.cartesian_prod(*(torch.linspace(beginning[j], end[j], points_per_axis) for j in range(n_dimensions))).reshape(-1, n_dimensions)
    return X_plot


def initial_safe_samples(gt, num_safe_points, X_plot, R, safety_threshold):
    num_safe_points = num_safe_points
    sampling_logic = torch.logical_and(gt.fX > torch.quantile(gt.fX, 0.75), gt.fX < torch.quantile(gt.fX, 0.85))  # Student t
    random_indices_sample = torch.randint(high=X_plot[sampling_logic].shape[0], size=(num_safe_points,))
    X_sample = X_plot[sampling_logic][random_indices_sample]
    Y_sample = gt.conduct_experiment(X_sample)
    return X_sample, Y_sample


class PACSBO():
    def __init__(self, X_plot, X_sample, safety_threshold):
        # self.gt = gt  # at least for toy experiments it works like this.
        self.discr_domain = X_plot
        self.n_dimensions = X_plot.shape[1]
        self.safety_threshold = safety_threshold
        self.x_sample = X_sample.clone().detach()
        self.S = (self.discr_domain==self.x_sample).all(dim=1)
        if sum(self.S) == 0:
            raise Exception("Safe set is empty")



    def compute_safe_set(self):
        # Safe set is defined as all points whose current lower confidence bound
        # is above the safety threshold (i.e., probabilistically safe).
        self.S = self.lcb_con > self.safety_threshold

        # Auxiliary objects of potential maximizers M and potential expanders G
        self.G = self.S.clone()
        self.M = self.S.clone()

    def maximizer_routine(self):
        self.M[:] = False  # initialize
        self.max_M_var = 0  # initialize
        if not torch.any(self.S):  # no safe points
            return

        self.M[self.S] = self.ucb_rew[self.S] >= max(self.lcb_rew[self.S])
        if not torch.any(self.M):
            return
        self.max_M_var = torch.max(self.ucb_rew[self.M] - self.lcb_rew[self.M])
        self.max_M_ucb = torch.max(self.ucb_rew[self.M])

    def expander_routine(self):
        self.G[:] = False  # initialize
        if not torch.any(self.S) or torch.all(self.S):  # no safe points or all are safe
            return

        # candidate safe points that are not maximizers and still uncertain enough in constraint model
        safe_non_max = torch.logical_and(
            torch.logical_and(self.S, ~self.M),
            (self.ucb_con - self.lcb_con) > self.max_M_var,
        )
        if not torch.any(safe_non_max):
            return

        unsafe_mask = ~self.S
        if not torch.any(unsafe_mask):
            return

        safe_pts = self.discr_domain[safe_non_max]
        unsafe_pts = self.discr_domain[unsafe_mask]

        # Identify safe non-max points that are closest to the unsafe set (G_t from the paper)
        # G_t := { a in S_t \ M_t | a is among the closest points in S_t \ M_t to domain \ S_t }
        dists = torch.cdist(safe_pts, unsafe_pts)
        min_dist_per_safe, _ = dists.min(dim=1)
        min_dist = min_dist_per_safe.min()
        is_closest = min_dist_per_safe == min_dist

        # Build mask over the full discretized domain
        self.G[safe_non_max] = is_closest

        # avoid double-sampling the point already sampled
        # self.G[self.G.clone()] = ~(self.discr_domain[self.G] == self.x_sample).any(dim=0).flatten(); is this even necessary?


       
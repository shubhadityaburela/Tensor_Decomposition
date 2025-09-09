import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator

from Minimizer import argmin_H
from minimizer_helper import generate_phi
import opt_einsum as oe

import os


# Plots the results
impath = "plots/Wildlandfire_1d/"  # For plots
immpath = "data/Wildlandfire_1d/"  # For plots
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)
cmap = "YlOrRd"


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



class Parameters:
    nf: int = 3
    n: int = None
    m: int = None
    K: np.ndarray = np.asarray([1, 1, 1])  # We use a single type of basis in each frame
    type_basis: np.ndarray = np.asarray([["gaussian"], ["gaussian"], ["gaussian"]])  # We use a single Gaussian basis for each frame
    K_st: np.ndarray = np.asarray([0, 1, 2, 3])  # Just to get the indexing access of the array right
    sigma: np.ndarray = np.asarray([[4.0], [4.0], [4.0]])  # Gaussian variance for each frame

    x: np.ndarray = None
    t: np.ndarray = None

    degree: np.ndarray = np.asarray([2, 1, 2])  # We use a linear polynomial for the first and last frame and a constant for the middle frame
    degree_st: np.ndarray = np.asarray([0, 2, 3, 5])  # Just to get the indexing access of the array right
    beta_init = [[0.1, -0.5], [0], [-0.8, 1.0]]  # Initial guess value for the coefficients of the shifts
    type_shift = ["polynomial", "polynomial", "polynomial"]  # We use polynomial shifts for all the frames

    alpha_init: np.ndarray = None
    alpha_solver_ck: float = 1e8
    alpha_solver_lamda_1: float = 1e-3

    beta_solver_dk: float = 1.0
    beta_solver_tau: np.ndarray = np.asarray([1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
    beta_solver_sigma: np.ndarray = 0.99 / beta_solver_tau
    beta_solver_lamda_2: float = 1e-3
    beta_solver_rho_n: float = 1.0
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 5

    gtol: float = 1e-8
    maxit: int = 100000


if __name__ == '__main__':

    # Load the wild land fire data
    q = np.load('data/Wildlandfire_1d/SnapShotMatrix558.49.npy', allow_pickle=True)
    X = np.load('data/Wildlandfire_1d/1D_Grid.npy', allow_pickle=True)
    t = np.load('data/Wildlandfire_1d/Time.npy', allow_pickle=True)
    x = X[0]
    Nx = int(np.size(x))
    Nt = int(np.size(t))
    Q = q[:Nx, :]

    # Normalize the input data
    Q = (Q - Q.min())/(Q.max() - Q.min())

    XX, TT = np.meshgrid(x, t)

    # # Plot the data
    # fig1 = plt.figure(figsize=(5, 5))
    # ax1 = fig1.add_subplot(111)
    # vmin = np.min(Q)
    # vmax = np.max(Q)
    # im1 = ax1.pcolormesh(XX.T, TT.T, Q, vmin=vmin, vmax=vmax, cmap=cmap)
    # ax1.set_xlabel(r"$x$")
    # ax1.set_ylabel(r"$t$")
    # ax1.axis('off')
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig1.colorbar(im1, cax=cax, orientation='vertical')
    # fig1.supylabel(r"time $t$")
    # fig1.supxlabel(r"space $x$")
    # fig1.savefig(impath + "Q", dpi=300, transparent=True)

    # Instantiate the constants for the optimization
    param = Parameters()
    param.n = Nx
    param.m = Nt
    param.x = x
    param.t = t
    param.center_of_matrix = np.asarray([x[-1] // 2, x[-1] // 2, x[-1] // 2])
    param.alpha_init = np.zeros((sum(param.K), param.m))

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(param, beta)
    Q1 = oe.contract('ijk,ik->jk', phiBeta[param.K_st[0]:param.K_st[1], ...], alpha[param.K_st[0]:param.K_st[1], :])
    Q2 = oe.contract('ijk,ik->jk', phiBeta[param.K_st[1]:param.K_st[2], ...], alpha[param.K_st[1]:param.K_st[2], :])
    Q3 = oe.contract('ijk,ik->jk', phiBeta[param.K_st[2]:param.K_st[3], ...], alpha[param.K_st[2]:param.K_st[3], :])

    # Save the data
    np.save(os.path.join(immpath, "Q.npy"), Q)
    np.save(os.path.join(immpath, "Q1.npy"), Q1)
    np.save(os.path.join(immpath, "Q2.npy"), Q2)
    np.save(os.path.join(immpath, "Q3.npy"), Q3)

    # # Plot the separated frames
    # fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True, sharex=True)
    # # Original
    # im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
    # axs[0].set_title(r"$Q$")
    # axs[0].set_ylabel(r"$t$")
    # axs[0].set_xlabel(r"$x$")
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # cbar0 = fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)
    #
    # # Frame 1
    # im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
    # axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    # axs[1].set_ylabel(r"$t$")
    # axs[1].set_xlabel(r"$x$")
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    # cbar1 = fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)
    #
    # # Frame 2
    # im2 = axs[2].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
    # axs[2].set_title(r"$\mathcal{T}^2Q^2$")
    # axs[2].set_ylabel(r"$t$")
    # axs[2].set_xlabel(r"$x$")
    # axs[2].set_xticks([])
    # axs[2].set_yticks([])
    # cbar2 = fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)
    #
    # # Frame 3
    # im3 = axs[3].pcolormesh(Q3.T, vmin=vmin, vmax=vmax, cmap=cmap)
    # axs[3].set_title(r"$\mathcal{T}^3Q^3$")
    # axs[3].set_ylabel(r"$t$")
    # axs[3].set_xlabel(r"$x$")
    # axs[3].set_xticks([])
    # axs[3].set_yticks([])
    # cbar3 = fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)
    #
    # # Frame 4
    # im4 = axs[4].pcolormesh((Q1 + Q2 + Q3).T, vmin=vmin, vmax=vmax, cmap=cmap)
    # axs[4].set_title(r"$\tilde{Q}$")
    # axs[4].set_ylabel(r"$t$")
    # axs[4].set_xlabel(r"$x$")
    # axs[4].set_xticks([])
    # axs[4].set_yticks([])
    # cbar4 = fig.colorbar(im4, ax=axs[4], orientation="vertical", fraction=0.046, pad=0.04)
    #
    # fig.tight_layout()
    # fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
    #
    # # Plot the cost functional
    # fig1 = plt.figure(figsize=(12, 12))
    # ax1 = fig1.add_subplot(111)
    # ax1.semilogy(np.arange(len(J)), J, color="C0", label=r"$J(\alpha, \beta)$")
    # ax1.set_xlabel(r"$n_{\mathrm{iter}}$")
    # ax1.set_ylabel(r"$J$")
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax1.tick_params(axis='x')
    # ax1.tick_params(axis='y')
    # ax1.legend()
    # fig1.savefig(impath + "J", dpi=300, transparent=True)

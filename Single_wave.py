import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
import os

from Minimizer import argmin_H
from data_generation import generate_data_single_wave
from minimizer_helper import generate_phi
import opt_einsum as oe

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


impath = "plots/single_wave/"  # For plots
immpath = "data/single_wave/"  # For data
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)
cmap = "YlOrRd"


class Parameters:
    nf: int = 1
    n: int = 300
    m: int = 300
    K: np.ndarray = np.asarray([1])  # We use a single type of basis
    type_basis: np.ndarray = np.asarray([["gaussian"]])  # We use a single Gaussian basis
    K_st: np.ndarray = np.asarray([0, 1])  # Just to get the indexing access of the array right
    sigma: np.ndarray = np.asarray([[4.0]])  # Gaussian variance

    x: np.ndarray = None
    t: np.ndarray = None
    t_start: float = -10.0
    t_end: float = 10.0

    degree: np.ndarray = np.asarray([2])  # We use a linear polynomial
    degree_st: np.ndarray = np.asarray([0, 2])  # Just to get the indexing access of the array right
    beta_init: np.ndarray = np.asarray([2.0, -1.0])  # Initial guess value for the coefficients of the shifts
    type_shift = ["polynomial"]  # We use polynomial shift
    beta: np.ndarray = np.asarray([-10.0, 1.5])
    center_of_matrix: np.ndarray = np.asarray([150.0])

    alpha_init: np.ndarray = np.ones((sum(K), m))
    alpha_solver_ck: float = 1e5
    alpha_solver_lamda_1: float = 1e-3

    beta_solver_dk: float = 1e-1
    beta_solver_tau: float = 5e-6
    beta_solver_sigma: float = 0.99 / beta_solver_tau
    beta_solver_lamda_2: float = 1e-3
    beta_solver_rho_n: float = 1.0
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 5

    gtol: float = 1e-10
    maxit: int = 50000


if __name__ == '__main__':

    # Instantiate the constants for the optimization and construct the grid
    param = Parameters()
    param.x = np.arange(0, param.n)
    param.t = np.linspace(param.t_start, param.t_end, param.m)

    # Generate the data
    Q = generate_data_single_wave(param)

    # # Plot the data
    # fig1 = plt.figure(figsize=(5, 5))
    # ax1 = fig1.add_subplot(111)
    # vmin = np.min(Q)
    # vmax = np.max(Q)
    # im1 = ax1.pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
    # ax1.set_xlabel(r"$x$")
    # ax1.set_ylabel(r"$t$")
    # ax1.axis('off')
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig1.colorbar(im1, cax=cax, orientation='vertical')
    # fig1.supylabel(r"time $t$")
    # fig1.supxlabel(r"space $x$")
    # fig1.savefig(impath + "Q", dpi=300, transparent=True)

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(param, beta)
    Q1 = oe.contract('ijk,ik->jk', phiBeta, alpha)


    # Save the data
    np.save(os.path.join(immpath, "Q.npy"), Q)
    np.save(os.path.join(immpath, "Q1.npy"), Q1)


    # # Plots the results
    # # Plot the separated frames
    # fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True, sharex=True)
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
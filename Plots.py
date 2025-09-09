import matplotlib.pyplot as plt
import numpy as np

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



def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


# problem = "Single_wave"
# problem = "StraightCubic_wave"
# problem = "Sine_StraightCubic_wave"
problem = "Wildlandfire_1d"
cmap = "YlOrRd"

if problem == "Single_wave":
    impath = "plots/single_wave/"
    immpath = "data/single_wave/"
    Q = np.load(immpath + "Q.npy")
    Q1 = np.load(immpath + "Q1.npy")
    vmin = np.min(Q)
    vmax = np.max(Q)

    # Plots the results
    # Plot the separated frames
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True, sharex=True)
    # Original
    im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 1
    im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title(r"$\mathcal{T}^1Q^1 = \tilde{Q}$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
    save_fig(impath + 'Q_opti', fig)

    rec_err = np.linalg.norm(Q - Q1) / np.linalg.norm(Q)
    print(f"RecErr: {rec_err}")

elif problem == "StraightCubic_wave":
    impath = "plots/StraightCubic_wave/"
    immpath = "data/StraightCubic_wave/"
    Q = np.load(immpath + "Q.npy")
    Q1 = np.load(immpath + "Q1.npy")
    Q2 = np.load(immpath + "Q2.npy")
    vmin = np.min(Q)
    vmax = np.max(Q)

    # Plot the separated frames
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
    # Original
    im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 1
    im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 2
    im2 = axs[2].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[2].set_title(r"$\mathcal{T}^2Q^2$")
    axs[2].set_ylabel(r"$t$")
    axs[2].set_xlabel(r"$x$")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    cbar2 = fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

    # Reconstructed
    im3 = axs[3].pcolormesh((Q1 + Q2).T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[3].set_title(r"$\tilde{Q}$")
    axs[3].set_ylabel(r"$t$")
    axs[3].set_xlabel(r"$x$")
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    cbar3 = fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
    save_fig(impath + 'Q_opti', fig)

    rec_err = np.linalg.norm(Q - (Q1 + Q2)) / np.linalg.norm(Q)
    print(f"RecErr: {rec_err}")

elif problem == "Sine_StraightCubic_wave":
    impath = "plots/Sine_StraightCubic_wave/"
    immpath = "data/Sine_StraightCubic_wave/"
    Q = np.load(immpath + "Q.npy")
    Q1 = np.load(immpath + "Q1.npy")
    Q2 = np.load(immpath + "Q2.npy")
    vmin = np.min(Q)
    vmax = np.max(Q)

    # Plot the separated frames
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
    # Original
    im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 1
    im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 2
    im2 = axs[2].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[2].set_title(r"$\mathcal{T}^2Q^2$")
    axs[2].set_ylabel(r"$t$")
    axs[2].set_xlabel(r"$x$")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    cbar2 = fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

    # Reconstructed
    im3 = axs[3].pcolormesh((Q1 + Q2).T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[3].set_title(r"$\tilde{Q}$")
    axs[3].set_ylabel(r"$t$")
    axs[3].set_xlabel(r"$x$")
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    cbar3 = fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
    save_fig(impath + 'Q_opti', fig)

    rec_err = np.linalg.norm(Q - (Q1 + Q2)) / np.linalg.norm(Q)
    print(f"RecErr: {rec_err}")

elif problem == "Wildlandfire_1d":
    impath = "plots/Wildlandfire_1d/"
    immpath = "data/Wildlandfire_1d/"
    Q = np.load(immpath + "Q.npy")
    Q1 = np.load(immpath + "Q1.npy")
    Q2 = np.load(immpath + "Q2.npy")
    Q3 = np.load(immpath + "Q3.npy")
    vmin = np.min(Q)
    vmax = np.max(Q)


    # Plot the separated frames
    fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True, sharex=True)
    # Original
    im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 1
    im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 2
    im2 = axs[2].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[2].set_title(r"$\mathcal{T}^2Q^2$")
    axs[2].set_ylabel(r"$t$")
    axs[2].set_xlabel(r"$x$")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    cbar2 = fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 3
    im3 = axs[3].pcolormesh(Q3.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[3].set_title(r"$\mathcal{T}^3Q^3$")
    axs[3].set_ylabel(r"$t$")
    axs[3].set_xlabel(r"$x$")
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    cbar3 = fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

    # Frame 4
    im4 = axs[4].pcolormesh((Q1 + Q2 + Q3).T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[4].set_title(r"$\tilde{Q}$")
    axs[4].set_ylabel(r"$t$")
    axs[4].set_xlabel(r"$x$")
    axs[4].set_xticks([])
    axs[4].set_yticks([])
    cbar4 = fig.colorbar(im4, ax=axs[4], orientation="vertical", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
    save_fig(impath + 'Q_opti', fig)

    rec_err = np.linalg.norm(Q - (Q1 + Q2 + Q3)) / np.linalg.norm(Q)
    print(f"RecErr: {rec_err}")
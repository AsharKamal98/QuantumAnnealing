from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator



#import matplotlib
#import matplotlib.cm as cm
#import matplotlib.patches as mpatches
#from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
#import matplotlib.colors as colors

#import pandas as pd
#import seaborn as sns
#import ast

#import random
#from sklearn.preprocessing import StandardScaler, FunctionTransformer
#from scipy import stats
#from scipy.stats import gaussian_kde

#import glob


minorLocator=AutoMinorLocator()
mpl.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
colorsbasis=['#00A0B0','#6A4A3C','#CC333F','#EB6841','#EDC951','#A3A948','#B3E099','#5165E9','#B1B59D']
#-------------------------------------------------#

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams.update({'font.size': 20})
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams["figure.figsize"] = [8.5, 5.5]


def PhiHistory1(basis):
    """ Makes subplots of all state pobabilities for given basis """
    num_subplots = 8
    num_rows = 2
    num_cols = 4
    plot_nth_element = 10


    with open("Statistics/PhiHistoryCpp_{}.txt".format(basis), "r") as f:
        l = f.readlines()
    phi_history = np.array([l[i].split() for i in range(len(l))])
    phi_history = phi_history.astype(np.float64)[::plot_nth_element]
    iterations = np.arange(phi_history.shape[0])


    fig, axes = plt.subplots(2,4, figsize=(17,10), sharex=True, sharey=True)
    fig.tight_layout()

    for i in range(num_subplots):
        grid_index1 = int(i/num_cols)
        grid_index2 = i%num_cols

        axes[grid_index1, grid_index2].scatter(iterations, phi_history[:,i], s=7)
        axes[grid_index1, grid_index2].grid(which='minor', color='lightgrey', linestyle='--')
        axes[grid_index1, grid_index2].grid(which='major', color='grey', linestyle='-')
        axes[grid_index1, grid_index2].tick_params(labelsize=25)
        axes[grid_index1, grid_index2].set_axisbelow(True)

        if grid_index1 == num_rows-1:
            axes[grid_index1, grid_index2].set_xlabel(r'Iterations', fontweight='bold', fontsize=20)
        if grid_index2 == 0:
            axes[grid_index1, grid_index2].set_ylabel(r'Probability', fontweight='bold', fontsize=20)

        if grid_index1 == 0 and grid_index2 == 0:
            axes[grid_index1, grid_index2].set_title(r'Ground state', fontweight='bold', fontsize=30)
        elif grid_index1 == num_rows-1 and grid_index2 == num_cols-1:
            axes[grid_index1, grid_index2].set_title(r'Highest excited state', fontweight='bold', fontsize=30)
        else:
            axes[grid_index1, grid_index2].set_title(r'Excited state', fontweight='bold', fontsize=30)


    plt.tight_layout()
    plt.savefig("Figures/PhiHistoryCpp_{}.pdf".format(basis), bbox_inches="tight", dpi=800)

    return


def PhiHistory2():
    """ Makes plot of probability of being in ground state in instataneous basis """
    plot_nth_element = 10


    with open("PhiHistoryCpp_x.txt", "r") as f:
        l = f.readlines()
    phi_history = np.array([l[i].split() for i in range(len(l))])
    phi_history = phi_history.astype(np.float64)[::plot_nth_element]
    iterations = np.arange(phi_history.shape[0])


    fig, ax = plt.subplots(figsize=(17,10))
    fig.tight_layout()


    ax.scatter(iterations, phi_history[:,0], s=7)
    ax.grid(which='minor', color='lightgrey', linestyle='--')
    ax.grid(which='major', color='grey', linestyle='-')
    ax.tick_params(labelsize=25)
    ax.set_axisbelow(True)

    ax.set_xlabel(r'Iterations', fontweight='bold', fontsize=20)
    ax.set_ylabel(r'Probability', fontweight='bold', fontsize=20)

    ax.set_title(r'Ground state', fontweight='bold', fontsize=30)


    plt.tight_layout()
    plt.savefig("Figures/PhiHistoryCpp_x2.pdf", bbox_inches="tight", dpi=800)

    return


def SR():

    with open("Statistics/AverageProbCppFinal.txt", "r") as f:
        l = f.readlines()
    data = np.array([l[i].split() for i in range(len(l))])
    data = data.astype(np.float64)
    num_qbits = data[:,0]
    MCWF_probs = data[:,1]
    std_devs = data[:,2]
    std_errors = data[:,3]
    boltzman_probs = data[:,4]

    fig, ax = plt.subplots(figsize=(10,8))
    fig.tight_layout()

    #ax.scatter(num_qbits, MCWF_probs, s=10)
    #ax.scatter(num_qbits, boltzman_probs, s=10)

    ax.errorbar(num_qbits, MCWF_probs, yerr=std_errors, ls='none', marker='.', markersize=15, label="MCWF")
    ax.plot(num_qbits, boltzman_probs,  marker='.', markersize=15, label="Boltzman Distribution")

    ax.grid(which='minor', color='lightgrey', linestyle='--')
    ax.grid(which='major', color='grey', linestyle='-')
    ax.tick_params(labelsize=25)
    ax.set_axisbelow(True)
    ax.legend()

    ax.set_xlabel(r'Number of Qbits', fontweight='bold', fontsize=20)
    ax.set_xlim(1.90, 6)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel(r'Success Rate', fontweight='bold', fontsize=20)
    ax.set_ylim(0, 1)

    #ax.set_title(r'Ground state', fontweight='bold', fontsize=30)


    plt.tight_layout()
    plt.savefig("Figures/SuccessRate.pdf", bbox_inches="tight", dpi=800)


    return


#PhiHistory1(basis='x4')
#PhiHistory2()

SR()

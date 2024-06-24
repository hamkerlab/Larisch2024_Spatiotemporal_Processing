import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from scipy import signal, stats

import os

def plot_compare():

    if not os.path.exists('compare_DeltaT'):
        os.mkdir('compare_DeltaT')


    model_list = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']

    n_models = len(model_list)

    dsi_data_ref_all = []
    dsi_data_nolag_Inh_all = []
    crossCorr_pref_0p5_50ms_all = []
    crossCorr_null_0p5_50ms_all = []

    crossCorr_pref_nolag_all = []
    crossCorr_null_nolag_all = []

    for m in range(n_models):


        dsi_data_ref = np.squeeze(np.load(model_list[m]+'/basis/dsi_kim_Cells_TC_0p5_50ms.npy'))
        dsi_data_ref_all.append(dsi_data_ref)

        dsi_data_0p5_50ms = np.squeeze(np.load(model_list[m]+'/0p5_50ms/work/dsi_TC_basis_STRF_kim.npy'))
        dsi_data_nolag_Inh = np.squeeze( np.load(model_list[m]+'/nolagged_Inh/work/dsi_TC_basis_STRF_kim.npy'))
        dsi_data_nolag_Inh_all.append(dsi_data_nolag_Inh)
    
        crossCorr_pref_0p5_50ms =np.squeeze( np.load(model_list[m]+'/0p5_50ms/work/dircection_corrCorr_Currents_pref_same.npy'))
        crossCorr_null_0p5_50ms =np.squeeze( np.load(model_list[m]+'/0p5_50ms/work/dircection_corrCorr_Currents_null_same.npy'))    

        crossCorr_pref_0p5_50ms_all.append(crossCorr_pref_0p5_50ms)
        crossCorr_null_0p5_50ms_all.append(crossCorr_null_0p5_50ms)

        crossCorr_pref_nolag_Inh =np.squeeze( np.load(model_list[m]+'/nolagged_Inh/work/dircection_corrCorr_Currents_pref_same_basis.npy'))
        crossCorr_null_nolag_Inh = np.squeeze(np.load(model_list[m]+'/nolagged_Inh/work/dircection_corrCorr_Currents_null_same_basis.npy'))

        crossCorr_pref_nolag_all.append(crossCorr_pref_nolag_Inh)
        crossCorr_null_nolag_all.append(crossCorr_null_nolag_Inh)

    dsi_data_ref_all            = np.asarray(dsi_data_ref_all)
    dsi_data_nolag_Inh_all          = np.asarray(dsi_data_nolag_Inh_all)
    crossCorr_pref_0p5_50ms_all = np.asarray(crossCorr_pref_0p5_50ms_all)
    crossCorr_null_0p5_50ms_all = np.asarray(crossCorr_null_0p5_50ms_all)
    crossCorr_pref_nolag_all    = np.asarray(crossCorr_pref_nolag_all)
    crossCorr_null_nolag_all    = np.asarray(crossCorr_null_nolag_all)

    n_models, n_cells, n_steps = np.shape(crossCorr_pref_0p5_50ms_all)


    dsi_data_ref_all = np.reshape(dsi_data_ref_all,(n_models* n_cells))
    dsi_data_nolag_Inh_all = np.reshape(dsi_data_nolag_Inh_all,(n_models* n_cells))
    crossCorr_pref_0p5_50ms_all = np.reshape(crossCorr_pref_0p5_50ms_all,(n_models * n_cells,n_steps))
    crossCorr_null_0p5_50ms_all = np.reshape(crossCorr_null_0p5_50ms_all,(n_models * n_cells,n_steps))
    crossCorr_pref_nolag_all = np.reshape(crossCorr_pref_nolag_all,(n_models * n_cells,n_steps))
    crossCorr_null_nolag_all = np.reshape(crossCorr_null_nolag_all,(n_models * n_cells,n_steps))    

  
    idx_high = np.where(dsi_data_ref_all >= 0.8)[0]


    crossCorr_pref_0p5_50ms_all_high = crossCorr_pref_0p5_50ms_all[idx_high]
    crossCorr_pref_nolag_all_high = crossCorr_pref_nolag_all[idx_high]

    crossCorr_null_0p5_50ms_all_high = crossCorr_null_0p5_50ms_all[idx_high]
    crossCorr_null_nolag_all_high = crossCorr_null_nolag_all[idx_high]




    delta_DSI = dsi_data_ref_all[idx_high]-dsi_data_nolag_Inh_all[idx_high]


    t_wi = 21
    lags = signal.correlation_lags(t_wi, t_wi)
    crossCorr_pref_0p5_50ms_high_maxT    = lags[np.argmax(crossCorr_pref_0p5_50ms_all_high,axis=1)]
    crossCorr_pref_nolag_Inh_high_maxT   = lags[np.argmax(crossCorr_pref_nolag_all_high,axis=1)]

    crossCorr_null_0p5_50ms_high_maxT    = lags[np.argmax(crossCorr_null_0p5_50ms_all_high,axis=1)]
    crossCorr_null_nolag_Inh_high_maxT   = lags[np.argmax(crossCorr_null_nolag_all_high,axis=1)]



    delta_dT_pref = (crossCorr_pref_0p5_50ms_high_maxT - crossCorr_pref_nolag_Inh_high_maxT)
    delta_dT_null = (crossCorr_null_0p5_50ms_high_maxT - crossCorr_null_nolag_Inh_high_maxT)

    delta_dT_noLag = np.abs(crossCorr_pref_nolag_Inh_high_maxT - crossCorr_null_nolag_Inh_high_maxT)
    delta_dT_Lag = np.abs(crossCorr_pref_0p5_50ms_high_maxT - crossCorr_null_0p5_50ms_high_maxT)



    fig = plt.figure(figsize=(5,4))
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    #ax_cb = fig.add_subplot(gs[1:4,0])    

    scat = ax_joint.scatter(crossCorr_pref_0p5_50ms_high_maxT, crossCorr_pref_nolag_Inh_high_maxT, c=delta_DSI, vmin=-0.2, vmax=1, alpha=0.7, cmap='inferno')
    ax_joint.vlines(0,-21,21, color='dimgray',linestyles='dotted')
    ax_joint.hlines(0,-21,21, color='dimgray',linestyles='dotted')
    ax_joint.text(-19,16,'preferred direction',bbox=dict(boxstyle='round', facecolor='white'))

    ax_marg_x.hist(crossCorr_pref_0p5_50ms_high_maxT, range=(-21,21), bins=15, color='gray', density=True)
    ax_marg_x.vlines(0,0,0.13, color='dimgray',linestyles ='dotted')

    ax_marg_y.hist(crossCorr_pref_nolag_Inh_high_maxT,range=(-21,21), bins=15, color='gray', orientation="horizontal", density=True)
    ax_marg_y.hlines(0,0,0.13, color='dimgray',linestyles ='dotted')
    #ax_marg_y.hlines(-5,0,45, color='red')

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel(r'$\Delta$T with Lag')
    ax_joint.set_ylabel(r'$\Delta$T no Lag')
    ax_joint.set_xlim(-21,21)
    ax_joint.set_ylim(-21,21)
    #ax_joint.legend()

    # Set labels on marginals
    ax_marg_y.set_xlabel('% Cells')
    ax_marg_y.set_xlim(0,0.13)
    ax_marg_y.set_ylim(-21,21)
    ax_marg_x.set_ylabel('% Cells')
    ax_marg_x.set_ylim(0,0.13)
    ax_marg_x.set_xlim(-21,21)
    # add colorbar
    cax = ax_joint.inset_axes([0.0, 1.4, 1, 0.04])
    cb = fig.colorbar(scat,cax=cax, orientation='horizontal', label=r'$\Delta$ DSI')
    cax.yaxis.set_ticks_position('left')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    plt.savefig('./compare_DeltaT/compare_prefDSI_fancy.png', dpi=300,bbox_inches='tight')
    plt.close()



    fig = plt.figure(figsize=(5,4))
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    
    scat = ax_joint.scatter(crossCorr_null_0p5_50ms_high_maxT, crossCorr_null_nolag_Inh_high_maxT, c=delta_DSI, vmin=-0.2, vmax=1, alpha=0.7, cmap='viridis')
    ax_joint.vlines(0,-21,21, color='dimgray',linestyles='dotted')
    ax_joint.hlines(0,-21,21, color='dimgray',linestyles='dotted')
    ax_joint.text(-19,16,'null direction',bbox=dict(boxstyle='round', facecolor='white'))

    ax_marg_x.hist(crossCorr_null_0p5_50ms_high_maxT, range=(-21,21), bins=15, color='gray', density=True)
    ax_marg_x.vlines(0,0,0.13, color='dimgray',linestyles ='dotted')

    ax_marg_y.hist(crossCorr_null_nolag_Inh_high_maxT,range=(-21,21), bins=15, color='gray', orientation="horizontal", density=True)
    ax_marg_y.hlines(0,0,0.13, color='dimgray',linestyles ='dotted')
    #ax_marg_y.hlines(-5,0,45, color='red')

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel(r'$\Delta$T with Lag')
    ax_joint.set_ylabel(r'$\Delta$T no Lag')
    ax_joint.set_xlim(-21,21)
    ax_joint.set_ylim(-21,21)
    #ax_joint.legend()

    # Set labels on marginals
    ax_marg_y.set_xlabel('% Cells')
    ax_marg_y.set_xlim(0,0.13)
    ax_marg_y.set_ylim(-21,21)
    ax_marg_x.set_ylabel('% Cells')
    ax_marg_x.set_ylim(0,0.13)
    ax_marg_x.set_xlim(-21,21)
    # add colorbar
    cax = ax_joint.inset_axes([0.0, 1.4, 1, 0.04])
    cb = fig.colorbar(scat,cax=cax, orientation='horizontal', label=r'$\Delta$ DSI')
    cax.yaxis.set_ticks_position('left')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    plt.savefig('./compare_DeltaT/compare_nullDSI_fancy.png', dpi=300,bbox_inches='tight')
    plt.close()


if __name__=="__main__":
    plot_compare()

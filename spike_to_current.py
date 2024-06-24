import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plotData():

    model_list = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']    

    n_model = len(model_list)

    ## get all the information that are needed

    tm_nSpk_pref_all_high = [] 
    tm_nSpk_null_all_high = []
    tm_nSpk_pref_all_low  = []
    tm_nSpk_null_all_low  = []


    max_crossCalc_pref_high_all = []
    max_crossCalc_pref_low_all  = []
    max_crossCalc_null_high_all = []
    max_crossCalc_null_low_all  = []

    for m in range(n_model):
        data = np.load(model_list[m]+'/0p5_50ms/work/direct_tm_nSpk_pref_all_high_DSI.npy') #
        print(np.shape(data))
        tm_nSpk_pref_all_high.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/direct_tm_nSpk_null_all_high_DSI.npy') #
        tm_nSpk_null_all_high.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/direct_tm_nSpk_pref_all_low_DSI.npy') #
        tm_nSpk_pref_all_low.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/direct_tm_nSpk_null_all_low_DSI.npy')#
        tm_nSpk_null_all_low.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/DSI_max_crossCalc_pref_high.npy')#
        max_crossCalc_pref_high_all.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/DSI_max_crossCalc_pref_low.npy')#
        max_crossCalc_pref_low_all.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/DSI_max_crossCalc_null_high.npy')
        max_crossCalc_null_high_all.append(data)

        data = np.load(model_list[m]+'/0p5_50ms/work/DSI_max_crossCalc_null_low.npy')
        max_crossCalc_null_low_all.append(data)


    ######
    # prepare all the data
    ######


    tm_nSpk_pref_all_high_oR_sumTB = []
    tm_nSpk_null_all_high_oR_sumTB = []    
    tm_nSpk_pref_all_low_oR_sumTB = []
    tm_nSpk_null_all_low_oR_sumTB = []


    max_crossCalc_pref_high = []
    max_crossCalc_pref_low  = []
    max_crossCalc_null_high = []
    max_crossCalc_null_low  = []




    for m in range(n_model):


        tm_nSpk_pref_all_high_oR = np.mean(tm_nSpk_pref_all_high[m],axis=1) 
        tm_nSpk_null_all_high_oR = np.mean(tm_nSpk_null_all_high[m],axis=1)
        tm_nSpk_pref_all_low_oR = np.mean(tm_nSpk_pref_all_low[m],axis=1)
        tm_nSpk_null_all_low_oR = np.mean(tm_nSpk_null_all_low[m],axis=1)

        if m == 0:
            tm_nSpk_pref_all_high_oR_sumTB = np.mean(tm_nSpk_pref_all_high_oR,axis=1)
            tm_nSpk_null_all_high_oR_sumTB = np.mean(tm_nSpk_null_all_high_oR,axis=1)
            tm_nSpk_pref_all_low_oR_sumTB  = np.mean(tm_nSpk_pref_all_low_oR,axis=1)
            tm_nSpk_null_all_low_oR_sumTB  = np.mean(tm_nSpk_null_all_low_oR,axis=1)

            max_crossCalc_pref_high = max_crossCalc_pref_high_all[m]
            max_crossCalc_pref_low =  max_crossCalc_pref_low_all[m]
            max_crossCalc_null_high = max_crossCalc_null_high_all[m]
            max_crossCalc_null_low =  max_crossCalc_null_low_all[m]

        else:
            print(np.shape(tm_nSpk_pref_all_high_oR_sumTB), np.shape(tm_nSpk_pref_all_high_oR))
            tm_nSpk_pref_all_high_oR_sumTB = np.concatenate((tm_nSpk_pref_all_high_oR_sumTB, np.mean(tm_nSpk_pref_all_high_oR,axis=1)), axis=0)
            tm_nSpk_null_all_high_oR_sumTB = np.concatenate((tm_nSpk_null_all_high_oR_sumTB, np.mean(tm_nSpk_null_all_high_oR,axis=1)), axis=0)
            tm_nSpk_pref_all_low_oR_sumTB =  np.concatenate((tm_nSpk_pref_all_low_oR_sumTB, np.mean(tm_nSpk_pref_all_low_oR,axis=1)), axis=0)
            tm_nSpk_null_all_low_oR_sumTB =  np.concatenate((tm_nSpk_null_all_low_oR_sumTB, np.mean(tm_nSpk_null_all_low_oR,axis=1)), axis=0)



            max_crossCalc_pref_high = np.concatenate((max_crossCalc_pref_high,max_crossCalc_pref_high_all[m]), axis=0)
            max_crossCalc_pref_low = np.concatenate(( max_crossCalc_pref_low, max_crossCalc_pref_low_all[m]), axis=0)
            max_crossCalc_null_high = np.concatenate((max_crossCalc_null_high,max_crossCalc_null_high_all[m]), axis=0)
            max_crossCalc_null_low = np.concatenate(( max_crossCalc_null_low, max_crossCalc_null_low_all[m]), axis=0)            
    ## 

    n_cells_high, bins_tm = np.shape(tm_nSpk_pref_all_high_oR_sumTB)

    n_cells_low, _ = np.shape(tm_nSpk_pref_all_low_oR_sumTB)

    t_win = 25
    tm_bins = 6
    tm_bin_list = np.linspace(-0.8,0.8,tm_bins-1)
    tm_labels = [-0.8, -0.4, 0, 0.4, 0.8] #tm_bin_list

    fig, ax = plt.subplots(2,2, figsize=(7,7), sharex=True, sharey=True)
    boxes = []
    for b in range(bins_tm):
        plot_data = max_crossCalc_pref_high[~np.isnan(max_crossCalc_pref_high[:,b]),b]
        ax[0,0].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
        #boxes.append(bp)
    ax[0,0].text(4.0,t_win-4,'DSI > 0.8 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
    #ax[0,0].set_title('pref high')
    ax[0,0].set_ylim(-t_win-1,t_win+1)
    ax[0,0].set_ylabel(r'$\Delta T$')
    ax[0,0].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    #ax[0,0].legend([bp["boxes"][0] for bp in boxes],tm_labels, loc='upper right'  )
    plot_data = []
    for b in range(bins_tm):
        #plot_data.append(max_crossCalc_pref_low[~np.isnan(max_crossCalc_pref_low[:,b]),b])
        plot_data = max_crossCalc_pref_low[~np.isnan(max_crossCalc_pref_low[:,b]),b]
        ax[1,0].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    #ax[1,0].boxplot(plot_data, patch_artist=True)
    ax[1,0].text(4.0,t_win-4,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
    #ax[1,0].set_title('pref low')
    ax[1,0].set_ylim(-t_win-1,t_win+1)
    ax[1,0].set_ylabel(r'$\Delta T$')
    ax[1,0].set_xlabel(r'cos')
    ax[1,0].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        #plot_data.append(max_crossCalc_null_high[~np.isnan(max_crossCalc_null_high[:,b]),b])
        plot_data = max_crossCalc_null_high[~np.isnan(max_crossCalc_null_high[:,b]),b]
        ax[0,1].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    #ax[0,1].boxplot(plot_data, patch_artist=True)
    ax[0,1].text(4.0,t_win-4,'DSI > 0.8 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    #ax[0,1].set_title('null high')
    ax[0,1].set_ylim(-t_win-1,t_win+1)

    ax[0,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        #plot_data.append(max_crossCalc_null_low[~np.isnan(max_crossCalc_null_low[:,b]),b])
        plot_data = max_crossCalc_null_low[~np.isnan(max_crossCalc_null_low[:,b]),b]
        ax[1,1].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    #ax[1,1].boxplot(plot_data, patch_artist=True)
    ax[1,1].set_xticks(np.linspace(1,tm_bins-1,tm_bins-1))
    ax[1,1].set_xticklabels(tm_labels)
    ax[1,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    ax[1,1].text(4.0,t_win-4,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    #ax[1,1].set_title('null low')
    ax[1,1].set_xlabel(r'cos')
    ax[1,1].set_ylim(-t_win-1,t_win+1)
    fig.tight_layout()
    plt.savefig('Cross_Corr_deltaT_TMwise_boxplot',dpi=300,bbox_inches='tight')
    plt.close()

    #################

    fig, ax = plt.subplots(2,2, figsize=(7,7), sharex=True, sharey=True)
    boxes = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_pref_all_high_oR_sumTB[:,b]
        ax[0,0].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[0,0].text(4.0,5.5,'DSI > 0.8 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
    ax[0,0].set_ylim(-0.5,5.75)
    ax[0,0].set_ylabel(r'$ \overline{g_{Inh}}$ [nA]')
    #ax[0,0].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_null_all_high_oR_sumTB[:,b]
        ax[1,0].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[1,0].text(4.0,5.5,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
    ax[1,0].set_ylim(-0.5,5.75)
    ax[1,0].set_ylabel(r'$ \overline{g_{Inh}}$ [nA]')
    ax[1,0].set_xlabel(r'cos')
    #ax[1,0].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_pref_all_low_oR_sumTB[:,b]
        ax[0,1].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[0,1].text(4.0,5.5,'DSI > 0.8 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    ax[0,1].set_ylim(-0.5,5.75)
    #ax[0,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_null_all_low_oR_sumTB[:,b]
        ax[1,1].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[1,1].set_xticks(np.linspace(1,tm_bins-1,tm_bins-1))
    ax[1,1].set_xticklabels(tm_labels)
    #ax[1,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    ax[1,1].text(4.0,5.5,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    ax[1,1].set_xlabel(r'cos')
    ax[1,1].set_ylim(-0.5,5.75)
    plt.savefig('Sum_Inh_overTM_box',dpi=300,bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plotData()

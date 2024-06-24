import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def plotDSI(strf):
    model_list = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']
    dir_list = ['0p3_50ms','0p5_30ms','0p5_50ms','0p5_70ms','0p7_50ms','lagged_noInh','nolagged_Inh','nolagged_noInh','no_STRF']
    label_list = ['0p3_50ms','0p5_30ms','0p5_50ms','0p5_70ms','0p7_50ms','lagged noInh','noLagged','noLagged noInh','no STRF']
    color_list=['navy','steelblue','seagreen','tomato','maroon','peru','yellowgreen','gold','slategrey','black']
    n_dir = len(dir_list)
    n_model = len(model_list)


    if not os.path.exists('DSI_TC_basis_'+strf):
        os.mkdir('DSI_TC_basis_'+strf)


    dsi_E1_all = []
    dsi_path = 'dsi_TC_basis_'+strf+'_kim.npy'
    for m in range(n_model):
        dsi_E1 = []
        for d in range(n_dir):
        
            dsi = np.load(model_list[m]+'/'+dir_list[d]+'/work/'+dsi_path)
            dsi_E1.append(dsi) # only one contrast level at the moment (!)

        dsi_E1_all.append(dsi_E1)

    dsi_E1_all = np.asarray(dsi_E1_all)

    print(np.shape(dsi_E1_all)) 
 
    
    ######
    ######
    # mean dsi over complete population

    best_dsi_0p3_50ms   = dsi_E1_all[:,0].flatten()
    best_dsi_0p5_30ms   = dsi_E1_all[:,1].flatten()
    best_dsi_0p5_50ms   = dsi_E1_all[:,2].flatten()
    best_dsi_0p5_70ms   = dsi_E1_all[:,3].flatten()
    best_dsi_0p7_50ms   = dsi_E1_all[:,4].flatten()
    best_dsi_lag_noIn   = dsi_E1_all[:,5].flatten()
    best_dsi_noLagged   = dsi_E1_all[:,6].flatten()
    best_dsi_nol_noIn   = dsi_E1_all[:,7].flatten()
    best_dsi_0p5_noSTRF = dsi_E1_all[:,8].flatten()
    #best_dsi_noRGC      = dsi_E1_all[:,9].flatten()

    ### get rid of nan values

    best_dsi_0p3_50ms   = best_dsi_0p3_50ms[~np.isnan(best_dsi_0p3_50ms)]
    best_dsi_0p5_30ms   = best_dsi_0p5_30ms[~np.isnan(best_dsi_0p5_30ms)]
    best_dsi_0p5_50ms   = best_dsi_0p5_50ms[~np.isnan(best_dsi_0p5_50ms)]
    best_dsi_0p5_70ms   = best_dsi_0p5_70ms[~np.isnan(best_dsi_0p5_70ms)]
    best_dsi_0p7_50ms   = best_dsi_0p7_50ms[~np.isnan(best_dsi_0p7_50ms)]
    best_dsi_lag_noIn   = best_dsi_lag_noIn[~np.isnan(best_dsi_lag_noIn)]
    best_dsi_noLagged   = best_dsi_noLagged[~np.isnan(best_dsi_noLagged)]
    best_dsi_nol_noIn   = best_dsi_nol_noIn[~np.isnan(best_dsi_nol_noIn)]
    best_dsi_0p5_noSTRF = best_dsi_0p5_noSTRF[~np.isnan(best_dsi_0p5_noSTRF)]
    #best_dsi_noRGC      = best_dsi_noRGC[~np.isnan(best_dsi_noRGC)]


    maxy = 1000#160*5

    props= dict(boxstyle='round', facecolor='white', alpha=0.5)

    plt.figure(figsize=(5,3))
    plt.title('50%, 50ms')
    plt.hist(best_dsi_0p5_50ms, label=label_list[2],fill=True, lw=0, color=color_list[2], alpha=0.5)
    plt.grid(which='both',linestyle='-.')
    plt.vlines(np.mean(best_dsi_0p5_50ms), 0,250, 'black')
    plt.xlabel('DSI')
    plt.ylabel('number of cells')
    plt.text(0.8,250,'mean = %.2f'%(np.mean(best_dsi_0p5_50ms)), bbox=props)
    plt.savefig('DSI_TC_basis_'+strf+'/TC_hist_50ms',dpi=300,bbox_inches='tight')



    ## plot a cumulative plot
    print(np.shape(best_dsi_0p5_50ms))

    h,b = np.histogram(best_dsi_0p5_50ms,20)
    total = np.sum(h)
    acu = np.zeros(len(b)-1)
    for i in range(len(b)-1):
       acu[i] = np.sum(h[0:i]) 

    dsi_fbI = []
    dsi_fwI = []
    dsi_shI = []
    
    for m in range(n_model):
        data = np.load(model_list[m]+'/shuffle_FBInhib/work/'+dsi_path)
        dsi_fbI.append(data)
        data = np.load(model_list[m]+'/shuffle_FWInhib/work/'+dsi_path)
        dsi_fwI.append(data)
        data = np.load(model_list[m]+'/shuffleInhib_complete/work/'+dsi_path)
        dsi_shI.append(data)



    ### make some nice bars and violins
    print(np.shape(dsi_fbI))
    best_dsi_0p3_50ms_bars   = dsi_E1_all[:,0]
    best_dsi_0p5_30ms_bars   = dsi_E1_all[:,1]
    best_dsi_0p5_50ms_bars   = dsi_E1_all[:,2]
    best_dsi_0p5_70ms_bars   = dsi_E1_all[:,3]
    best_dsi_0p7_50ms_bars   = dsi_E1_all[:,4]
    best_dsi_lag_noIn_bars   = dsi_E1_all[:,5]
    best_dsi_noLagged_bars   = dsi_E1_all[:,6]
    best_dsi_nol_noIn_bars   = dsi_E1_all[:,7]
    best_dsi_0p5_noSTRF_bars = dsi_E1_all[:,8]



    n_bins = 11
    dsi_bins = np.linspace(0,1,n_bins)
    _, n_cells = np.shape(best_dsi_0p5_50ms_bars)
    
    n_cells_bin_0p5_50ms = np.zeros((n_bins-1,n_model))
    n_cells_bin_noSTRF   = np.zeros((n_bins-1,n_model))
    n_cells_bin_noL_noIn = np.zeros((n_bins-1,n_model))
    n_cells_bin_lag_noIn = np.zeros((n_bins-1,n_model))
    n_cells_noLagg       = np.zeros((n_bins-1,n_model))

    n_cells_bin_0p3_50ms = np.zeros((n_bins-1,n_model))
    n_cells_bin_0p5_30ms = np.zeros((n_bins-1,n_model))
    n_cells_bin_0p5_70ms = np.zeros((n_bins-1,n_model))
    n_cells_bin_0p7_50ms = np.zeros((n_bins-1,n_model))


    n_cells_bin_fbI = np.zeros((n_bins-1,n_model))
    n_cells_bin_fwI = np.zeros((n_bins-1,n_model))
    n_cells_bin_shI = np.zeros((n_bins-1,n_model))

    for m in range(n_model):
        for b in range(n_bins-2):
            idx = np.where((best_dsi_0p5_50ms_bars[m]>=dsi_bins[b]) & (best_dsi_0p5_50ms_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_0p5_50ms[b,m] = (len(idx)/n_cells)*100


            idx = np.where((best_dsi_0p3_50ms_bars[m]>=dsi_bins[b]) & (best_dsi_0p3_50ms_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_0p3_50ms[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_0p5_30ms_bars[m]>=dsi_bins[b]) & (best_dsi_0p5_30ms_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_0p5_30ms[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_0p5_70ms_bars[m]>=dsi_bins[b]) & (best_dsi_0p5_70ms_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_0p5_70ms[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_0p7_50ms_bars[m]>=dsi_bins[b]) & (best_dsi_0p7_50ms_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_0p7_50ms[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_0p5_noSTRF_bars[m]>=dsi_bins[b]) & (best_dsi_0p5_noSTRF_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_noSTRF[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_nol_noIn_bars[m]>=dsi_bins[b]) & (best_dsi_nol_noIn_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_noL_noIn[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_lag_noIn_bars[m]>=dsi_bins[b]) & (best_dsi_lag_noIn_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_lag_noIn[b,m] = (len(idx)/n_cells)*100

            idx = np.where((best_dsi_noLagged_bars[m]>=dsi_bins[b]) & (best_dsi_noLagged_bars[m]<dsi_bins[b+1]) )[0]
            n_cells_noLagg[b,m] = (len(idx)/n_cells)*100


            idx = np.where((dsi_fbI[m]>=dsi_bins[b]) & (dsi_fbI[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_fbI[b,m] = (len(idx)/n_cells)*100

            idx = np.where((dsi_fwI[m]>=dsi_bins[b]) & (dsi_fwI[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_fwI[b,m] = (len(idx)/n_cells)*100

            idx = np.where((dsi_shI[m]>=dsi_bins[b]) & (dsi_shI[m]<dsi_bins[b+1]) )[0]
            n_cells_bin_shI[b,m] = (len(idx)/n_cells)*100


        ## make the last bin extra
        idx = np.where(best_dsi_0p5_50ms_bars[m]>=0.9 )[0]
        n_cells_bin_0p5_50ms[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_0p3_50ms_bars[m]>=0.9 )[0]
        n_cells_bin_0p3_50ms[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_0p5_30ms_bars[m]>=0.9 )[0]
        n_cells_bin_0p5_30ms[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_0p5_70ms_bars[m]>=0.9 )[0]
        n_cells_bin_0p5_70ms[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_0p7_50ms_bars[m]>=0.9 )[0]
        n_cells_bin_0p7_50ms[-1,m] = (len(idx)/n_cells)*100



        idx = np.where(best_dsi_0p5_noSTRF_bars[m]>=0.9 )[0]
        n_cells_bin_noSTRF[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_nol_noIn_bars[m]>=0.9 )[0]
        n_cells_bin_noL_noIn[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_lag_noIn_bars[m]>=0.9 )[0]
        n_cells_bin_lag_noIn[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(best_dsi_noLagged_bars[m]>=0.9 )[0]
        n_cells_noLagg[-1,m] = (len(idx)/n_cells)*100




        idx = np.where(dsi_fbI[m]>=0.9 )[0]
        n_cells_bin_fbI[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(dsi_fwI[m]>=0.9 )[0]
        n_cells_bin_fwI[-1,m] = (len(idx)/n_cells)*100

        idx = np.where(dsi_shI[m]>=0.9 )[0]
        n_cells_bin_shI[-1,m] = (len(idx)/n_cells)*100

    label_list = np.linspace(0.0,1,11)
    print(label_list)
    y_labels = ['%.1f - %.1f'%(label_list[i],label_list[i+1]) for i in range(0,11,3)]
    print(y_labels)
    print(np.shape(n_cells_bin_0p5_50ms))
    ### bars ###
    x_bar = np.linspace(0,n_bins-2,n_bins-1, dtype='int32')
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(12, 3))
    ax[0].set_title("50 ms, 50%")
    ax[0].set_ylabel('DSI',fontsize=12)
    ax[0].barh(x_bar,np.mean(n_cells_bin_0p5_50ms,1), xerr = np.std(n_cells_bin_0p5_50ms,1,ddof=1),color = color_list[2], capsize = 2, alpha=0.75)
    ax[0].set_xlabel('% Cells',fontsize=12)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].set_yticks(np.linspace(0,9,4))   
    ax[0].set_yticklabels(y_labels)    

    ax[1].set_title("Shuffel feedback")
    ax[1].barh(x_bar,np.mean(n_cells_bin_fbI,1), xerr = np.std(n_cells_bin_fbI,1,ddof=1),color = color_list[5], capsize = 2, alpha=0.75)
    ax[1].set_xlabel('% Cells',fontsize=12)
    ax[1].grid(which='both',linestyle='-.')

    ax[2].set_title("Shuffle feedforward")
    ax[2].barh(x_bar,np.mean(n_cells_bin_fwI,1), xerr = np.std(n_cells_bin_fwI,1,ddof=1),color = color_list[6], capsize = 2, alpha=0.75)
    ax[2].set_xlabel('% Cells',fontsize=12)
    ax[2].grid(which='both',linestyle='-.')

    ax[3].set_title("Shuffle all")
    ax[3].barh(x_bar,np.mean(n_cells_bin_shI,1), xerr = np.std(n_cells_bin_shI,1,ddof=1),color = color_list[7], capsize = 2, alpha=0.75)
    ax[3].set_xlabel('% Cells',fontsize=12)
    ax[3].grid(which='both',linestyle='-.')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_shuffle_bar.png',bbox_inches='tight',dpi=300)
    plt.close()



    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12, 3))
    ax[0].set_title('no STRF')
    ax[0].set_ylabel('DSI',fontsize=12)
    ax[0].barh(x_bar,np.mean(n_cells_bin_noSTRF,1), xerr = np.std(n_cells_bin_noSTRF,1,ddof=1),color = color_list[-2], capsize = 2, alpha=0.75)
    ax[0].set_xlabel('% Cells',fontsize=12)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].set_yticks(np.linspace(0,9,4))   
    ax[0].set_yticklabels(y_labels)    

    ax[1].set_title('STRF')
    ax[1].barh(x_bar,np.mean(n_cells_bin_noL_noIn,1), xerr = np.std(n_cells_bin_noL_noIn,1,ddof=1),color = color_list[7], capsize = 2, alpha=0.75)
    ax[1].set_xlabel('% Cells',fontsize=12)
    ax[1].grid(which='both',linestyle='-.')

    ax[2].set_title('STRF, lagged cells')
    ax[2].barh(x_bar,np.mean(n_cells_bin_lag_noIn,1), xerr = np.std(n_cells_bin_lag_noIn,1,ddof=1),color = color_list[5], capsize = 2, alpha=0.75)
    ax[2].set_xlabel('% Cells',fontsize=12)
    ax[2].grid(which='both',linestyle='-.')

    ax[3].set_title('STRF, inhibition')
    ax[3].barh(x_bar,np.mean(n_cells_noLagg,1), xerr = np.std(n_cells_noLagg,1,ddof=1),color = color_list[6], capsize = 2, alpha=0.75)
    ax[3].set_xlabel('% Cells',fontsize=12)
    ax[3].grid(which='both',linestyle='-.')

    ax[4].set_title('STRF, lagged, inhib')
    ax[4].barh(x_bar,np.mean(n_cells_bin_0p5_50ms,1), xerr = np.std(n_cells_bin_0p5_50ms,1,ddof=1),color = color_list[2], capsize = 2, alpha=0.75)
    ax[4].set_xlabel('% Cells',fontsize=12)
    ax[4].grid(which='both',linestyle='-.')

    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_Bar_noInhLagg_hor.png',bbox_inches='tight',dpi=300)
    plt.close()


    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(12, 3))
    ax[0].set_title('STRF')
    ax[0].set_ylabel('DSI',fontsize=12)
    ax[0].barh(x_bar,np.mean(n_cells_bin_noL_noIn,1), xerr = np.std(n_cells_bin_noL_noIn,1,ddof=1),color = color_list[7], capsize = 2, alpha=0.75)
    ax[0].set_xlabel('% Cells',fontsize=12)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].set_yticks(np.linspace(0,9,4))   
    ax[0].set_yticklabels(y_labels)    
    ax[1].set_title('STRF, lagged cells')
    ax[1].barh(x_bar,np.mean(n_cells_bin_lag_noIn,1), xerr = np.std(n_cells_bin_lag_noIn,1,ddof=1),color = color_list[5], capsize = 2, alpha=0.75)
    ax[1].set_xlabel('% Cells',fontsize=12)
    ax[1].grid(which='both',linestyle='-.')
    ax[2].set_title('STRF, inhibition')
    ax[2].barh(x_bar,np.mean(n_cells_noLagg,1), xerr = np.std(n_cells_noLagg,1,ddof=1),color = color_list[6], capsize = 2, alpha=0.75)
    ax[2].set_xlabel('% Cells',fontsize=12)
    ax[2].grid(which='both',linestyle='-.')
    ax[3].set_title('STRF, lagged, inhib')
    ax[3].barh(x_bar,np.mean(n_cells_bin_0p5_50ms,1), xerr = np.std(n_cells_bin_0p5_50ms,1,ddof=1),color = color_list[2], capsize = 2, alpha=0.75)
    ax[3].set_xlabel('% Cells',fontsize=12)
    ax[3].grid(which='both',linestyle='-.')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_Bar_noInhLagg_hor_noNOSTRF.png',bbox_inches='tight',dpi=300)
    plt.close()

    max_x = 20
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle('50% lagged cells', fontsize=15)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].barh(x_bar,np.mean(n_cells_bin_0p5_30ms,1), xerr = np.std(n_cells_bin_0p5_30ms,1,ddof=1),color = color_list[1], capsize = 2, alpha=0.75)
    ax[0].set_xlabel('% Cell',fontsize=12)
    ax[0].set_yticks(np.linspace(0,9,4))   
    ax[0].set_yticklabels(y_labels)    
    ax[0].set_xlim(0,max_x)
    ax[0].set_title('30ms')
    ax[1].grid(which='both',linestyle='-.')
    ax[1].barh(x_bar,np.mean(n_cells_bin_0p5_50ms,1), xerr = np.std(n_cells_bin_0p5_50ms,1,ddof=1),color = color_list[2], capsize = 2, alpha=0.75)
    ax[1].set_xlabel('% Cell',fontsize=12)
    ax[1].set_yticks(np.linspace(0,9,4))   
    ax[1].set_yticklabels(y_labels)    
    ax[1].set_xlim(0,max_x)
    ax[1].set_title('50ms')
    ax[2].grid(which='both',linestyle='-.')
    ax[2].barh(x_bar,np.mean(n_cells_bin_0p5_70ms,1), xerr = np.std(n_cells_bin_0p5_70ms,1,ddof=1),color = color_list[3], capsize = 2, alpha=0.75)
    ax[2].set_xlabel('% Cell',fontsize=12)
    ax[2].set_yticks(np.linspace(0,9,4))   
    ax[2].set_yticklabels(y_labels)    
    ax[2].set_xlim(0,max_x)
    ax[2].set_title('70ms')
    ax[0].set_ylabel('DSI',fontsize=12)
    #fig.text(0.5, 0.0, r'$g_{Exc} [nA]$ ', ha='center')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_Bar_0p5_hor.png',bbox_inches='tight',dpi=300)
    plt.close()  


    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle('50 ms offset', fontsize=15)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].barh(x_bar,np.mean(n_cells_bin_0p3_50ms,1), xerr = np.std(n_cells_bin_0p5_30ms,1,ddof=1),color = color_list[0], capsize = 2, alpha=0.75)
    ax[0].set_xlabel('% Cell',fontsize=12)
    ax[0].set_yticks(np.linspace(0,9,4))   
    ax[0].set_yticklabels(y_labels)    
    ax[0].set_xlim(0,max_x)
    ax[0].set_title('30%')
    ax[1].grid(which='both',linestyle='-.')
    ax[1].barh(x_bar,np.mean(n_cells_bin_0p5_50ms,1), xerr = np.std(n_cells_bin_0p5_50ms,1,ddof=1),color = color_list[2], capsize = 2, alpha=0.75)
    ax[1].set_xlabel('% Cell',fontsize=12)
    ax[1].set_yticks(np.linspace(0,9,4))   
    ax[1].set_yticklabels(y_labels)    
    ax[1].set_xlim(0,max_x)
    ax[1].set_title('50%')
    ax[2].grid(which='both',linestyle='-.')
    ax[2].barh(x_bar,np.mean(n_cells_bin_0p7_50ms,1), xerr = np.std(n_cells_bin_0p5_70ms,1,ddof=1),color = color_list[4], capsize = 2, alpha=0.75)
    ax[2].set_xlabel('% Cell',fontsize=12)
    ax[2].set_yticks(np.linspace(0,9,4))   
    ax[2].set_yticklabels(y_labels)    
    ax[2].set_xlim(0,max_x)
    ax[2].set_title('70%')
    ax[0].set_ylabel('DSI',fontsize=12)
    #fig.text(0.5, 0.0, r'$g_{Exc} [nA]$ ', ha='center')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_Bar_50ms_hor.png',bbox_inches='tight',dpi=300)
    plt.close()  


    dsi_fbI = np.asarray(dsi_fbI).flatten()
    dsi_fwI = np.asarray(dsi_fwI).flatten()
    dsi_shI = np.asarray(dsi_shI).flatten()
    print(np.shape(dsi_fbI))
    
    #### another point of view ####
    max_x = 300#400
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle('50 ms offset', fontsize=15)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].hist(best_dsi_0p3_50ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[0], color=color_list[0],alpha=0.5)
    ax[0].set_xlabel('Nbr cells',fontsize=12)
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,max_x)
    ax[0].set_title("30%")
    ax[1].grid(which='both',linestyle='-.')
    ax[1].hist(best_dsi_0p5_50ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[2], color=color_list[2],alpha=0.5)
    ax[1].set_xlabel('Nbr cells',fontsize=12)
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,max_x)
    ax[1].set_title("50%")
    ax[2].grid(which='both',linestyle='-.')
    ax[2].hist(best_dsi_0p7_50ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[4], color=color_list[4],alpha=0.5)
    ax[2].set_xlabel('Nbr cells',fontsize=12)
    ax[2].set_ylim(0,1)
    ax[2].set_title("70%")
    ax[2].set_xlim(0,max_x)
    ax[0].set_ylabel('DSI',fontsize=12)
    #fig.text(0.5, 0.0, r'$g_{Exc} [nA]$ ', ha='center')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_hist_50ms_hor.png',bbox_inches='tight',dpi=300)
    plt.close()  

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle('50% lagged cells', fontsize=15)
    ax[0].grid(which='both',linestyle='-.')
    ax[0].hist(best_dsi_0p5_30ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[1], color=color_list[1],alpha=0.5)
    ax[0].set_xlabel('Nbr cells',fontsize=12)
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,max_x)
    ax[0].set_title('30ms')
    ax[1].grid(which='both',linestyle='-.')
    ax[1].hist(best_dsi_0p5_50ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[2], color=color_list[2],alpha=0.5)
    ax[1].set_xlabel('Nbr cells',fontsize=12)
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,max_x)
    ax[1].set_title('50ms')
    ax[2].grid(which='both',linestyle='-.')
    ax[2].hist(best_dsi_0p5_70ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[3], color=color_list[3],alpha=0.5)
    ax[2].set_xlabel('Nbr cells',fontsize=12)
    ax[2].set_ylim(0,1)
    ax[2].set_xlim(0,max_x)
    ax[2].set_title('70ms')
    ax[0].set_ylabel('DSI',fontsize=12)
    #fig.text(0.5, 0.0, r'$g_{Exc} [nA]$ ', ha='center')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_hist_0p5_hor.png',bbox_inches='tight',dpi=300)
    plt.close()  


    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12, 4))
    ax[0].grid(which='both',linestyle='-.') 
    ax[0].hist(best_dsi_0p5_noSTRF, orientation='horizontal',bins=10, range=(0,1), label='no STRF', color=color_list[-2],alpha=0.5)
    ax[0].set_xlabel('Nbr cells',fontsize=12)
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,max_x+350)
    ax[0].set_title('no STRF')
    ax[1].grid(which='both',linestyle='-.') 
    ax[1].hist(best_dsi_nol_noIn, orientation='horizontal',bins=10, range=(0,1), label='STRF, no lagged, no inhib', color=color_list[7],alpha=0.5)
    ax[1].set_xlabel('Nbr cells',fontsize=12)
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,max_x+350)
    ax[1].set_title('STRF')
    ax[2].grid(which='both',linestyle='-.')
    ax[2].hist(best_dsi_lag_noIn, orientation='horizontal',bins=10, range=(0,1), label='STRF, lagged, no inhib', color=color_list[5],alpha=0.5)
    ax[2].set_xlabel('Nbr cells',fontsize=12)
    ax[2].set_ylim(0,1)
    ax[2].set_xlim(0,max_x+350)
    ax[2].set_title('STRF, lagged cells')
    ax[3].grid(which='both',linestyle='-.')
    ax[3].hist(best_dsi_noLagged, orientation='horizontal',bins=10, range=(0,1), label='STRF, no lagged, inhib', color=color_list[6],alpha=0.5)
    ax[3].set_xlabel('Nbr cells',fontsize=12)
    ax[3].set_ylim(0,1)
    ax[3].set_xlim(0,max_x+350)
    ax[3].set_title('STRF, inhibition')
    ax[4].grid(which='both',linestyle='-.')
    ax[4].hist(best_dsi_0p5_50ms, orientation='horizontal',bins=10, range=(0,1), label='STRF, lagged, inhib', color=color_list[2],alpha=0.5)
    ax[4].set_xlabel('Nbr cells',fontsize=12)
    ax[4].set_ylim(0,1)
    ax[4].set_xlim(0,max_x+350)
    ax[4].set_title('STRF, lagged, inhib')


    ax[0].set_ylabel('DSI',fontsize=12)
    #fig.text(0.5, 0.0, r'$g_{Exc} [nA]$ ', ha='center')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_hist_noInhLagg_hor_2.png',bbox_inches='tight',dpi=300)
    plt.close()  

    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(12, 4))
    ax[0].grid(which='both',linestyle='-.')
    ax[0].hist(best_dsi_0p5_50ms, orientation='horizontal',bins=10, range=(0,1), label=label_list[2], color=color_list[2],alpha=0.5)
    ax[0].set_xlabel('Nbr cells',fontsize=12)
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,max_x)
    ax[0].set_title("50%, 50ms")

    ax[1].grid(which='both',linestyle='-.')
    ax[1].hist(dsi_fbI, orientation='horizontal',bins=10, range=(0,1), label='shuffled fBInh', color=color_list[5],alpha=0.5)
    ax[1].set_xlabel('Nbr cells',fontsize=12)
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,max_x)
    ax[1].set_title('shuffled fBInh')

    ax[2].grid(which='both',linestyle='-.')
    ax[2].hist(dsi_fwI, orientation='horizontal',bins=10, range=(0,1), label='shuffled ffWInh', color=color_list[6],alpha=0.5)
    ax[2].set_xlabel('Nbr cells',fontsize=12)
    ax[2].set_ylim(0,1)
    ax[2].set_xlim(0,max_x)
    ax[2].set_title('shuffled ffWInh')

    ax[3].grid(which='both',linestyle='-.')
    ax[3].hist(dsi_shI, orientation='horizontal',bins=10, range=(0,1), label='shuffled Inh', color=color_list[7],alpha=0.5)
    ax[3].set_xlabel('Nbr cells',fontsize=12)
    ax[3].set_ylim(0,1)
    ax[3].set_xlim(0,max_x)
    ax[3].set_title('shuffled Inh')
    ax[0].set_ylabel('DSI',fontsize=12)
    #fig.text(0.5, 0.0, r'$g_{Exc} [nA]$ ', ha='center')
    plt.savefig('DSI_TC_basis_'+strf+'/TC_bestDSI_shuffle_hor.png',bbox_inches='tight',dpi=300)
    plt.close()  


if __name__ == '__main__':
    
    strf='STRF'
    plotDSI(strf)

    strf='noSTRF'
    plotDSI(strf)


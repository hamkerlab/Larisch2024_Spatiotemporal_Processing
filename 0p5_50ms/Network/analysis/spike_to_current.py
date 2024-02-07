import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm 
from scipy import signal

def getDeltaT(data,t_win):
    lags = signal.correlation_lags(t_win+1, t_win+1)
    idx_zero = np.where(lags==0)[0]

    delta_T = 0
    t_w = 3
    delta_T_found = False
    while delta_T_found == False:
        ## look in the are between [-delta_T, + delta_T] where is there a maximum
        ## increase delta_T step by step
        ## the search is over, if the value "behind" or "next" to delta_T (so -delta_T-1 or +delta_T+1) lower then the value at delta_T
        area_start = data[int(idx_zero-delta_T) : int(idx_zero+delta_T+1) ]
        area_new = data[int(idx_zero-(delta_T+t_w)) : int(idx_zero+(delta_T+t_w)+1) ]
        if np.max(area_start) >= np.max(area_new):
            delta_T_found = True
            arg_m = np.argmax(area_start)
            area_lags = lags[int(idx_zero-delta_T) : int(idx_zero+delta_T+1) ]
            final_lag = area_lags[arg_m]
        else:
            delta_T+=1
    #print(final_lag)
    return(final_lag)

def calcCrossCorr(x,y,t_win):
    # calculate the crosscorrelation for a specific time window
    crossCorr = np.zeros(t_win*2+1)

    c = 0
    for i in range(t_win,0,-1): # first 100 shifts
        y_calc = y[i:]
        x_calc = x[:len(y_calc)]
        crossCorr[c] = np.corrcoef(x_calc,y_calc)[0,1]
        c+=1
    ## identic ##
    crossCorr[c] = np.corrcoef(x,y)[0,1]
    c+=1

    for i in range(1,t_win+1):
        x_calc = x[i:] 
        y_calc = y[:len(x_calc)]
        crossCorr[c] = np.corrcoef(x_calc,y_calc)[0,1]
        c+=1

    return(crossCorr)

## load only from "interesting" cells the data
def loadTime_Data(cell_pref, idx_cells):

    print('Load the Data for each time step')
    
    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    total_t   = sim_param['totalT']
    n_degree = sim_param['n_degree']

    n_amp = 1
    a = 0
    repeats = 5
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)

    n_cells = np.shape(cell_pref)[1]
    n_cells_I = int(n_cells/4)

    spk_E = np.zeros((len(idx_cells),n_degree,repeats,total_t))
    gEx_E = np.zeros((len(idx_cells),n_degree,repeats,total_t))
    gIn_E = np.zeros((len(idx_cells),n_degree,repeats,total_t))
    
    spk_I = np.zeros((len(idx_cells),n_degree,repeats,n_cells_I,total_t))

    ## iterate over all cells
    for c in tqdm(range(len(idx_cells)), ncols=80):
        speed = int(cell_pref[0,idx_cells[c],0])
        spat_F = int(cell_pref[0,idx_cells[c],1])

        ## load the spike times only for the preff temp_F and spat_F
        spk_time =  np.load('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i.npy'%(a, speed, spat_F))
        spk_E[c] = spk_time[:,:,idx_cells[c],:]


        ## load the corresponding input currents
        gEx =  np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(a,speed,spat_F))
        gEx_E[c] = gEx[:,:,:,idx_cells[c]]

        gIn =  np.load('./work/directGrating_Sinus_gInh_E1_amp%i_spatF%i_tempF%i.npy'%(a,speed,spat_F))    
        gIn_E[c] = gIn[:,:,:,idx_cells[c]]

        ## load all inhibory spike times corresponding to each excitatory neuron
        spk_time = np.load('./work/directGrating_Sinus_SpikeTimes_I1_amp%i_spatF%i_tempF%i.npy'%(a,speed,spat_F))
        spk_time = spk_time[:,:,0:n_cells_I,:]
        spk_I[c] = spk_time



    return(spk_E, gEx_E, gIn_E, spk_I)

def analyzeDSI(preff_E1, idx_cell, kind):

    ### load the feedback weights
    fb_w = np.loadtxt('./Input_network/INtoV1.txt')
    #fb_w = fb_w/np.max(fb_w)
    fb_w = fb_w[idx_cell]


    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']
    total_T = sim_param['totalT']


    temp_E_I = np.load('./work/DSI_RF_tempE1_I1.npy')
    temp_E_I = temp_E_I[idx_cell]

    spk_E, gEx_E, gIn_E, spk_I = loadTime_Data(preff_E1, idx_cell)

    print(np.shape(spk_E), np.shape(gEx_E), np.shape(spk_I))


    step_degree = int(360/n_steps)

    ## NOTE: it makes no sense to calculate the mean over spike times!
    ## calculate the cross correlation between the spike times

    n_cells = len(spk_E)    
    n_Icells = np.shape(spk_I)[3]

    corr_w = 50
    repeats = 5

    tm_bins = 6
    tm_bin_list = np.linspace(-0.8,0.8,tm_bins)

    t_kickOff = 500

    t_bin_size = 2#5#10
    n_bins = int( (total_T-t_kickOff)/t_bin_size)

    crossCorr_gE_gI_pref = np.zeros((n_cells, 1+(corr_w*2)))
    crossCorr_gE_gI_null = np.zeros((n_cells, 1+(corr_w*2)))

    peak_gE_gI_pref = np.zeros(n_cells)
    peak_gE_gI_null = np.zeros(n_cells)


    crossCorr_E_I_pref = np.zeros((n_cells, n_Icells, 1+(corr_w*2)))
    crossCorr_E_I_null = np.zeros((n_cells, n_Icells, 1+(corr_w*2)))

    peak_crossCorr_pref = np.zeros((n_cells, n_Icells))
    peak_crossCorr_null = np.zeros((n_cells, n_Icells))

    max_crossCorr_gInh_TM = np.zeros((repeats,n_cells, tm_bins-1))


    tm_nSpk_pref_all= np.zeros((n_cells, repeats, n_bins, tm_bins-1))    
    tm_nSpk_null_all= np.zeros((n_cells, repeats, n_bins, tm_bins-1))    
    

    gEx_E_pref_bins_all = np.zeros((n_cells, repeats, n_bins))      
    gEx_E_null_bins_all = np.zeros((n_cells, repeats, n_bins))

    gIn_E_pref_bins_all = np.zeros((n_cells, repeats, n_bins))      
    gIn_E_null_bins_all = np.zeros((n_cells, repeats, n_bins))


    n_cells_TM_bin = np.zeros((n_cells,tm_bins-1))

    for c in tqdm(range(n_cells),ncols=80):
        ## get the preferred direction
        preff_arg = int(preff_E1[0,idx_cell[c],2])
        ## convert in degree to calculate the null direction 
        preff_D = preff_arg*step_degree
        null_D = (preff_D+180)%360
        ## calculate the index of null direction
        null_arg = int(null_D/step_degree)

        ## get the spike train for pref and null direction
        spk_E_pref = spk_E[c,preff_arg,:,t_kickOff:]
        spk_E_null = spk_E[c,null_arg,:,t_kickOff:]
        

        ## get the corresponding currents
        gEx_E_pref = gEx_E[c,preff_arg,:,t_kickOff:]
        gEx_E_null = gEx_E[c,null_arg,:,t_kickOff:]

        gIn_E_pref = gIn_E[c,preff_arg,:,t_kickOff:]
        gIn_E_null = gIn_E[c,null_arg,:,t_kickOff:]

        ## get the corresponding inhibitory spike times
        spk_I_pref = spk_I[c,preff_arg,:,:,t_kickOff:]
        spk_I_null = spk_I[c,null_arg,:,:,t_kickOff:]

        cell_gIn_pref = np.mean(gIn_E_pref,axis=0) # first axis is repetition?

        x = np.linspace(0,len(cell_gIn_pref),len(cell_gIn_pref))
        
        """
        plt.figure()
        plt.plot(x,cell_gIn_pref)
        for i in range(n_Icells):
            plt.scatter(x, spk_I_pref[0,i])
        plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/gInh_C_%i.png'%(c))
        plt.close()
        """

        ## create time bins and look, how many inhibitory neurons spiked in each bin
        ## and sort it over template match

       
        tm_nSpk_pref = np.zeros((repeats, n_bins, tm_bins-1))        
        tm_nSpk_null = np.zeros((repeats, n_bins, tm_bins-1))    

        ## create a binned version of the gExc and gInh
        gEx_E_pref_bins = np.zeros((repeats, n_bins))   
        gEx_E_null_bins = np.zeros((repeats, n_bins))   

        gIn_E_pref_bins = np.zeros((repeats, n_bins))   
        gIn_E_null_bins = np.zeros((repeats, n_bins))   

        ## iterate over the N ms long time bins!
        for tb in range(n_bins):    
            for r in range(repeats):  
                ## for the pref orientation
                # get the spikes in the time frame
                spT_pref = spk_I_pref[r,:,0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)]
                ## sum over the time to know, how mutch each inhibitory neuron spiked
                sum_spT_pref = np.sum(spT_pref,axis=1)
                idx_zero = np.where(sum_spT_pref == 0)[0]

                ## for the null orientation
                spT_null = spk_I_null[r,:,0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)]
                sum_spT_null = np.sum(spT_null,axis=1)

                ## now iterate over the tamplate match bins
                for tm_b in range(len(tm_bin_list)-1):
                    idx = np.where((temp_E_I[c] >= tm_bin_list[tm_b]) & (temp_E_I[c] < tm_bin_list[tm_b+1]))[0]
                    #print(idx)
                    n_cells_TM_bin[c,tm_b] = len(idx) # save the number of cells with the specific template match in relation to the ecxitatory cell -> independent from the input !
                    tm_nSpk_pref[r,tb,tm_b] = np.sum(sum_spT_pref[idx]*fb_w[c,idx])# ) ### weight with the feedback-synapse between them    
                    tm_nSpk_null[r,tb,tm_b] = np.sum(sum_spT_null[idx]*fb_w[c,idx])#)  ### weight with the feedback-synapse between them    

                gEx_E_pref_bins[r,tb] = np.mean(gEx_E_pref[r, 0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)])
                gEx_E_null_bins[r,tb] = np.mean(gEx_E_null[r, 0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)])

                gIn_E_pref_bins[r,tb] = np.mean(gIn_E_pref[r, 0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)])
                gIn_E_null_bins[r,tb] = np.mean(gIn_E_null[r, 0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)])


        tm_nSpk_pref_all[c] = tm_nSpk_pref
        tm_nSpk_null_all[c] = tm_nSpk_null      

        gEx_E_pref_bins_all[c] = gEx_E_pref_bins
        gEx_E_null_bins_all[c] = gEx_E_null_bins  

        gIn_E_pref_bins_all[c] = gIn_E_pref_bins
        gIn_E_null_bins_all[c] = gIn_E_null_bins

        """
        fig,axs = plt.subplots(2, figsize=(50,5))
        img_0 = axs[0].imshow(tm_nSpk_pref[0].T)
        plt.colorbar(img_0, ax=axs[0])
        img_1 = axs[1].imshow(tm_nSpk_null[0].T)
        plt.colorbar(img_1, ax=axs[1])
        plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/time_TM_spike_Sum_%i.png'%(c))
        plt.close()
        


        fig,axs = plt.subplots(2, figsize=(50,5))
        img_0 = axs[0].imshow(np.mean(tm_nSpk_pref,axis=0).T)
        plt.colorbar(img_0, ax=axs[0])
        img_1 = axs[1].imshow(np.mean(tm_nSpk_null,axis=0).T)
        plt.colorbar(img_1, ax=axs[1])
        plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/time_TM_spike_Sum_meanOR_%i.png'%(c))
        plt.close()
        """
        #print(np.shape(tm_nSpk_pref))

        ### calculate the summed inhibitory input between pref and null for each TM-bin
        crossCorr_gInh_TM = np.zeros((repeats,len(tm_bin_list)-1, 1+(corr_w*2) ))
        for r in range(repeats):
            for b in range(len(tm_bin_list)-1):
                ## check if one the both sums are empty if not calculate the CrossCorr
                if np.sum(tm_nSpk_pref[r,:,b]) == 0:
                    continue
                if np.sum(tm_nSpk_null[r,:,b]) == 0:
                    continue
                else:
                    crossCorr_gInh_TM[r,b] = calcCrossCorr(tm_nSpk_pref[r,:,b], tm_nSpk_null[r,:,b], corr_w )
                    max_crossCorr_gInh_TM[r,c,b] = np.argmax(crossCorr_gInh_TM[r,b]) - corr_w
        """
        fig,axs = plt.subplots(len(tm_bin_list)-1, figsize=(10,20))
        for i in range(len(tm_bin_list)-1):
            axs[i].plot(np.mean(crossCorr_gInh_TM[:,i],axis=0))
        plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/CrossCor_TM_bin_%i.png'%(c))
        plt.close()
        """
        ## calculate some cross correlations, just for fun...
        for r in range(repeats):
            crossCorr_gE_gI_pref += calcCrossCorr(gEx_E_pref[r]/np.max(gEx_E_pref[r]), gIn_E_pref[r]/np.max(gIn_E_pref[r]), corr_w)
            crossCorr_gE_gI_null += calcCrossCorr(gEx_E_null[r]/np.max(gEx_E_null[r]), gIn_E_null[r]/np.max(gIn_E_null[r]), corr_w)

        crossCorr_gE_gI_pref /= repeats
        crossCorr_gE_gI_null /= repeats

        peak_gE_gI_pref[c] = np.argmax(crossCorr_gE_gI_pref) - corr_w
        peak_gE_gI_null[c] = np.argmax(crossCorr_gE_gI_null) - corr_w


    np.save('./work/direct_tm_nSpk_pref_all_'+kind, tm_nSpk_pref_all)
    np.save('./work/direct_tm_nSpk_null_all_'+kind, tm_nSpk_null_all)

    np.save('./work/direct_gEx_E_pref_bins_all_'+kind, gEx_E_pref_bins_all)
    np.save('./work/direct_gEx_E_null_bins_all_'+kind, gEx_E_null_bins_all)

    np.save('./work/direct_gIn_E_pref_bins_all_'+kind, gIn_E_pref_bins_all)
    np.save('./work/direct_gIn_E_null_bins_all_'+kind, gIn_E_null_bins_all)

    np.save('./work/direct_cellsTM_'+kind, n_cells_TM_bin)

    plt.figure()
    plt.bar([0,1],[np.median(peak_gE_gI_pref),np.median(peak_gE_gI_null)] )
    plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/CrossCorr_currents.png')
    plt.close()
   
    plt.figure()
    plt.bar(np.linspace(0,tm_bins-2,tm_bins-1),np.mean(max_crossCorr_gInh_TM,axis=(0,1)))
    plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/CrossCorr_TM_mean_currents.png')
    plt.close()

    plt.figure()
    plt.bar(np.linspace(0,tm_bins-2,tm_bins-1),np.mean(np.abs(max_crossCorr_gInh_TM),axis=(0,1)))
    plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/CrossCorr_TM_mean_abs_currents.png')
    plt.close()

    tm_nSpk_pref_all_meanOR = np.mean(tm_nSpk_pref_all,axis=1)
    tm_nSpk_pref_all_meanOR_meanOC = np.mean(tm_nSpk_pref_all_meanOR,axis=0)

    tm_nSpk_null_all_meanOR = np.mean(tm_nSpk_null_all,axis=1)
    tm_nSpk_null_all_meanOR_meanOC = np.mean(tm_nSpk_null_all_meanOR,axis=0)

    fig, acx = plt.subplots(2,tm_bins-1 , figsize=(20,5))
    for b in range(tm_bins-1):
        x = np.random.rand(n_bins)
        acx[0,b].scatter(x,tm_nSpk_pref_all_meanOR_meanOC[:,b])

        acx[1,b].scatter(x,tm_nSpk_null_all_meanOR_meanOC[:,b])

    plt.savefig('Output/DSI_TC/Spike_to_Curr/'+kind+'/gInh_perTimeBin.png')

def analyze():

    if not os.path.exists('Output/DSI_TC/Spike_to_Curr'):
        os.mkdir('Output/DSI_TC/Spike_to_Curr')

    if not os.path.exists('Output/DSI_TC/Spike_to_Curr/high_DSI'):
        os.mkdir('Output/DSI_TC/Spike_to_Curr/high_DSI')

    if not os.path.exists('Output/DSI_TC/Spike_to_Curr/low_DSI'):
        os.mkdir('Output/DSI_TC/Spike_to_Curr/low_DSI')

    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')
    print(np.shape(preff_E1))

    
    dsi_kim = np.squeeze(np.load('./work/dsi_kim_Cells_TC.npy'))
    


    ## perform the evaluation only on cells with a DSI (>0.8) and a low DSI (<0.2) to save time
   
    idx_highDSI = np.where(dsi_kim > 0.8)[0]
    idx_lowDSI = np.where(dsi_kim<0.2)[0]
    

    analyzeDSI(preff_E1, idx_highDSI, 'high_DSI')
    analyzeDSI(preff_E1, idx_lowDSI, 'low_DSI')

    ###
    # Load Data with high DSI
    ###
    tm_nSpk_pref_all_high = np.load('./work/direct_tm_nSpk_pref_all_high_DSI.npy')
    tm_nSpk_null_all_high = np.load('./work/direct_tm_nSpk_null_all_high_DSI.npy')
    
    gExc_bin_pref_high = np.load('./work/direct_gEx_E_pref_bins_all_high_DSI.npy')
    gExc_bin_null_high = np.load('./work/direct_gEx_E_null_bins_all_high_DSI.npy')
    
    gInh_bin_pref_high = np.load('./work/direct_gIn_E_pref_bins_all_high_DSI.npy')
    gInh_bin_null_high = np.load('./work/direct_gIn_E_null_bins_all_high_DSI.npy')

    n_cellsTM = np.load('./work/direct_cellsTM_high_DSI.npy')


    ###
    # Load Data with low DSI
    ###
    tm_nSpk_pref_all_low = np.load('./work/direct_tm_nSpk_pref_all_low_DSI.npy')
    tm_nSpk_null_all_low = np.load('./work/direct_tm_nSpk_null_all_low_DSI.npy')
    
    gExc_bin_pref_low = np.load('./work/direct_gEx_E_pref_bins_all_low_DSI.npy')
    gExc_bin_null_low = np.load('./work/direct_gEx_E_null_bins_all_low_DSI.npy')
    
    gInh_bin_pref_low = np.load('./work/direct_gIn_E_pref_bins_all_low_DSI.npy')
    gInh_bin_null_low = np.load('./work/direct_gIn_E_null_bins_all_low_DSI.npy')


    
    print(np.shape(tm_nSpk_pref_all_high), np.shape(gExc_bin_pref_high), np.shape(gInh_bin_pref_high))

    n_cells_high, repeats, bins_tims, bins_tm = np.shape(tm_nSpk_pref_all_high)
    n_cells_low , _, _, _ = np.shape(tm_nSpk_pref_all_low)

    tm_nSpk_pref_all_high_oR = np.mean(tm_nSpk_pref_all_high,axis=1)
    tm_nSpk_null_all_high_oR = np.mean(tm_nSpk_null_all_high,axis=1)
    gExc_bin_pref_high_oR = np.mean(gExc_bin_pref_high, axis=1)
    gExc_bin_null_high_oR = np.mean(gExc_bin_null_high, axis=1)
    gInh_bin_pref_high_oR = np.mean(gInh_bin_pref_high, axis=1)
    gInh_bin_null_high_oR = np.mean(gInh_bin_null_high, axis=1)



    tm_nSpk_pref_all_low_oR = np.mean(tm_nSpk_pref_all_low,axis=1)
    tm_nSpk_null_all_low_oR = np.mean(tm_nSpk_null_all_low,axis=1)
    gExc_bin_pref_low_oR = np.mean(gExc_bin_pref_low, axis=1)
    gExc_bin_null_low_oR = np.mean(gExc_bin_null_low, axis=1)
    gInh_bin_pref_low_oR = np.mean(gInh_bin_pref_low, axis=1)
    gInh_bin_null_low_oR = np.mean(gInh_bin_null_low, axis=1)

    print(np.shape(tm_nSpk_pref_all_high_oR))
    tm_nSpk_pref_all_high_oR_sumTB = np.mean(tm_nSpk_pref_all_high_oR,axis=1)
    tm_nSpk_null_all_high_oR_sumTB = np.mean(tm_nSpk_null_all_high_oR,axis=1)

    tm_nSpk_pref_all_low_oR_sumTB = np.mean(tm_nSpk_pref_all_low_oR,axis=1)
    tm_nSpk_null_all_low_oR_sumTB = np.mean(tm_nSpk_null_all_low_oR,axis=1)


    t_win = 25
    
    crossCalc_pref_high = np.zeros((n_cells_high,bins_tm, 2*t_win+1)) 
    crossCalc_null_high = np.zeros((n_cells_high,bins_tm, 2*t_win+1))    

    crossCalc_pref_low = np.zeros((n_cells_low,bins_tm, 2*t_win+1)) 
    crossCalc_null_low = np.zeros((n_cells_low,bins_tm, 2*t_win+1))    

    ## first for all cells with a high DSI
    for c in range(n_cells_high):
        for b in range(bins_tm):  
            if np.sum(tm_nSpk_pref_all_high_oR[c,:,b]) > 0:
                crossCalc = calcCrossCorr(gExc_bin_pref_high_oR[c]/np.max(gExc_bin_pref_high_oR[c]) , tm_nSpk_pref_all_high_oR[c,:,b]/np.max(tm_nSpk_pref_all_high_oR[c,:,b]), t_win)
                crossCalc_pref_high[c,b] = crossCalc
            else:
                crossCalc_pref_high[c,b] = np.ones(2*t_win+1)*np.nan
            if np.sum(tm_nSpk_null_all_high_oR[c,:,b]) > 0:
                crossCalc = calcCrossCorr(gExc_bin_null_high_oR[c]/np.max(gExc_bin_null_high_oR[c]) , tm_nSpk_null_all_high_oR[c,:,b]/np.max(tm_nSpk_null_all_high_oR[c,:,b]), t_win)
                crossCalc_null_high[c,b] = crossCalc
            else:
                crossCalc_null_high[c,b] = np.ones(2*t_win+1)*np.nan


    ## then for all cells with a low DSI
    for c in range(n_cells_low):
        for b in range(bins_tm):  
            if np.sum(tm_nSpk_pref_all_low_oR[c,:,b]) > 0:
                crossCalc = calcCrossCorr(gExc_bin_pref_low_oR[c]/np.max(gExc_bin_pref_low_oR[c]) , tm_nSpk_pref_all_low_oR[c,:,b]/np.max(tm_nSpk_pref_all_low_oR[c,:,b]), t_win)
                crossCalc_pref_low[c,b] = crossCalc
            else:
                crossCalc_pref_low[c,b] = np.ones(2*t_win+1)*np.nan
            if np.sum(tm_nSpk_null_all_low_oR[c,:,b]) > 0:
                crossCalc = calcCrossCorr(gExc_bin_null_low_oR[c]/np.max(gExc_bin_null_low_oR[c]) , tm_nSpk_null_all_low_oR[c,:,b]/np.max(tm_nSpk_null_all_low_oR[c,:,b]), t_win)
                crossCalc_null_low[c,b] = crossCalc
            else:
                crossCalc_null_low[c,b] = np.ones(2*t_win+1)*np.nan

    plt.figure()
    plt.subplot(411)
    plt.plot(crossCalc_pref_high[0,3,:])
    plt.subplot(412)
    plt.plot(crossCalc_pref_high[1,3,:])
    plt.subplot(413)
    plt.plot(crossCalc_pref_high[2,3,:])
    plt.subplot(414)
    plt.plot(crossCalc_pref_high[3,3,:])
    plt.savefig('Output/test_plot')


    max_crossCalc_pref_high = np.zeros((n_cells_high,bins_tm))
    max_crossCalc_pref_low = np.zeros((n_cells_low,bins_tm))

    max_crossCalc_null_high= np.zeros((n_cells_high,bins_tm))
    max_crossCalc_null_low = np.zeros((n_cells_low,bins_tm))

    lags = signal.correlation_lags(t_win+1, t_win+1)

    for c in range(n_cells_high):
        for b in range(bins_tm):

            if np.isnan(np.sum(crossCalc_pref_high[c,b])): 
                max_crossCalc_pref_high[c,b] = np.nan
            else:
                max_crossCalc_pref_high[c,b] = getDeltaT(crossCalc_pref_high[c,b], t_win)#lags[np.argmax(crossCalc_pref_high[c,b])]

            if np.isnan(np.sum(crossCalc_null_high[c,b])): 
                max_crossCalc_null_high[c,b] = np.nan
            else:
                max_crossCalc_null_high[c,b] = getDeltaT(crossCalc_null_high[c,b], t_win)#lags[np.argmax(crossCalc_null_high[c,b])] #


    for c in range(n_cells_low):
        for b in range(bins_tm):
            if np.isnan(np.sum(crossCalc_pref_low[c,b])): 
                max_crossCalc_pref_low[c,b] = np.nan
            else:
                max_crossCalc_pref_low[c,b] = getDeltaT(crossCalc_pref_low[c,b], t_win)#lags[np.argmax(crossCalc_pref_low[c,b])] #

            if np.isnan(np.sum(crossCalc_null_low[c,b])): 
                max_crossCalc_null_low[c,b] = np.nan
            else:
                max_crossCalc_null_low[c,b] = getDeltaT(crossCalc_null_low[c,b], t_win)#lags[np.argmax(crossCalc_null_low[c,b])] #


    np.save('./work/DSI_max_crossCalc_pref_high',max_crossCalc_pref_high)
    np.save('./work/DSI_max_crossCalc_pref_low' ,max_crossCalc_pref_low)
    np.save('./work/DSI_max_crossCalc_null_high',max_crossCalc_null_high)
    np.save('./work/DSI_max_crossCalc_null_low' ,max_crossCalc_null_low)



    x = np.linspace(0,bins_tm-1,bins_tm)
    print(x)
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    for b in range(bins_tm):
        ax[0,0].violinplot(max_crossCalc_pref_high[~np.isnan(max_crossCalc_pref_high[:,b]),b],[b], showmedians=True)
        ax[0,0].set_title('pref high')
    for b in range(bins_tm):
        ax[1,0].violinplot(max_crossCalc_pref_low[~np.isnan(max_crossCalc_pref_low[:,b]),b],[b], showmedians=True)
        ax[1,0].set_title('pref low')
    for b in range(bins_tm):
        ax[0,1].violinplot(max_crossCalc_null_high[~np.isnan(max_crossCalc_null_high[:,b]),b],[b], showmedians=True)
        ax[0,1].set_title('null high')
    for b in range(bins_tm):
        ax[1,1].violinplot(max_crossCalc_null_low[~np.isnan(max_crossCalc_null_low[:,b]),b],[b], showmedians=True)
        ax[1,1].set_title('null low')
    plt.savefig('Output/DSI_TC/Spike_to_Curr/Cross_Corr_deltaT_TMwise_violon')
    plt.close()


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
    ax[1,1].set_xticks(np.linspace(1,tm_bins-1,tm_bins-1), tm_labels)
    ax[1,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    ax[1,1].text(4.0,t_win-4,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    #ax[1,1].set_title('null low')
    ax[1,1].set_xlabel(r'cos')
    ax[1,1].set_ylim(-t_win-1,t_win+1)
    plt.savefig('Output/DSI_TC/Spike_to_Curr/Cross_Corr_deltaT_TMwise_boxplot',dpi=300,bbox_inches='tight')
    plt.close()



    fig, ax = plt.subplots(2,2, figsize=(7,7), sharex=True, sharey=True)
    boxes = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_pref_all_high_oR_sumTB[:,b]
        ax[0,0].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[0,0].text(4.0,7.9,'DSI > 0.8 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
    ax[0,0].set_ylim(-0.5,8.75)
    ax[0,0].set_ylabel(r'$ \overline{g_{Inh}}$ [nA]')
    #ax[0,0].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_null_all_high_oR_sumTB[:,b]
        ax[1,0].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[1,0].text(4.0,7.9,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
    ax[1,0].set_ylim(-0.5,8.75)
    ax[1,0].set_ylabel(r'$ \overline{g_{Inh}}$ [nA]')
    ax[1,0].set_xlabel(r'cos')
    #ax[1,0].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_pref_all_low_oR_sumTB[:,b]
        ax[0,1].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[0,1].text(4.0,7.9,'DSI > 0.8 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    ax[0,1].set_ylim(-0.5,8.75)
    #ax[0,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    plot_data = []
    for b in range(bins_tm):
        plot_data = tm_nSpk_null_all_low_oR_sumTB[:,b]
        ax[1,1].boxplot(plot_data, positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.45,medianprops=dict(color='black', linewidth=2.0))
    ax[1,1].set_xticks(np.linspace(1,tm_bins-1,tm_bins-1), tm_labels)
    #ax[1,1].hlines(0,0,bins_tm+1,colors='gray', linestyles='dashed' )
    ax[1,1].text(4.0,7.9,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white'))
    ax[1,1].set_xlabel(r'cos')
    ax[1,1].set_ylim(-0.5,8.75)
    plt.savefig('Output/DSI_TC/Spike_to_Curr/Sum_Inh_overTM_box',dpi=300,bbox_inches='tight')
    plt.close()



    fig, ax = plt.subplots(bins_tm,2, figsize=(7,15), sharex=True)
    for b in range(bins_tm):
        ax[b,0].plot(np.nanmean(crossCalc_pref_high[:,b],axis=0))
        ax[b,0].plot(np.nanmean(crossCalc_null_high[:,b],axis=0))
        #ax[b,0].vlines(t_win, -1,1, linestyles='dashed', colors="gray")
        #ax[b,0].hlines(0, 0,2*t_win, linestyles='dashed', colors="gray")
        ax[b,0].set_xticks(np.linspace(0,2*t_win-1,5), np.linspace(-t_win,t_win,5))

        ax[b,1].plot(np.nanmean(crossCalc_pref_low[:,b],axis=0))
        ax[b,1].plot(np.nanmean(crossCalc_null_low[:,b],axis=0))
        #ax[b,1].vlines(t_win, -1,1, linestyles='dashed', colors="gray")
        #ax[b,1].hlines(0, 0,2*t_win, linestyles='dashed', colors="gray")
        ax[b,1].set_xticks(np.linspace(0,2*t_win-1,5), np.linspace(-t_win,t_win,5))

    plt.savefig('Output/DSI_TC/Spike_to_Curr/Cross_Corr_gEx_Inh_templateWise')

    mean_crossCalc_pref_high = np.zeros((n_cells_high,bins_tm,2*t_win+1))
    mean_crossCalc_null_high = np.zeros((n_cells_high,bins_tm,2*t_win+1))

    
    print(np.shape(crossCalc_pref_high), np.shape(mean_crossCalc_pref_high))


    for c in range(n_cells_high):
        for b in range(bins_tm):
            mean_crossCalc_pref_high[c,b] = crossCalc_pref_high[c,b]#*n_cellsTM[c,b]
            mean_crossCalc_null_high[c,b] = crossCalc_null_high[c,b]#*n_cellsTM[c,b]


    mean_crossCalc_pref_high = np.nanmean(mean_crossCalc_pref_high, axis=0)
    mean_crossCalc_null_high = np.nanmean(mean_crossCalc_null_high, axis=0)

    plt.figure()
    plt.plot(np.nanmean(mean_crossCalc_pref_high,axis=0),label='pref')
    plt.plot(np.nanmean(mean_crossCalc_null_high,axis=0),label='null')
    #plt.vlines(t_win, -0.5,0.5, linestyles='dashed', colors="gray")
    #plt.hlines(0, 0,2*t_win, linestyles='dashed', colors="gray")
    plt.legend()
    plt.xticks(np.linspace(0,2*t_win,5), np.linspace(-t_win,t_win,5) )
    plt.savefig('Output/DSI_TC/Spike_to_Curr/meanCrossCorr_overTM')    

    ## just to get sure
    cross_gE_gI_pref_high = np.zeros((n_cells_high,2*t_win+1 ))
    cross_gE_gI_null_high = np.zeros((n_cells_high,2*t_win+1 ))
    cross_gE_gI_pref_low = np.zeros((n_cells_low,2*t_win+1 ))
    cross_gE_gI_null_low = np.zeros((n_cells_low,2*t_win+1 ))
    cross_pref_test = np.zeros((n_cells_high,2*t_win+1 ))
    cross_null_test = np.zeros((n_cells_high,2*t_win+1 ))
    for c in range(n_cells_high):
        cross_gE_gI_pref_high[c] = calcCrossCorr(gExc_bin_pref_high_oR[c]/np.max(gExc_bin_pref_high_oR[c]) , gInh_bin_pref_high_oR[c]/np.max(gInh_bin_pref_high_oR[c]) , t_win) 
        cross_gE_gI_null_high[c] = calcCrossCorr(gExc_bin_null_high_oR[c]/np.max(gExc_bin_null_high_oR[c]) , gInh_bin_null_high_oR[c]/np.max(gInh_bin_null_high_oR[c]), t_win)

        cross_gE_gI_pref_low[c] = calcCrossCorr(gExc_bin_pref_low_oR[c]/np.max(gExc_bin_pref_low_oR[c]) , gInh_bin_pref_low_oR[c]/np.max(gInh_bin_pref_low_oR[c]), t_win) 
        cross_gE_gI_null_low[c] = calcCrossCorr(gExc_bin_null_low_oR[c]/np.max(gExc_bin_null_low_oR[c]) , gInh_bin_null_low_oR[c]/np.max(gInh_bin_null_low_oR[c]), t_win)

        #print(cross_gE_gI_pref_high[c])
        corr = np.correlate(gExc_bin_pref_high_oR[c]/np.max(gExc_bin_pref_high_oR[c]),gInh_bin_pref_high_oR[c]/np.max(gInh_bin_pref_high_oR[c]), 'full' )
        l = int(len(corr)/2)
        cross_pref_test[c] = corr[l-t_win:l+t_win+1]

        corr = np.correlate(gExc_bin_null_high_oR[c]/np.max(gExc_bin_null_high_oR[c]),gInh_bin_null_high_oR[c]/np.max(gInh_bin_null_high_oR[c]), 'full' )
        l = int(len(corr)/2)
        cross_null_test[c] = corr[l-t_win:l+t_win+1]

    plt.figure()
    plt.plot(np.mean(cross_pref_test,axis=0))
    plt.plot(np.mean(cross_null_test,axis=0))
    plt.savefig('Output/test_cross_plot')

    plt.figure()
    plt.subplot(121)
    plt.plot(np.mean(cross_gE_gI_pref_high,axis=0), label='pref')
    plt.plot(np.mean(cross_gE_gI_null_high,axis=0), label='null')
    plt.vlines(t_win, -1,1, linestyles='dashed', colors="gray")
    plt.hlines(0, 0,2*t_win, linestyles='dashed', colors="gray")
    plt.legend()
    plt.xticks(np.linspace(0,2*t_win,5), np.linspace(-t_win,t_win,5) )
    plt.subplot(122)
    plt.plot(np.mean(cross_gE_gI_pref_low,axis=0), label='pref')
    plt.plot(np.mean(cross_gE_gI_null_low,axis=0), label='null')
    plt.vlines(t_win, -0.5,.5, linestyles='dashed', colors="gray")
    plt.hlines(0, 0,2*t_win, linestyles='dashed', colors="gray")
    plt.legend()
    plt.xticks(np.linspace(0,2*t_win,5), np.linspace(-t_win,t_win,5) )
    plt.savefig('Output/DSI_TC/Spike_to_Curr/Cross_Corr_gEx_Inh_test')    



if __name__ == '__main__':
    analyze()

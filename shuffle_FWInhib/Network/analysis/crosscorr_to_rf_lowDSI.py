import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm 
from scipy import signal

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
        if np.nanmax(area_start) >= np.nanmax(area_new):
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

def analyze():

    if not os.path.exists('Output/DSI_TC/Inhib_RF'):
        os.mkdir('Output/DSI_TC/Inhib_RF')


    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']
    total_T = sim_param['totalT']

    step_degree = int(360/n_steps)

    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')
    dsi_kim = np.squeeze(np.load('./work/dsi_kim_Cells_TC.npy'))
    temp_E_I = np.load('./work/DSI_RF_tempE1_I1.npy')


    fb_w = np.loadtxt('./Input_network/INtoV1.txt')
    _,n_inh = np.shape(fb_w)
        
    gabor_e = np.load('./work/parameter.npy')

    ## perform the evaluation only on cells with a DSI (>0.8) and a low DSI (<0.2) to save time
   
    idx_lowDSI = np.where(dsi_kim<0.2)[0]

    gabor_e_high = np.copy(gabor_e[idx_lowDSI])
    ori_ex = gabor_e_high[:,1]/np.pi*180.0
    phase_ex = gabor_e_high[:,5]
    xc_ex = gabor_e_high[:,6]
    yc_ex = gabor_e_high[:,7]

    preff_D_highDSI = preff_E1[0,idx_lowDSI,2]
    temp_pref = temp_E_I[idx_lowDSI]

    ### just let look only on cells with a high DSI
    spk_E, gEx_E, gIn_E, spk_I = loadTime_Data(preff_E1, idx_lowDSI)

    print(np.shape(spk_E), np.shape(gEx_E), np.shape(spk_I))
    


    t_kickOff = 500
    repeats = 5

    t_bin_size = 2#5#10
    n_bins = int( (total_T-t_kickOff)/t_bin_size)


    ### calculate the crosscorrelation
    corr_w = 25
    crossCorr_pref = np.zeros((len(spk_E), n_inh, corr_w*2+1))
    crossCorr_pref_maxDT = np.zeros((len(spk_E), n_inh))
    crossCorr_null = np.zeros((len(spk_E), n_inh, corr_w*2+1))
    crossCorr_null_maxDT = np.zeros((len(spk_E), n_inh))

    for c in tqdm(range(len(spk_E)), ncols=80):

        aprox_gInh_pref = np.zeros((repeats, n_inh, n_bins))
        aprox_gInh_null = np.zeros((repeats, n_inh, n_bins))

        gEx_E_pref = np.zeros((repeats, n_bins))
        gEx_E_null = np.zeros((repeats, n_bins))

        preff_arg = int(preff_D_highDSI[c])
        preff_D = preff_arg*step_degree
        null_D = (preff_D+180)%360
        ## calculate the index of null direction
        null_arg = int(null_D/step_degree)


        gEx_E_pref_cell = gEx_E[c,preff_arg,:,t_kickOff:]
        gEx_E_null_cell = gEx_E[c,null_arg,:,t_kickOff:]
        ## get the corresponding inhibitory spike times
        spk_I_pref = spk_I[c,preff_arg,:,:,t_kickOff:]
        spk_I_null = spk_I[c,null_arg,:,:,t_kickOff:]

        ### get the spike count of each inibitory neuron in a time bin and multiply with weight to get the aproximated inhibitory current
        for tb in range(n_bins): # go through all time bins
            for r in range(repeats):
                spT_pref = spk_I_pref[r,:,0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)]
                aprox_gInh_pref[r,:,tb] = np.sum(spT_pref,axis=1)*fb_w[c,:]

                spT_null = spk_I_null[r,:,0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)]
                aprox_gInh_null[r,:,tb] = np.sum(spT_null,axis=1)*fb_w[c,:]
        
                ## and do it also for the excitatory cells, use the excitatory current
                gE_pref = gEx_E_pref_cell[r,0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)]
                gEx_E_pref[r,tb] = np.sum(gE_pref)

                gE_null = gEx_E_null_cell[r,0+(t_bin_size*tb):t_bin_size+(t_bin_size*tb)]
                gEx_E_null[r,tb] = np.sum(gE_null)

        ## calculate mean over repitions
        mean_aprox_gInh_pref = np.nanmean(aprox_gInh_pref,axis=0)
        mean_aprox_gInh_null = np.nanmean(aprox_gInh_null,axis=0)

        mean_gEx_E_pref = np.nanmean(gEx_E_pref,axis=0)
        mean_gEx_E_null = np.nanmean(gEx_E_null,axis=0)

        for i in range(n_inh):

            crossCorr_p = calcCrossCorr(mean_gEx_E_pref/np.nanmax(mean_gEx_E_pref), mean_aprox_gInh_pref[i]/np.nanmax(mean_aprox_gInh_pref[i]), corr_w)
            crossCorr_pref[c,i] = crossCorr_p
            if np.isnan(np.sum(crossCorr_p)):
                crossCorr_pref_maxDT[c,i] = np.nan
            else:
                crossCorr_pref_maxDT[c,i] = getDeltaT(crossCorr_p,corr_w)

            crossCorr_n = calcCrossCorr(mean_gEx_E_null/np.nanmax(mean_gEx_E_null), mean_aprox_gInh_null[i]/np.nanmax(mean_aprox_gInh_null[i]), corr_w)
            crossCorr_null[c,i] = crossCorr_n#np.argmax(crossCorr_null)-corr_w
            if np.isnan(np.sum(crossCorr_n)):
                crossCorr_null_maxDT[c,i] = np.nan
            else:
                crossCorr_null_maxDT[c,i] = getDeltaT(crossCorr_n,corr_w)
            

    np.save('./work/crossCorr_pref_currents_singleInh_lowDSI', crossCorr_pref)
    np.save('./work/maxDT_pref_currents_singleInh_lowDSI', crossCorr_pref_maxDT)
    np.save('./work/maxDT_null_currents_singleInh_lowDSI', crossCorr_null_maxDT)
    np.save('./work/crossCorr_null_currents_singleInh_lowDSI', crossCorr_null)

def plotAllCells():

    if not os.path.exists('Output/DSI_TC/Inhib_RF'):
        os.mkdir('Output/DSI_TC/Inhib_RF')

    if not os.path.exists('Output/DSI_TC/Inhib_RF/cells'):
        os.mkdir('Output/DSI_TC/Inhib_RF/cells')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']
    total_T = sim_param['totalT']

    step_degree = int(360/n_steps)

    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')
    dsi_kim = np.squeeze(np.load('./work/dsi_kim_Cells_TC.npy'))
    temp_E_I = np.load('./work/DSI_RF_tempE1_I1.npy')


    fb_w = np.loadtxt('./Input_network/INtoV1.txt')
    ff_I = np.loadtxt('./Input_network/InhibW.txt')

    n_exc, n_inh = np.shape(fb_w)
    n_i , n_l = np.shape(ff_I)
    ff_I = np.reshape(ff_I, (n_i, 18,18,2))
    rf_I = ff_I[:,:,:,0] - ff_I[:,:,:,1]

    gabor_i = np.load('./work/gabors_parameter_inhib.npy')
    print(np.shape(gabor_i))

    ori_inhib = gabor_i[:,1]/np.pi*180.0
    phase_inhib = gabor_i[:,5]
    xc_inhib = gabor_i[:,6]
    yc_inhib = gabor_i[:,7]
        

    gabor_e = np.load('./work/parameter.npy')

    idx_lowDSI = np.where(dsi_kim < 0.2)[0]

    gabor_e_high = np.copy(gabor_e[idx_lowDSI])
    ori_ex = gabor_e_high[:,1]/np.pi*180.0
    phase_ex = gabor_e_high[:,5]
    xc_ex = gabor_e_high[:,6]
    yc_ex = gabor_e_high[:,7]

    temp_pref = temp_E_I[idx_lowDSI]

    crossCorr_pref = np.load('./work/crossCorr_pref_currents_singleInh_lowDSI.npy')
    crossCorr_pref_maxDT = np.load('./work/maxDT_pref_currents_singleInh_lowDSI.npy')

    crossCorr_null = np.load('./work/crossCorr_null_currents_singleInh_lowDSI.npy')
    crossCorr_null_maxDT = np.load('./work/maxDT_null_currents_singleInh_lowDSI.npy')



    ## calculate the difference between the center of the excitatory cell and all inhibiotry cells
    s_dist = np.zeros((len(gabor_e_high),n_inh))
    o_dist = np.zeros((len(gabor_e_high),n_inh))
    p_dist = np.zeros((len(gabor_e_high),n_inh))
    for c in tqdm(range(len(gabor_e_high)), ncols=80):
        for i in range(n_inh):
            s_dist[c,i] = np.sqrt((xc_ex[c] - xc_inhib[i])**2 + (yc_ex[c] - yc_inhib[i])**2 )
            o_dist[c,i] = np.abs(ori_ex[c]%90 - ori_inhib[i]%90)#%90
            p_dist[c,i] = np.abs(phase_ex[c] - phase_inhib[i])#%(np.pi/2)



    ## do some plots for one single cell
    for c in tqdm(range(len(idx_lowDSI)), ncols=80):




        ### plot a 2D map with the center position of the inhibitory cells and the corresponding deltaT
        im_pref = np.zeros((18,18))
        cm_pref = np.ones((18,18))

        im_null = np.zeros((18,18))
        cm_null = np.ones((18,18))
        for i in range(n_inh):
            im_pref[np.clip(int(xc_inhib[i]),0,17),np.clip(int(yc_inhib[i]),0,17)] += crossCorr_pref_maxDT[c,i]
            cm_pref[np.clip(int(xc_inhib[i]),0,17),np.clip(int(yc_inhib[i]),0,17)] +=1

            im_null[np.clip(int(xc_inhib[i]),0,17),np.clip(int(yc_inhib[i]),0,17)] += crossCorr_null_maxDT[c,i]
            cm_null[np.clip(int(xc_inhib[i]),0,17),np.clip(int(yc_inhib[i]),0,17)] +=1

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.imshow(im_pref/cm_pref,cmap='gray')
        plt.colorbar()
        plt.scatter(xc_ex[c],yc_ex[c], c = 'red')
        plt.subplot(122)
        plt.imshow(im_null/cm_null,cmap='gray')
        plt.colorbar()
        plt.scatter(xc_ex[c],yc_ex[c], c = 'red')
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_estDeltaT_Map_lowDSI.png'%(c))
        plt.close()

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot(crossCorr_pref_maxDT[c],s_dist[c], 'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('euclidian distance')
        plt.subplot(122)
        plt.plot(crossCorr_null_maxDT[c],s_dist[c], 'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('euclidian distance')
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_DeltaTtoDistance_lowDSI.png'%(c))
        plt.close()

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot(crossCorr_pref_maxDT[c],o_dist[c], 'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('orientation distance')
        plt.subplot(122)
        plt.plot(crossCorr_null_maxDT[c],o_dist[c], 'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('orientation distance')
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_DeltaTtoOrientation_lowDSI.png'%(c))
        plt.close()

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot(crossCorr_pref_maxDT[c],p_dist[c], 'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('phase difference')
        plt.subplot(122)
        plt.plot(crossCorr_null_maxDT[c],p_dist[c], 'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('phase difference')
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_DeltaTtoPhase_lowDSI.png'%(c))
        plt.close()



        plt.figure()
        plt.subplot(411)
        plt.plot(crossCorr_pref[c,0,:])
        plt.subplot(412)
        plt.plot(crossCorr_pref[c,1,:])
        plt.subplot(413)
        plt.plot(crossCorr_pref[c,2,:])
        plt.subplot(414)
        plt.plot(crossCorr_pref[c,3,:])
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_testCrossCorr_pref_lowDSI.png'%(c))
        plt.close()

        plt.figure()
        plt.subplot(411)
        plt.plot(crossCorr_null[c,0,:])
        plt.subplot(412)
        plt.plot(crossCorr_null[c,1,:])
        plt.subplot(413)
        plt.plot(crossCorr_null[c,2,:])
        plt.subplot(414)
        plt.plot(crossCorr_null[c,3,:])
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_testCrossCorr_null_lowDSI.png'%(c))
        plt.close()
        
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot(crossCorr_pref_maxDT[c],temp_pref[c],'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('cos')
        plt.subplot(122)
        plt.plot(crossCorr_null_maxDT[c],temp_pref[c],'o')
        plt.xlabel(r'$\Delta T$')
        plt.ylabel('cos')
        plt.savefig('Output/DSI_TC/Inhib_RF/cells/cell_%i_TMtoDeltaT_lowDSI.png'%(c))
        plt.close()


def getPartialCorrelation(dataX, dataY, dataZ):

    ## calculate the Pearson-Correlation between all pairs
    r_X_Z = np.corrcoef(dataX,dataZ)[0,1] 
    r_Y_Z = np.corrcoef(dataY,dataZ)[0,1] 
    r_X_Y = np.corrcoef(dataX,dataY)[0,1] 

    ## calculate the partial correlation
    partial_corr_X = (r_X_Z - (r_Y_Z*r_X_Y))/np.sqrt((1 - (r_Y_Z**2) )) * np.sqrt(( 1 - (r_X_Y**2) ) )
    partial_corr_Y = (r_Y_Z - (r_X_Z*r_X_Y))/np.sqrt((1 - (r_X_Z**2) )) * np.sqrt(( 1 - (r_X_Y**2) ) )

    return(partial_corr_X, partial_corr_Y)  
    
def plotBinned():

    ### now do all the things again but for all cells and bin it! ###

    if not os.path.exists('Output/DSI_TC/Inhib_RF'):
        os.mkdir('Output/DSI_TC/Inhib_RF')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']
    total_T = sim_param['totalT']

    step_degree = int(360/n_steps)

    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')
    dsi_kim = np.squeeze(np.load('./work/dsi_kim_Cells_TC.npy'))
    temp_E_I = np.load('./work/DSI_RF_tempE1_I1.npy')


    fb_w = np.loadtxt('./Input_network/INtoV1.txt')
    ff_I = np.loadtxt('./Input_network/InhibW.txt')

    n_exc, n_inh = np.shape(fb_w)
    n_i , n_l = np.shape(ff_I)
    ff_I = np.reshape(ff_I, (n_i, 18,18,2))
    rf_I = ff_I[:,:,:,0] - ff_I[:,:,:,1]

    gabor_i = np.load('./work/gabors_parameter_inhib.npy')
    print(np.shape(gabor_i))

    ori_inhib = gabor_i[:,1]/np.pi*180.0
    phase_inhib = gabor_i[:,5]
    xc_inhib = gabor_i[:,6]
    yc_inhib = gabor_i[:,7]
        

    gabor_e = np.load('./work/parameter.npy')

    idx_lowDSI = np.where(dsi_kim < 0.2)[0]

    gabor_e_high = np.copy(gabor_e[idx_lowDSI])
    ori_ex = gabor_e_high[:,1]/np.pi*180.0
    phase_ex = gabor_e_high[:,5]
    xc_ex = gabor_e_high[:,6]
    yc_ex = gabor_e_high[:,7]

    temp_pref = temp_E_I[idx_lowDSI]

    crossCorr_pref = np.load('./work/crossCorr_pref_currents_singleInh_lowDSI.npy')
    crossCorr_pref_maxDT = np.load('./work/maxDT_pref_currents_singleInh_lowDSI.npy')

    crossCorr_null = np.load('./work/crossCorr_null_currents_singleInh_lowDSI.npy')
    crossCorr_null_maxDT = np.load('./work/maxDT_null_currents_singleInh_lowDSI.npy')



    ## calculate the difference between the center of the excitatory cell and all inhibiotry cells
    s_dist = np.zeros((len(gabor_e_high),n_inh))
    o_dist = np.zeros((len(gabor_e_high),n_inh))
    p_dist = np.zeros((len(gabor_e_high),n_inh))
    for c in tqdm(range(len(gabor_e_high)), ncols=80):
        for i in range(n_inh):
            s_dist[c,i] = np.sqrt((xc_ex[c] - xc_inhib[i])**2 + (yc_ex[c] - yc_inhib[i])**2 )
            o_dist[c,i] = np.abs(ori_ex[c]%90 - ori_inhib[i]%90)#%90
            p_dist[c,i] = np.abs(phase_ex[c] - phase_inhib[i])#%(np.pi/2)

    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(crossCorr_pref_maxDT.flatten(),s_dist.flatten(), 'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('euclidian distance')
    plt.subplot(122)
    plt.plot(crossCorr_null_maxDT.flatten(),s_dist.flatten(), 'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('euclidian distance')
    plt.savefig('Output/DSI_TC/Inhib_RF/DeltaTtoDistance_allC_lowDSI.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(crossCorr_pref_maxDT.flatten(),o_dist.flatten(), 'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('orientation distance')
    plt.subplot(122)
    plt.plot(crossCorr_null_maxDT.flatten(),o_dist.flatten(), 'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('orientation distance')
    plt.savefig('Output/DSI_TC/Inhib_RF/DeltaTtoOrientation_allC_lowDSI.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(crossCorr_pref_maxDT.flatten(),p_dist.flatten(), 'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('phase difference')
    plt.subplot(122)
    plt.plot(crossCorr_null_maxDT.flatten(),p_dist.flatten(), 'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('phase difference')
    plt.savefig('Output/DSI_TC/Inhib_RF/DeltaTtoPhase_allC_lowDSI.png')
    plt.close()


    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(crossCorr_pref_maxDT.flatten(),temp_pref.flatten(),'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('cos')
    plt.subplot(122)
    plt.plot(crossCorr_null_maxDT.flatten(),temp_pref.flatten(),'o')
    plt.xlabel(r'$\Delta T$')
    plt.ylabel('cos')
    plt.savefig('Output/DSI_TC/Inhib_RF/TMtoDeltaT_allC_lowDSI.png')
    plt.close()    
    

    #### calculate the partial calculation ####

    ## between spatial difference and orientation difference
    partial_corr_sp_pref_1 = np.zeros(len(idx_lowDSI))
    partial_corr_or_pref_1 = np.zeros(len(idx_lowDSI))
    partial_corr_sp_null_1 = np.zeros(len(idx_lowDSI))
    partial_corr_or_null_1 = np.zeros(len(idx_lowDSI))

    ## between spatial difference and phase difference
    partial_corr_sp_pref_2 = np.zeros(len(idx_lowDSI))
    partial_corr_ph_pref_2 = np.zeros(len(idx_lowDSI))
    partial_corr_sp_null_2 = np.zeros(len(idx_lowDSI))
    partial_corr_ph_null_2 = np.zeros(len(idx_lowDSI))

    ## between phase difference and orientation difference
    partial_corr_or_pref_3 = np.zeros(len(idx_lowDSI))
    partial_corr_ph_pref_3 = np.zeros(len(idx_lowDSI))
    partial_corr_or_null_3 = np.zeros(len(idx_lowDSI))
    partial_corr_ph_null_3 = np.zeros(len(idx_lowDSI))


    for c in range(len(idx_lowDSI)):

        partial_corr_sp_pref_1[c], partial_corr_or_pref_1[c] = getPartialCorrelation(s_dist[c], o_dist[c], crossCorr_pref_maxDT[c])
        partial_corr_sp_null_1[c], partial_corr_or_null_1[c] = getPartialCorrelation(s_dist[c], o_dist[c], crossCorr_null_maxDT[c])


        partial_corr_sp_pref_2[c], partial_corr_ph_pref_2[c] = getPartialCorrelation(s_dist[c], p_dist[c], crossCorr_pref_maxDT[c])
        partial_corr_sp_null_2[c], partial_corr_ph_null_2[c] = getPartialCorrelation(s_dist[c], p_dist[c], crossCorr_null_maxDT[c])


        partial_corr_or_pref_3[c], partial_corr_ph_pref_3[c] = getPartialCorrelation(o_dist[c], p_dist[c], crossCorr_pref_maxDT[c])
        partial_corr_or_null_3[c], partial_corr_ph_null_3[c] = getPartialCorrelation(o_dist[c], p_dist[c], crossCorr_null_maxDT[c])




    fig, axs = plt.subplots(1,3, figsize = (10,3), sharey=True)
    axs[0].scatter(partial_corr_sp_pref_1, partial_corr_or_pref_1, c = 'steelblue', label='pref')
    axs[0].scatter(partial_corr_sp_null_1, partial_corr_or_null_1, c = 'tomato', label='null')
    axs[0].set_ylim(-1,1)
    axs[0].set_xlim(-1,1)
    axs[0].set_xlabel('partial correlation spatial distance')
    axs[0].set_ylabel('partial correlation orientation distance')
    axs[0].legend()

    axs[1].scatter(partial_corr_sp_pref_2, partial_corr_ph_pref_2, c = 'steelblue', label='pref')
    axs[1].scatter(partial_corr_sp_null_2, partial_corr_ph_null_2, c = 'tomato', label='null')
    axs[1].set_ylim(-1,1)
    axs[1].set_xlim(-1,1)
    axs[1].set_xlabel('partial correlation spatial distance')
    axs[1].set_ylabel('partial correlation phase distance')
    axs[1].legend()

    axs[2].scatter(partial_corr_or_pref_3, partial_corr_ph_pref_3, c = 'steelblue', label='pref')
    axs[2].scatter(partial_corr_or_null_3, partial_corr_ph_null_3, c = 'tomato', label='null')
    axs[2].set_ylim(-1,1)
    axs[2].set_xlim(-1,1)
    axs[2].set_xlabel('partial correlation orientation distance')
    axs[2].set_ylabel('partial correlation phase distance')
    axs[2].legend()

    plt.savefig('Output/DSI_TC/Inhib_RF/Partial_Correlation_space_orientation_lowDSI.png',dpi=300,bbox_inches='tight')
    plt.close()
    ####

    return -1


    ## Binning
    t_win = 25
    n_bins = 5
    
    ## bin over TM ##
    tm_bin_bounds = np.linspace(-.8, .8, n_bins+1)
    print(np.shape(crossCorr_pref_maxDT), np.shape(temp_pref))

    maxDT_TM_bins_pref = np.zeros((len(idx_lowDSI), n_bins))
    maxDT_TM_bins_null = np.zeros((len(idx_lowDSI), n_bins))
    print(tm_bin_bounds)
    for c in range(len(idx_lowDSI)): # go over all exc cells
        for b in range(len(tm_bin_bounds)-1): # go through all bounds (except the last one)
            idx = np.where((temp_pref[c] >=tm_bin_bounds[b])&(temp_pref[c] <tm_bin_bounds[b+1]))[0] # get index of cells inside the bounds
            if len(idx) > 0:
                maxDT_TM_bins_pref[c,b] = np.nanmean(crossCorr_pref_maxDT[c,idx]) # average maxDT in this bin
                maxDT_TM_bins_null[c,b] = np.nanmean(crossCorr_null_maxDT[c,idx])

    plt.figure()
    plt.plot(np.nanmean(maxDT_TM_bins_pref,axis=0),'o--', label='pref')
    plt.plot(np.nanmean(maxDT_TM_bins_null,axis=0),'o--', label='null')
    plt.legend()
    plt.savefig('Output/DSI_TC/Inhib_RF/TMtoDeltaT_Binned_Test_lowDSI.png')


    tm_labels=['%.2f - %.2f'%(tm_bin_bounds[b],tm_bin_bounds[b+1]) for b in range(len(tm_bin_bounds)-1) ]
    fig,axs = plt.subplots(1,2, figsize=(15,5), sharey=True)
    for b in range(n_bins):
        axs[0].boxplot(maxDT_TM_bins_pref[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[0].text(4.0,t_win-4,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
        axs[1].boxplot(maxDT_TM_bins_null[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[1].text(4.0,t_win-4,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white')) 
    axs[0].set_ylabel(r'$\Delta T$')
    axs[0].set_ylim(-t_win,t_win)
    axs[0].set_xlabel(r'cos')
    axs[0].set_xticks(np.linspace(1,n_bins,n_bins),tm_labels)
    axs[0].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    axs[1].set_xlabel(r'cos')
    axs[1].set_xticks(np.linspace(1,n_bins,n_bins),tm_labels)
    axs[1].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    plt.savefig('Output/DSI_TC/Inhib_RF/TMtoDeltaT_Binned_Box_lowDSI.png',dpi=300,bbox_inches='tight')


    ## bin over euclidean distance ##
    ed_bin_bounds = np.linspace(0, 21, n_bins+1)
    print(np.shape(crossCorr_pref_maxDT), np.shape(s_dist))

    maxDT_ED_bins_pref = np.zeros((len(idx_lowDSI), n_bins))
    maxDT_ED_bins_null = np.zeros((len(idx_lowDSI), n_bins))
    print(ed_bin_bounds)
    for c in range(len(idx_lowDSI)): # go over all exc cells
        for b in range(len(ed_bin_bounds)-1): # go through all bounds (except the last one)
            idx = np.where((s_dist[c] >=ed_bin_bounds[b])&(s_dist[c] <ed_bin_bounds[b+1]))[0] # get index of cells inside the bounds
            if len(idx) > 0:
                maxDT_ED_bins_pref[c,b] = np.nanmean(crossCorr_pref_maxDT[c,idx]) # average maxDT in this bin
                maxDT_ED_bins_null[c,b] = np.nanmean(crossCorr_null_maxDT[c,idx])

    plt.figure()
    plt.plot(np.nanmean(maxDT_ED_bins_pref,axis=0),'o--', label='pref')
    plt.plot(np.nanmean(maxDT_ED_bins_null,axis=0),'o--', label='null')
    plt.legend()
    plt.savefig('Output/DSI_TC/Inhib_RF/EDtoDeltaT_Binned_Test_lowDSI.png')

    ed_labels=['%.2f - %.2f'%(ed_bin_bounds[b],ed_bin_bounds[b+1]) for b in range(len(ed_bin_bounds)-1) ]
    fig,axs = plt.subplots(1,2, figsize=(15,5), sharey=True)
    for b in range(n_bins):
        axs[0].boxplot(maxDT_ED_bins_pref[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[0].text(4.0,t_win-4,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
        axs[1].boxplot(maxDT_ED_bins_null[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[1].text(4.0,t_win-4,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white')) 
    axs[0].set_ylabel(r'$\Delta T$')
    axs[0].set_ylim(-t_win,t_win)
    axs[0].set_xlabel(r'eucl dist')
    axs[0].set_xticks(np.linspace(1,n_bins,n_bins),ed_labels)
    axs[0].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    axs[1].set_xlabel(r'eucl dist')
    axs[1].set_xticks(np.linspace(1,n_bins,n_bins),ed_labels)
    axs[1].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    plt.savefig('Output/DSI_TC/Inhib_RF/EDtoDeltaT_Binned_Box_lowDSI.png',dpi=300,bbox_inches='tight')


    ## bin over orientation distance ##
    or_bin_bounds = np.linspace(0, 90, n_bins+1)
    print(np.shape(crossCorr_pref_maxDT), np.shape(o_dist))
    
    maxDT_OR_bins_pref = np.zeros((len(idx_lowDSI), n_bins))
    maxDT_OR_bins_null = np.zeros((len(idx_lowDSI), n_bins))
    print(or_bin_bounds)
    for c in range(len(idx_lowDSI)): # go over all exc cells
        for b in range(len(or_bin_bounds)-1): # go through all bounds (except the last one)
            idx = np.where((o_dist[c] >=or_bin_bounds[b])&(o_dist[c] <or_bin_bounds[b+1]))[0] # get index of cells inside the bounds
            if len(idx) > 0:
                maxDT_OR_bins_pref[c,b] = np.nanmean(crossCorr_pref_maxDT[c,idx]) # average maxDT in this bin
                maxDT_OR_bins_null[c,b] = np.nanmean(crossCorr_null_maxDT[c,idx])

    plt.figure()
    plt.plot(np.nanmean(maxDT_OR_bins_pref,axis=0),'o--', label='pref')
    plt.plot(np.nanmean(maxDT_OR_bins_null,axis=0),'o--', label='null')
    plt.legend()
    plt.savefig('Output/DSI_TC/Inhib_RF/ORtoDeltaT_Binned_Test_lowDSI.png')

    or_labels=['%.2f - %.2f'%(or_bin_bounds[b],or_bin_bounds[b+1]) for b in range(len(or_bin_bounds)-1) ]
    fig,axs = plt.subplots(1,2, figsize=(15,5), sharey=True)
    for b in range(n_bins):
        axs[0].boxplot(maxDT_OR_bins_pref[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[0].text(4.0,t_win-4,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
        axs[1].boxplot(maxDT_OR_bins_null[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[1].text(4.0,t_win-4,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white')) 
    axs[0].set_ylabel(r'$\Delta T$')
    axs[0].set_ylim(-t_win,t_win)
    axs[0].set_xlabel(r'ori diff [°]')
    axs[0].set_xticks(np.linspace(1,n_bins,n_bins),or_labels)
    axs[0].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    axs[1].set_xlabel(r'ori diff [°]')
    axs[1].set_xticks(np.linspace(1,n_bins,n_bins),or_labels)
    axs[1].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    plt.savefig('Output/DSI_TC/Inhib_RF/ORtoDeltaT_Binned_Box_lowDSI.png',dpi=300,bbox_inches='tight')



    ## bin over phase difference ##
    ph_bin_bounds = np.linspace(0, np.pi, n_bins+1)
    print(np.shape(crossCorr_pref_maxDT), np.shape(p_dist))
    maxDT_PH_bins_pref = np.zeros((len(idx_lowDSI), n_bins))
    maxDT_PH_bins_null = np.zeros((len(idx_lowDSI), n_bins))
   
    for c in range(len(idx_lowDSI)): # go over all exc cells
        for b in range(len(ph_bin_bounds)-1): # go through all bounds (except the last one)
            idx = np.where((p_dist[c] >=ph_bin_bounds[b])&(p_dist[c] <ph_bin_bounds[b+1]))[0] # get index of cells inside the bounds
            if len(idx) > 0:
                maxDT_PH_bins_pref[c,b] = np.nanmean(crossCorr_pref_maxDT[c,idx]) # average maxDT in this bin
                maxDT_PH_bins_null[c,b] = np.nanmean(crossCorr_null_maxDT[c,idx])

    plt.figure()
    plt.plot(np.nanmean(maxDT_PH_bins_pref,axis=0),'o--', label='pref')
    plt.plot(np.nanmean(maxDT_PH_bins_null,axis=0),'o--', label='null')
    plt.legend()
    plt.savefig('Output/DSI_TC/Inhib_RF/PHtoDeltaT_Binned_Test_lowDSI.png')

    ph_labels=['%.2f - %.2f'%(ph_bin_bounds[b],ph_bin_bounds[b+1]) for b in range(len(ph_bin_bounds)-1) ]
    fig,axs = plt.subplots(1,2, figsize=(15,5), sharey=True)
    for b in range(n_bins):
        axs[0].boxplot(maxDT_PH_bins_pref[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='steelblue'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[0].text(4.0,t_win-4,'DSI < 0.2 \n preferred', bbox=dict(boxstyle='round', facecolor='white'))
        axs[1].boxplot(maxDT_PH_bins_null[:,b], positions= [b+1], patch_artist=True, boxprops=dict(facecolor='tomato'), widths=0.25,medianprops=dict(color='black', linewidth=2.0)) 
        axs[1].text(4.0,t_win-4,'DSI < 0.2 \n null', bbox=dict(boxstyle='round', facecolor='white')) 
    axs[0].set_ylabel(r'$\Delta T$')
    axs[0].set_ylim(-t_win,t_win)
    axs[0].set_xlabel(r'phase diff [rad]')
    axs[0].set_xticks(np.linspace(1,n_bins,n_bins),ph_labels)
    axs[0].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    axs[1].set_xlabel(r'phase diff [rad]')
    axs[1].set_xticks(np.linspace(1,n_bins,n_bins),ph_labels)
    axs[1].hlines(0,1,n_bins,colors='gray', linestyles='dashed')
    plt.savefig('Output/DSI_TC/Inhib_RF/PHtoDeltaT_Binned_Box_lowDSI.png',dpi=300,bbox_inches='tight')

if __name__ == '__main__':
    analyze()
    plotAllCells()
    plotBinned()


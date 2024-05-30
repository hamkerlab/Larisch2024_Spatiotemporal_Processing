import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def main():
    print('Analyse direction selectivity by preseting a moving Sinus')

    if not os.path.exists('Output/DSI/'):
        os.mkdir('Output/DSI/')

    spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')
    spkC_I1_all = np.load('./work/directGrating_Sinus_SpikeCount_I1.npy')
    spkC_LGN_all = np.load('./work/directGrating_Sinus_SpikeCount_LGN.npy')

    print('Mean FR LGN = ',np.mean(spkC_LGN_all))
    n_contrast, n_speeds, n_spatF,n_degress,repeats,n_cellsE1 = np.shape(spkC_E1_all)


    #sim_param = np.loadtxt('./work/directGrating_Sinus_parameter.txt')
    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']

    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]


    w_I1 = np.loadtxt('./Input_network/InhibW.txt')
    n_post,n_pre = np.shape(w_I1)
    rf_I1 = np.reshape(w_I1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_I1 = rf_I1[:,:,:,0] - rf_I1[:,:,:,1]


    dsi_maz = np.zeros((n_contrast,n_cellsE1)) #DSI after Mazurek et al. (2014) // (R_pref - R_null)/(R_pref)
    dsi_will = np.zeros((n_contrast,n_cellsE1))#DSI after Willson et al. (2018)// (R_pref - R_null)/(R_pref + R_null)
    dsi_kim = np.zeros((n_contrast,n_cellsE1))#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)

    preff_E1 = np.zeros((n_contrast,n_cellsE1,3)) # save the indices(!) of preffered speed[0], spatF[1], direction[2]

    for i in range(n_contrast):

        if not os.path.exists('Output/DSI/LVL_%i'%(i)):
            os.mkdir('Output/DSI/LVL_%i'%(i))

        step_degree = int(360/n_degress) 

        ## get through each cell and look, where the cells has spiked most
        for c in range(n_cellsE1):
            spkC_E1 = spkC_E1_all[i,:,:,:,:,c]
            spkC_E1 = np.mean(spkC_E1,axis=3)
            pref_speed, pref_spatF, pref_degree = np.where(spkC_E1 == np.max(spkC_E1))
            preff_E1[i,c,:] = pref_speed[0], pref_spatF[0], pref_degree[0]

            ##calculate the DSI
            spk_cell = spkC_E1[int(preff_E1[i,c,0]),int(preff_E1[i,c,1])]
            #spk_cell = spk_cell[0]
            preff_arg = int(preff_E1[i,c,2])
            preff_D = preff_arg*step_degree
            null_D = (preff_D+180)%360
            null_arg = int(null_D/step_degree)
            dsi_maz[i, c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg])
            dsi_will[i,c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg] + spk_cell[null_arg] )
            dsi_kim[i,c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])            


        plt.figure()
        plt.hist(dsi_maz[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Mazurek')
        plt.savefig('Output/DSI/LVL_%i/Hist_dsi_maz.png'%(i))

        plt.figure()
        plt.hist(dsi_will[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Willson')
        plt.savefig('Output/DSI/LVL_%i/Hist_dsi_will.png'%(i))

        plt.figure()
        plt.hist(dsi_kim[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Kim')
        plt.savefig('Output/DSI/LVL_%i/Hist_dsi_kim.png'%(i))
                    

    np.save('./work/dsi_kim_Cells',dsi_kim,allow_pickle=False)
    np.save('./work/dsi_preff_E1',preff_E1,allow_pickle=False)
#------------------------------------------------------------------------------
def moreDSI():

    if not os.path.exists('Output/direct_gratingSinus/maxDSI'):
        os.mkdir('Output/direct_gratingSinus/maxDSI')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    


    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]


    ##load the LGN-Delays
    delayON = np.load('./work/LGN_ON_delay.npy')
    delayOFF = np.load('./work/LGN_OFF_delay.npy')

    
    delayON_mean = np.mean(delayON,axis=1)
    delayOFF_mean = np.mean(delayOFF,axis=1)

    delays = np.append(delayON_mean,delayOFF_mean)
    delay_M = np.reshape(delays, ( int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2))
    delay_M = delay_M[:,:,0] - delay_M[:,:,1]
    print(np.shape(delay_M), np.shape(rf_E1))

    preff_E1 = np.load('./work/dsi_preff_E1.npy')

    dsi_kim = np.load('./work/dsi_kim_Cells.npy')
    print(np.shape(dsi_kim))
    n_cells = n_post

    mean_Delay_highW = np.zeros(n_cells)
    mean_Delay_lowW = np.zeros(n_cells)
    for i in range(n_cells):
        higW_arg = np.where(w_E1[i] >= np.max(w_E1[i])*0.75)[0]
        mean_Delay_highW[i] = np.mean(delays[higW_arg])
        
        lowW_arg = np.where(w_E1[i] <= np.max(w_E1[i])*0.25)[0]
        mean_Delay_lowW[i] = np.mean(delays[lowW_arg])
        #print(len(higW_arg),len(lowW_arg))
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(mean_Delay_highW,dsi_kim[0])
    plt.ylabel('DSI')
    plt.xlim(20,35)
    plt.subplot(1,2,2)
    plt.scatter(mean_Delay_lowW,dsi_kim[0])
    plt.ylabel('DSI')
    plt.xlim(20,35)
    plt.savefig('./Output/DSI/w_delay_scatter')

    return -1

    ## save how many cells have a specific speed or spatF level (depending on max FR)
    n_speed = len(lvl_speed)
    n_spatF = len(lvl_spFrq)

    img_speed_spat = np.zeros((n_speed,n_spatF))
    for c in range(n_cells):
        pref_speed = int(preff_E1[0,c,0])
        pref_spatF = int(preff_E1[0,c,1])
        img_speed_spat[pref_speed,pref_spatF] +=1


    plt.figure()
    plt.imshow(img_speed_spat,cmap='viridis')
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    cbar = plt.colorbar()
    cbar.set_label('# of cells')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    plt.savefig('./Output/DSI/sF_to_speed_imshow')


    max_prefD = preff_E1[0,:,2]
    step_degree = int(360/n_steps)
    max_prefD = max_prefD*step_degree
    ### the moving direction is orthogonal to the stimulus orientation so add 90 degree 
    max_prefD = (max_prefD+90)%360

    bin_size=10
    bins = np.arange(0,360,bin_size)
    a,b = np.histogram(max_prefD, bins=np.arange(0,360+bin_size,bin_size))
    print(a,b)
    plt.figure()
    ax = plt.subplot(121,projection='polar')
    bars = ax.bar(bins/180*np.pi, a, width=0.2, bottom=0.1)
    ax = plt.subplot(122)
    plt.hist(max_prefD,bins=10)
    plt.savefig('./Output/DSI/hist_PreffD.png')

    ### compare the pref. orientation with the pref. direction
    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')
    params_E1 = params_E1[:,0,5]
    orienSteps = 8
    preffO_E1 = params_E1*orienSteps

    plt.figure()
    plt.scatter(preffO_E1%180,max_prefD%180)
    plt.xlabel('preffO')
    plt.ylabel('preffD')
    plt.savefig('./Output/DSI/preffO_preffD')

    print(np.max(max_prefD),np.min(max_prefD))
    print(np.max(preffO_E1),np.min(preffO_E1))

    diff = np.abs((max_prefD%180) - (preffO_E1%180))%90

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(diff, 'o')
    plt.xlabel('cell index')
    plt.ylabel('prefD - preffO')
    plt.subplot(2,1,2)
    plt.hist(diff,7)
    plt.xlabel('prefD - preffO')
    plt.ylabel('# cells')
    plt.savefig('./Output/DSI/preffO_minus_preffD')


    if not os.path.exists('Output/DSI/E1'):
        os.mkdir('Output/DSI/E1')

    # make the direct plots for preff DSI 

    # print spikes per cell and degree for max_DSI
    alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
    alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
    alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
    alpha = alpha/180. * np.pi #change in radian
                        
    spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')

    print(np.shape(spkC_E1_all))
    for c in range(n_cells):
        spk_cell = spkC_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,0]),:,:,c]
        spk_cell = np.mean(spk_cell,axis=1)
        spk_cell = np.append(spk_cell,spk_cell[0])

        grid = plt.GridSpec(2,2,wspace=0.4,hspace=0.3)
        plt.figure()
        ax = plt.subplot(grid[0,0],projection='polar')
        ax.plot( alpha,spk_cell,'o-')
        ax.grid(True)
        ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
        ax = plt.subplot(grid[0,1])
        ax.set_title('DSI = %f'%dsi_kim[0,c])
        ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        ax.axis('off')
        #ax = plt.subplot(grid[1,:])
        #ax.plot(np.mean(preff_vm_resh[c],axis=0),color='steelblue',label='pref')
        #ax.plot(np.mean(null_vm_resh[c],axis=0),color='tomato',label='null')
        #ax.legend() 
        plt.savefig('./Output/DSI/E1/direct_Polar_%i'%(c))
        plt.close()

def tcDSI():

    if not os.path.exists('Output/tc_DSI'):
        os.mkdir('Output/tc_DSI')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_spatFr = sim_param['speed']
    lvl_timeFr = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    

    print(lvl_spatFr,lvl_timeFr)

    s_deg = 18 # degree steps
    list_O_dS = np.linspace(0,360-s_deg,n_steps)

    ##create receptive fields (for later)
    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]

    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')

    params_E1 = params_E1[:,0,5]
    orienSteps = 8 # degree steps
    preffO = params_E1*orienSteps
    n_steps = int(360/orienSteps)
    list_O_TC = np.linspace(0,360-orienSteps,n_steps)

    ## load the spike counts
    spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')
    spkC_E1_all = spkC_E1_all[0] # only one contrast level at the moment !

    n_spatFr, n_timeFr, n_steps, n_repeats, n_cells = np.shape(spkC_E1_all)

    # Less different Orientations for the direction selectivity task then for the orientation selectivity task!
    # Find the matching pref. orientations and orientation-ID's for each neuron
    # NOTE: The preffered direction is orthogonal (90 degree) to the preffered orientation! 
    spkC_E1_preffO = np.zeros((n_cells, n_spatFr, n_timeFr,n_repeats))
    spkC_E1_nullO = np.zeros((n_cells, n_spatFr, n_timeFr,n_repeats))
    

    for c in range(n_cells):
        # calculate the nearest degree to the preffered orientation
        min_diff = np.argmin(np.abs(list_O_dS - preffO[c]))
        pref_O = list_O_dS[min_diff]
        ortho_O= (pref_O + 90)%360
        null_O = (ortho_O + 180)%360
        prefDS_idx = int(ortho_O/s_deg)
        nullDS_idx = int(null_O/s_deg)

        spkC_E1_preffO[c] = spkC_E1_all[:,:,prefDS_idx,:,c]
        spkC_E1_nullO[c] = spkC_E1_all[:,:,nullDS_idx,:,c]


    spkC_E1_preffO = np.mean(spkC_E1_preffO,axis=3) # mean over repeats
    spkC_E1_nullO = np.mean(spkC_E1_nullO,axis=3)
    
    plt.figure()
    plt.imshow(np.mean(spkC_E1_preffO,axis=0))
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    cbar = plt.colorbar()
    cbar.set_label('# of cells')
    plt.xticks(np.linspace(0,len(lvl_timeFr)-1,len(lvl_timeFr)), lvl_timeFr)
    plt.yticks(np.linspace(0,len(lvl_spatFr)-1,len(lvl_spatFr)), lvl_spatFr)
    plt.savefig('Output/tc_DSI/sF_to_speed_imshow_meanSpk')


    dsi_maz = np.zeros(n_cells) #DSI after Mazurek et al. (2014) // (R_pref - R_null)/(R_pref)
    dsi_will = np.zeros(n_cells)#DSI after Willson et al. (2018)// (R_pref - R_null)/(R_pref + R_null)
    dsi_kim = np.zeros(n_cells)#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)

    for c in range(n_cells):
        preffs_idx = np.where(spkC_E1_preffO[c] == np.max(spkC_E1_preffO[c]))
        prefSpk = spkC_E1_preffO[c,preffs_idx[0][0],preffs_idx[1][0]]
        nullSpk = spkC_E1_nullO[c,preffs_idx[0][0],preffs_idx[1][0]]
        ## swap the activies if necessary
        if prefSpk < nullSpk:
            tmpA = np.copy(nullSpk)
            tmpB = np.copy(prefSpk)
            nullSpk = tmpB
            prefSpk = tmpA

        dsi_maz[c] = (prefSpk - nullSpk)/prefSpk
        dsi_will[c] = (prefSpk - nullSpk)/(prefSpk + nullSpk)
        dsi_kim[c] = 1- (nullSpk/prefSpk)

    plt.figure()
    plt.hist(dsi_maz) 
    plt.ylabel('#cells')
    plt.xlabel('DSI Maz')
    plt.savefig('Output/tc_DSI/hist_DSI_maz')
    
    plt.figure()
    plt.hist(dsi_will) 
    plt.ylabel('#cells')
    plt.xlabel('DSI Will')
    plt.savefig('Output/tc_DSI/hist_DSI_will')

    plt.figure()
    plt.hist(dsi_kim) 
    plt.ylabel('#cells')
    plt.xlabel('DSI Kim')
    plt.savefig('Output/tc_DSI/hist_DSI_kim')


#------------------------------------------------------------------------------
if __name__=="__main__":

    #main()
    moreDSI()
    #tcDSI()

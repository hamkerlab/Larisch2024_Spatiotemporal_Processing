import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def calcDSI(data, totalT, name,c_lvl,sf_lvl,spatF_LVL):
    print('Mean SpkC= ', np.mean(data))
    n_steps,n_repeats, n_cells = np.shape(data)
    step_degree = int(360/n_steps)
    mean_overR_spkC = np.mean(data,axis=1)

    print('Calculate DSI for '+name)
    dsi_maz = np.zeros(n_cells) #DSI after Mazurek et al. (2014) // (R_pref - R_null)/(R_pref)
    dsi_will = np.zeros(n_cells)#DSI after Willson et al. (2018)// (R_pref - R_null)/(R_pref + R_null)
    dsi_kim = np.zeros(n_cells)#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)

    for c in range(n_cells):
        spk_cell = mean_overR_spkC[:,c]
        preff_arg = np.argmax(spk_cell)
        preff_D = preff_arg*step_degree
        null_D = (preff_D+180)%360
        null_arg = int(null_D/step_degree)
        dsi_maz[c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg])
        dsi_will[c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg] + spk_cell[null_arg] )
        dsi_kim[c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])

    plt.figure()
    plt.hist(dsi_maz[~np.isnan(dsi_maz)],10)
    plt.axvline(np.mean(dsi_maz[~np.isnan(dsi_maz)]),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/dsi_maz_hist_'%(c_lvl,sf_lvl,spatF_LVL)+(name))
    plt.close()

    plt.figure()
    plt.hist(dsi_will[~np.isnan(dsi_will)],10)
    plt.axvline(np.mean(dsi_will[~np.isnan(dsi_will)]),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/dsi_will_hist_'%(c_lvl,sf_lvl,spatF_LVL)+(name))
    plt.close()

    plt.figure()
    plt.hist(dsi_kim[~np.isnan(dsi_kim)],10)
    plt.axvline(np.mean(dsi_kim[~np.isnan(dsi_kim)]),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/dsi_kim_hist_'%(c_lvl,sf_lvl,spatF_LVL)+(name))
    plt.close()

def evalDSI(totalT,cont_LVL,speed_LVL,spatF_LVL):

    #spkC_E2 = np.load('./work/directGrating_Sinus_SpikeCount_E2.npy')
    spkC_I1 = np.load('./work/directGrating_Sinus_SpikeCount_I1.npy') 
    #spkC_I2 = np.load('./work/directGrating_Sinus_SpikeCount_I2.npy')

    #spkC_E2 = spkC_E2[cont_LVL]
    spkC_I1 = spkC_I1[cont_LVL,speed_LVL,spatF_LVL]
    #spkC_I2 = spkC_I2[cont_LVL]

    #calcDSI(spkC_E2,totalT,'E2',cont_LVL)

    calcDSI(spkC_I1,totalT,'I1',cont_LVL,speed_LVL,spatF_LVL)

    #calcDSI(spkC_I2,totalT,'I2',cont_LVL)

    return()

def main():
    print('Analyse direction selectivity by preseting a moving Sinus')


    spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')
    vm_E1_all = np.load('./work/directGrating_Sinus_MembranPot_E1.npy')
    gE_E1_all = np.load('./work/directGrating_Sinus_gExc_E1.npy')
    gI_E1_all = np.load('./work/directGrating_Sinus_gInh_E1.npy')

    spkC_I1_all = np.load('./work/directGrating_Sinus_SpikeCount_I1.npy')
    vm_I1_all = np.load('./work/directGrating_Sinus_MembranPot_I1.npy')
    gE_I1_all = np.load('./work/directGrating_Sinus_gExc_I1.npy')
    gI_I1_all = np.load('./work/directGrating_Sinus_gInh_I1.npy')


    spkC_LGN_all = np.load('./work/directGrating_Sinus_SpikeCount_LGN.npy')

    print('Mean FR LGN = ',np.mean(spkC_LGN_all))
    print(np.shape(vm_E1_all))
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

    dsi_kim_all = np.zeros((n_contrast,n_speeds,n_spatF,n_cellsE1))

    preffD_E1 = np.zeros((n_contrast,n_speeds,n_spatF,n_cellsE1))

    for i in range(n_contrast):

        if not os.path.exists('Output/direct_gratingSinus/LVL_%i'%(i)):
            os.mkdir('Output/direct_gratingSinus/LVL_%i'%(i))

    # get over speed levels 
        for s in range(n_speeds):
            if not os.path.exists('Output/direct_gratingSinus/LVL_%i/speed_%i'%(i,s)):
                os.mkdir('Output/direct_gratingSinus/LVL_%i/speed_%i'%(i,s))

            # get over spat. Frequencies
            for f in range(n_spatF):

                if not os.path.exists('Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i'%(i,s,f)):
                    os.mkdir('Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i'%(i,s,f))
                if not os.path.exists('Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/E1'%(i,s,f)):
                    os.mkdir('Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/E1'%(i,s,f))
                if not os.path.exists('Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/I1'%(i,s,f)):
                    os.mkdir('Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/I1'%(i,s,f))


                spkC_E1 = spkC_E1_all[i,s,f] 
                vm_E1 = vm_E1_all[i,s,f]
                gE_E1 = gE_E1_all[i,s,f]
                gI_E1 = gI_E1_all[i,s,f]


                #print(np.shape(spkC_E1))
                print('Mean Spike for contrast %i, speed %i, spatFq %i: %f'%(i,s,f,np.mean(spkC_E1)))
                if np.mean(spkC_E1) == 0:
                    break
                else:

    
                    spkC_I1 = spkC_I1_all[i,s,f] 
                    vm_I1 = vm_I1_all[i,s,f]
                    gE_I1 = gE_I1_all[i,s,f]
                    gI_I1 = gI_I1_all[i,s,f]
                    

                    spkC_LGN = spkC_LGN_all[i,s,f]


                    n_steps,n_repeats,totalT,n_cellsE1 = np.shape(vm_E1)
                    n_cellsI1 = np.shape(vm_I1)[3]

                    step_degree = int(360/n_steps)    

                    mean_overR_vm_E1 = np.mean(vm_E1,axis=1)
                    mean_overR_gE_E1 = np.mean(gE_E1,axis=1)
                    mean_overR_gI_E1 = np.mean(gI_E1,axis=1)

                    mean_overR_spkC_E1 = np.mean(spkC_E1,axis=1)

                    mean_overR_vm_I1 = np.mean(vm_I1,axis=1)
                    mean_overR_gE_I1 = np.mean(gE_I1,axis=1)
                    mean_overR_gI_I1 = np.mean(gI_I1,axis=1)

                    mean_overR_spkC_I1 = np.mean(spkC_I1,axis=1)

                    print('Calculate DSI')
                    dsi_maz = np.zeros(n_cellsE1) #DSI after Mazurek et al. (2014) // (R_pref - R_null)/(R_pref)
                    dsi_will = np.zeros(n_cellsE1)#DSI after Willson et al. (2018)// (R_pref - R_null)/(R_pref + R_null)
                    dsi_kim = np.zeros(n_cellsE1)#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)


                    evalDSI(totalT,i,s,f)

                    preff_vm=np.zeros((n_cellsE1,totalT))
                    null_vm =np.zeros((n_cellsE1,totalT))

                    preff_gE=np.zeros((n_cellsE1,totalT))
                    null_gE =np.zeros((n_cellsE1,totalT))

                    preff_gI=np.zeros((n_cellsE1,totalT))
                    null_gI =np.zeros((n_cellsE1,totalT))


                    preff_vm_I1=np.zeros((n_cellsI1,totalT))
                    null_vm_I1 =np.zeros((n_cellsI1,totalT))

                    preff_gE_I1=np.zeros((n_cellsI1,totalT))
                    null_gE_I1 =np.zeros((n_cellsI1,totalT))

                    preff_gI_I1=np.zeros((n_cellsI1,totalT))
                    null_gI_I1 =np.zeros((n_cellsI1,totalT))



                    for c in range(n_cellsE1):
                        spk_cell = mean_overR_spkC_E1[:,c]
                        preff_arg = np.argmax(spk_cell)
                        preff_D = preff_arg*step_degree
                        preffD_E1[i,s,f,c] = preff_D
                        null_D = (preff_D+180)%360
                        null_arg = int(null_D/step_degree)
                        dsi_maz[c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg])
                        dsi_will[c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg] + spk_cell[null_arg] )
                        dsi_kim[c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])

                        preff_vm[c] = mean_overR_vm_E1[preff_arg,:,c]
                        null_vm[c] = mean_overR_vm_E1[null_arg,:,c]

                        preff_gE[c] = mean_overR_gE_E1[preff_arg,:,c]
                        null_gE[c] = mean_overR_gE_E1[null_arg,:,c]

                        preff_gI[c] = mean_overR_gI_E1[preff_arg,:,c]
                        null_gI[c] = mean_overR_gI_E1[null_arg,:,c]

                    for c in range(n_cellsI1):
                        spk_cell = mean_overR_spkC_E1[:,c]
                        preff_arg = np.argmax(spk_cell)
                        preff_D = preff_arg*step_degree
                        null_D = (preff_D+180)%360
                        null_arg = int(null_D/step_degree)

                        preff_vm_I1[c] = mean_overR_vm_I1[preff_arg,:,c]
                        null_vm_I1[c] = mean_overR_vm_I1[null_arg,:,c]

                        preff_gE_I1[c] = mean_overR_gE_I1[preff_arg,:,c]
                        null_gE_I1[c] = mean_overR_gE_I1[null_arg,:,c]

                        preff_gI_I1[c] = mean_overR_gI_I1[preff_arg,:,c]
                        null_gI_I1[c] = mean_overR_gI_I1[null_arg,:,c]

                    dsi_kim_all[i,s,f] = dsi_kim

                    cycleT = 250#int(1000/(lvl_spFrq[f]))
                    n_cycles = int(totalT/cycleT)

                    preff_vm_resh = np.reshape(preff_vm,(n_cellsE1,n_cycles,cycleT))
                    null_vm_resh = np.reshape(null_vm,(n_cellsE1,n_cycles,cycleT))
                    
                    preff_vm_I1_resh = np.reshape(preff_vm_I1,(n_cellsI1,n_cycles,cycleT))
                    null_vm_I1_resh = np.reshape(null_vm_I1,(n_cellsI1,n_cycles,cycleT))

                    plt.figure()
                    plt.hist(np.mean(preff_gE,axis=1),24,label='gExc', alpha = 0.5)
                    plt.hist(np.mean(preff_gI,axis=1),24,label='gInh', alpha = 0.5)
                    plt.legend()
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/gEx_gInh_preff_E1_hist.png'%(i,s,f))
                    plt.close()

                    plt.figure()
                    plt.hist(np.mean(null_gE,axis=1),24,label='gExc', alpha = 0.5)
                    plt.hist(np.mean(null_gI,axis=1),24,label='gInh', alpha = 0.5)
                    plt.legend()
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/gEx_gInh_null_E1_hist.png'%(i,s,f))
                    plt.close()


                    plt.figure()
                    plt.hist(np.mean(preff_gE_I1,axis=1),24,label='gExc', alpha = 0.5)
                    plt.hist(np.mean(preff_gI_I1,axis=1),24,label='gInh', alpha = 0.5)
                    plt.legend()
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/gEx_gInh_pref_I1_hist.png'%(i,s,f))

                    plt.figure()
                    plt.hist(np.mean(null_gE_I1,axis=1),24,label='gExc', alpha = 0.5)
                    plt.hist(np.mean(null_gI_I1,axis=1),24,label='gInh', alpha = 0.5)
                    plt.legend()
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/gEx_gInh_null_I1_hist.png'%(i,s,f))




                    print(np.shape(preff_vm))

                    plt.figure()
                    plt.plot(np.mean(preff_vm_resh[100],axis=0),color='steelblue',label='pref')
                    plt.plot(np.mean(null_vm_resh[100],axis=0),color='tomato',label='null')
                    plt.legend()
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/E1/preff_null_VM_100_resh.png'%(i,s,f))

                    plt.figure()
                    plt.plot(preff_vm[100,0:250],color='steelblue',label='pref')
                    plt.plot(null_vm[100,0:250],color='tomato',label='null')
                    plt.legend()
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/E1/preff_null_VM_100.png'%(i,s,f))



                    plt.figure()
                    plt.hist(dsi_maz,10)
                    plt.axvline(np.mean(dsi_maz),color='black',linestyle='--',linewidth=4)
                    plt.xlim(0,1)
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/dsi_maz_hist_E1'%(i,s,f))
                    plt.close()

                    plt.figure()
                    plt.hist(dsi_will,10)
                    plt.axvline(np.mean(dsi_will),color='black',linestyle='--',linewidth=4)
                    plt.xlim(0,1)
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/dsi_will_hist_E1'%(i,s,f))
                    plt.close()

                    plt.figure()
                    plt.hist(dsi_kim,10)
                    plt.axvline(np.mean(dsi_kim),color='black',linestyle='--',linewidth=4)
                    plt.xlim(0,1)
                    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/dsi_kim_hist_E1'%(i,s,f))
                    plt.close()



                    
                    alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
                    alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
                    alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
                    alpha = alpha/180. * np.pi #change in radian
                        
                    for c in range(n_cellsE1):
                        spk_cell = mean_overR_spkC_E1[:,c]
                        spk_cell = np.append(spk_cell,spk_cell[0])

                        grid = plt.GridSpec(2,2,wspace=0.4,hspace=0.3)
                        plt.figure()
                        ax = plt.subplot(grid[0,0],projection='polar')
                        ax.plot( alpha,spk_cell,'o-')
                        ax.grid(True)
                        ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
                        ax = plt.subplot(grid[0,1])
                        ax.set_title('DSI = %f'%dsi_maz[c])
                        ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
                        ax.axis('off')
                        ax = plt.subplot(grid[1,:])
                        ax.plot(np.mean(preff_vm_resh[c],axis=0),color='steelblue',label='pref')
                        ax.plot(np.mean(null_vm_resh[c],axis=0),color='tomato',label='null')
                        ax.legend() 
                        plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/E1/direct_Polar_%i'%(i,s,f,c))
                        plt.close()


                    #alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
                    #alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
                    #alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
                    #alpha = alpha/180. * np.pi #change in radian
                        
                    #for c in range(n_cellsI1):
                    #    spk_cell = mean_overR_spkC_I1[:,c]
                    #    spk_cell = np.append(spk_cell,spk_cell[0])

                    #    grid = plt.GridSpec(2,2,wspace=0.4,hspace=0.3)
                    #    plt.figure()
                    #    ax = plt.subplot(grid[0,0],projection='polar')
                    #    ax.plot( alpha,spk_cell,'o-')
                    #    ax.grid(True)
                    #    ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
                    #    ax = plt.subplot(grid[0,1])
                    #    ax.imshow(rf_I1[c],cmap='gray',aspect='equal',interpolation='none')
                    #    ax.axis('off')
                    #    ax = plt.subplot(grid[1,:])
                    #    ax.plot(np.mean(preff_vm_I1_resh[c],axis=0),color='steelblue',label='pref')
                    #    ax.plot(np.mean(null_vm_I1_resh[c],axis=0),color='tomato',label='null')
                    #    ax.legend() 
                    #    plt.savefig('./Output/direct_gratingSinus/LVL_%i/speed_%i/spatF_%i/I1/direct_Polar_%i'%(i,s,f,c))
                    #    plt.close()
                    
    np.save('./work/dsi_kim',dsi_kim_all,allow_pickle=False)
    np.save('./work/direct_SinusGratings_preffD_E1',preffD_E1,allow_pickle=False)
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

    prefD_E1 = np.load('./work/direct_SinusGratings_preffD_E1.npy')
    print(np.shape(prefD_E1))
    dsi_kim = np.load('./work/dsi_kim.npy')
    #print(np.shape(dsi_kim))
    # ignore the first spat-Freq. (0.75 cycle/Image
    lvl_spFrq = lvl_spFrq[1:]
    prefD_E1 = prefD_E1[:,:,1:,:]
    dsi_kim = dsi_kim[:,:,1:,:]

    n_amp,n_speed,n_sF,n_cells = (np.shape(dsi_kim))
    # get the index of max DSI for each cells / ignore speedLVL 0 and 1
    max_dsi = np.zeros((n_amp,n_cells))
    max_prefD = np.zeros((n_amp,n_cells)) 

    # sum dsi over complete population
    dsi_kim_sum = np.nanmean(dsi_kim[0],axis=2)#only for ampl = 1
    print('Best pop DSI :',np.where( dsi_kim_sum == np.max(dsi_kim_sum)))
    best_idx = np.where( dsi_kim_sum == np.max(dsi_kim_sum))
    
    plt.figure()
    plt.hist(dsi_kim[0,best_idx[0][0],best_idx[1][0]])
    plt.savefig('./Output/direct_gratingSinus/maxDSI/hist_DSI_speed%i_sF_%i.png'%(best_idx[0][0],best_idx[1][0]))

    # save the index of preffered speed and spatial Frequency !
    idx_speed = np.zeros((n_amp,n_cells)) 
    idx_sF = np.zeros((n_amp,n_cells))
    for a in range(n_amp):
        for c in range(n_cells):
            idx = np.where(dsi_kim[a,:,:,c] ==np.max(dsi_kim[a,:,:,c]))
            max_dsi[a,c] = dsi_kim[a,idx[0],idx[1],c][0]
            idx_speed[a,c] = idx[0][0]
            idx_sF[a,c] = idx[1][0]
            max_prefD[a,c] = prefD_E1[a,idx[0],idx[1],c][0]

    bin_size=10
    bins = np.arange(0,360,bin_size)
    a,b = np.histogram(max_prefD[0], bins=np.arange(0,360+bin_size,bin_size))

    plt.figure()
    ax = plt.subplot(121,projection='polar')
    bars = ax.bar(bins/180*np.pi, a, width=0.2, bottom=0.1)
    ax = plt.subplot(122)
    plt.hist(max_prefD[0],bins=20)
    plt.savefig('./Output/direct_gratingSinus/maxDSI/hist_PreffD.png')


    # count the number of cells, which have a specific spat. Freq to a specific input speed
    
    idx_speed_l = np.linspace(0,len(lvl_speed)-1,len(lvl_speed),dtype='int16')
    idx_spatF_l = np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq),dtype='int16')
    n_speed_sF = np.zeros((len(lvl_speed),len(lvl_spFrq)))
    for s in range(n_speed):
        idx = np.where(idx_speed[0,:] == idx_speed_l[s])[0] # select neurons with preffered speed
        selectd_Cells = idx_sF[0,idx]
        for f in range(n_sF):
            idx_2 = np.where(selectd_Cells == idx_spatF_l[f])[0]
            n_speed_sF[s,f] = len(idx_2)

    ## get the mean RF of the neurons which are selectivate for a specific direction ##

    step_degree = int(360/n_steps) 
    l_degree =np.linspace(0,360-step_degree,n_steps,dtype='int32')
    alpha = np.roll(l_degree,int(90/step_degree)) 

    list_RFs = np.zeros((n_steps,int(np.sqrt(n_pre/2)),int(np.sqrt(n_pre/2))))
    for d in range(n_steps):
        idx_d = np.where(max_prefD[0,:] == l_degree[d])[0]
        list_RFs[d] = np.mean(rf_E1[idx_d],axis=0)
        plt.figure()
        for i in range(8):
            plt.subplot(3,3,i+1)
            plt.axis('off')
            plt.imshow(rf_E1[idx_d[i]],cmap=plt.get_cmap('gray'),aspect='auto',interpolation='none')
            plt.axis('equal')
        plt.subplot(3,3,9) 
        plt.axis('off')
        plt.imshow(list_RFs[d],cmap=plt.get_cmap('gray'),aspect='auto',interpolation='none')
        plt.axis('equal')
        plt.savefig('./Output/direct_gratingSinus/maxDSI/RFtoDegree_%i'%alpha[d])
        plt.close()

    plt.figure()
    plt.imshow(n_speed_sF,cmap='viridis')
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    cbar = plt.colorbar()
    cbar.set_label('# of cells')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/sF_to_speed_imshow')

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(np.nanmean(dsi_kim[0],axis=2),cmap='viridis',vmin=0,vmax=1)
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.nanmean(dsi_kim[0],axis=2),cmap='viridis',vmin=0,vmax=1)
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    plt.colorbar()
    plt.savefig('./Output/direct_gratingSinus/maxDSI/meanDSI_perSpeed_persF')

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(np.nansum(dsi_kim[0],axis=2),cmap='viridis')
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.nansum(dsi_kim[0],axis=2),cmap='viridis')
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    plt.colorbar()
    plt.savefig('./Output/direct_gratingSinus/maxDSI/Pop_Sum_DSI_perSpeed_persF')



    plt.figure()
    plt.subplot(121)
    plt.hist(max_dsi[0])
    plt.ylabel('# cells')
    plt.xlabel('maxDSI amp = 0')
    plt.subplot(122)
    plt.hist(max_dsi[0])
    plt.ylabel('# cells')
    plt.xlabel('maxDSI amp = 1')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/hist_maxDSI')

    plt.figure()
    plt.subplot(121)
    plt.hist(idx_speed[0],bins=n_speed)
    plt.ylabel('# cells')
    plt.xlabel('pref Speed (idx) amp = 0')
    plt.subplot(122)
    plt.hist(idx_speed[0],bins=n_speed)
    plt.ylabel('# cells')
    plt.xlabel('pref Speed (idx) amp = 1')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/hist_SpeedIdx')

    plt.figure()
    plt.subplot(121)
    plt.hist(idx_sF[0],bins=n_sF)
    plt.ylabel('# cells')
    plt.xlabel('pref spat. Freq. (idx) amp = 0')
    plt.subplot(122)
    plt.hist(idx_sF[0],bins=n_sF)
    plt.ylabel('# cells')
    plt.xlabel('pref spat. Freq. (idx) amp = 1')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/hist_spatFreq')



    # make the direct plots for preff DSI again

    if not os.path.exists('Output/direct_gratingSinus/maxDSI/E1'):
        os.mkdir('Output/direct_gratingSinus/maxDSI/E1')
    
    spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')
    spkC_E1_all = spkC_E1_all[0]
    #print(np.shape(spkC_E1_all))




    # print spikes per cell and degree for max_DSI
    alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
    alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
    alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
    alpha = alpha/180. * np.pi #change in radian
                        

    for c in range(n_cells):
        spk_cell = spkC_E1_all[int(idx_speed[0,c]),int(idx_sF[0,c]),:,:,c ]
        spk_cell = np.mean(spk_cell,axis=1)
        spk_cell = np.append(spk_cell,spk_cell[0])

        grid = plt.GridSpec(2,2,wspace=0.4,hspace=0.3)
        plt.figure()
        ax = plt.subplot(grid[0,0],projection='polar')
        ax.plot( alpha,spk_cell,'o-')
        ax.grid(True)
        ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
        ax = plt.subplot(grid[0,1])
        ax.set_title('max DSI = %f'%max_dsi[0,c])
        ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        ax.axis('off')
        #ax = plt.subplot(grid[1,:])
        #ax.plot(np.mean(preff_vm_resh[c],axis=0),color='steelblue',label='pref')
        #ax.plot(np.mean(null_vm_resh[c],axis=0),color='tomato',label='null')
        #ax.legend() 
        plt.savefig('./Output/direct_gratingSinus/maxDSI/E1/direct_Polar_%i'%(c))
        plt.close()


    return -1
    ### plot the STRF's in order to the max DSI ###
    # load STRFs
    strfs_E1 = np.load('./work/STRF_data_E1.npy',allow_pickle = True)
    list_STRF = []
    for i in range(n_cells):
        list_STRF.append(np.sum(strfs_E1[i],axis=2))
    
    idx_maxDSI = np.argsort(max_dsi[0]*-1)


    fig = plt.figure(figsize=(12,12))
    maxV = np.max(np.max(list_STRF,axis=1),axis=1)
    minV = np.min(np.min(list_STRF,axis=1),axis=1)
    h = int(np.sqrt(n_post))
    for i in range(n_post):
        plt.subplot(h,h,i+1)
        plt.axis('off')
        plt.imshow(list_STRF[idx_maxDSI[i]],cmap=plt.get_cmap('gray',7),aspect='auto',interpolation='none',vmin=minV[i],vmax=maxV[i])
        plt.title(np.round(max_dsi[0,idx_maxDSI[i]],2))
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    fig.savefig('./Output/direct_gratingSinus/maxDSI/STRF_sortMaxDSI.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)


#------------------------------------------------------------------------------
if __name__=="__main__":

    #main()
    moreDSI()

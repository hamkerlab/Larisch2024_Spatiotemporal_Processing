import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def main():

    if not os.path.exists('Output/direct_gratingSinus/maxDSI/maxDSI/'):
        os.mkdir('Output/direct_gratingSinus/maxDSI/maxDSI/')
    if not os.path.exists('Output/direct_gratingSinus/maxDSI/maxDSI/E1/'):
        os.mkdir('Output/direct_gratingSinus/maxDSI/maxDSI//E1/')

    if not os.path.exists('Output/direct_gratingSinus/maxDSI/maxDSI/I1/'):
        os.mkdir('Output/direct_gratingSinus/maxDSI/maxDSI//I1/')

    spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1_maxDSI.npy')
    spkC_I1_all = np.load('./work/directGrating_Sinus_SpikeCount_I1_maxDSI.npy')
    spkC_LGN_all = np.load('./work/directGrating_Sinus_SpikeCount_LGN_maxDSI.npy')
    
    mp_E1_all = np.load('./work/directGrating_Sinus_MembranPot_E1_maxDSI.npy')
    gE_E1_all = np.load('./work/directGrating_Sinus_gExc_E1_maxDSI.npy')
    gI_E1_all = np.load('./work/directGrating_Sinus_gInh_E1_maxDSI.npy')

    mp_I1_all = np.load('./work/directGrating_Sinus_MembranPot_I1_maxDSI.npy')
    gE_I1_all = np.load('./work/directGrating_Sinus_gExc_I1_maxDSI.npy')
    gI_I1_all = np.load('./work/directGrating_Sinus_gInh_I1_maxDSI.npy')

    print()
    n_time = np.shape(mp_E1_all)[3]

    print('Mean FR LGN = ',np.mean(spkC_LGN_all))
    n_contrast,n_degress,repeats,n_cellsE1 = np.shape(spkC_E1_all)


    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]


    w_I1 = np.loadtxt('./Input_network/InhibW.txt')
    n_post,n_pre = np.shape(w_I1)
    rf_I1 = np.reshape(w_I1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_I1 = rf_I1[:,:,:,0] - rf_I1[:,:,:,1]
    n_cellsI1 = n_post

    dsi_kim_all_E1 = np.zeros((n_contrast,n_cellsE1))
    dsi_kim_all_I1 = np.zeros((n_contrast,n_cellsI1))

    preffD_E1 = np.zeros((n_contrast,n_cellsE1))
    preffD_I1 = np.zeros((n_contrast,n_cellsI1))

    preff_mp_E1 = np.zeros((n_contrast,n_cellsE1,repeats,n_time))
    null_mp_E1 = np.zeros((n_contrast,n_cellsE1,repeats,n_time))

    preff_gE_E1 = np.zeros((n_contrast,n_cellsE1,repeats,n_time))
    null_gE_E1 = np.zeros((n_contrast,n_cellsE1,repeats,n_time))

    preff_gI_E1 = np.zeros((n_contrast,n_cellsE1,repeats,n_time))
    null_gI_E1 = np.zeros((n_contrast,n_cellsE1,repeats,n_time))

    #########

    preff_mp_I1 = np.zeros((n_contrast,n_cellsI1,repeats,n_time))
    null_mp_I1 = np.zeros((n_contrast,n_cellsI1,repeats,n_time))

    preff_gE_I1 = np.zeros((n_contrast,n_cellsI1,repeats,n_time))
    null_gE_I1 = np.zeros((n_contrast,n_cellsI1,repeats,n_time))

    preff_gI_I1 = np.zeros((n_contrast,n_cellsI1,repeats,n_time))
    null_gI_I1 = np.zeros((n_contrast,n_cellsI1,repeats,n_time))

    for i in range(n_contrast):

        spkC_E1 = spkC_E1_all[i] 


        #print(np.shape(spkC_E1))
        print('Mean Spike for contrast %i : %f'%(i,np.mean(spkC_E1)))
        if np.mean(spkC_E1) == 0:
            break
        else:
            spkC_I1 = spkC_I1_all[i] 
            spkC_LGN = spkC_LGN_all[i]
            n_steps = n_degress#np.shape(vm_E1)
            n_cellsI1 = int(n_cellsE1/4) #np.shape(vm_I1)[3]

            step_degree = int(360/n_steps)    

            mean_overR_spkC_E1 = np.mean(spkC_E1,axis=1)
            mean_overR_spkC_I1 = np.mean(spkC_I1,axis=1)

            print('Calculate DSI')

            for c in range(n_cellsE1):
                spk_cell = mean_overR_spkC_E1[:,c]
                preff_arg = np.argmax(spk_cell)
                preff_mp_E1[i,c]= mp_E1_all[i,preff_arg,:,:,c]
                preff_gE_E1[i,c]= gE_E1_all[i,preff_arg,:,:,c]
                preff_gI_E1[i,c]= gI_E1_all[i,preff_arg,:,:,c]
                preff_D = preff_arg*step_degree
                preffD_E1[i,c] = preff_D
                null_D = (preff_D+180)%360
                null_arg = int(null_D/step_degree)
                null_mp_E1[i,c]= mp_E1_all[i,null_arg,:,:,c]
                null_gE_E1[i,c]= gE_E1_all[i,null_arg,:,:,c]
                null_gI_E1[i,c]= gI_E1_all[i,null_arg,:,:,c]
                #dsi_maz[c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg])
                #dsi_will[c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg] + spk_cell[null_arg] )
                dsi_kim_all_E1[i,c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])

            for c in range(n_cellsI1):
                spk_cell = mean_overR_spkC_I1[:,c]
                preff_arg = np.argmax(spk_cell)
                preff_mp_I1[i,c]= mp_I1_all[i,preff_arg,:,:,c]
                preff_gE_I1[i,c]= gE_I1_all[i,preff_arg,:,:,c]
                preff_gI_I1[i,c]= gI_I1_all[i,preff_arg,:,:,c]
                preff_D = preff_arg*step_degree
                preffD_I1[i,c] = preff_D
                null_D = (preff_D+180)%360
                null_arg = int(null_D/step_degree)
                null_mp_I1[i,c]= mp_I1_all[i,null_arg,:,:,c]
                null_gE_I1[i,c]= gE_I1_all[i,null_arg,:,:,c]
                null_gI_I1[i,c]= gI_I1_all[i,null_arg,:,:,c]
                dsi_kim_all_I1[i,c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])

            break
            # print spikes per cell and degree for max_DSI
            alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
            alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
            alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
            alpha = alpha/180. * np.pi #change in radian
                                
            
            for c in range(n_cellsE1):
                spk_cell = mean_overR_spkC_E1[:,c ]
                spk_cell = np.append(spk_cell,spk_cell[0])

                preff_mp = np.mean(preff_mp_E1[i,c],axis=0)
                null_mp = np.mean(null_mp_E1[i,c],axis=0)

                preff_gE = np.mean(preff_gE_E1[i,c],axis=0)
                null_gE = np.mean(null_gE_E1[i,c],axis=0)

                preff_gI = np.mean(preff_gI_E1[i,c],axis=0)
                null_gI = np.mean(null_gI_E1[i,c],axis=0)

                preff_mp = np.reshape(preff_mp,(1000,3))
                null_mp = np.reshape(null_mp,(1000,3))

                preff_gE = np.reshape(preff_gE,(1000,3))
                null_gE = np.reshape(null_gE,(1000,3))

                preff_gI = np.reshape(preff_gI,(1000,3))
                null_gI = np.reshape(null_gI,(1000,3))

                preff_mp = np.mean(preff_mp,axis=1)
                null_mp = np.mean(null_mp,axis=1)

                preff_gE = np.mean(preff_gE,axis=1)
                null_gE = np.mean(null_gE,axis=1)

                preff_gI = np.mean(preff_gI,axis=1)
                null_gI = np.mean(null_gI,axis=1)

                grid = plt.GridSpec(4,2,wspace=0.4,hspace=0.3)
                plt.figure(figsize=(15,12))
                ax = plt.subplot(grid[0,0],projection='polar')
                ax.plot( alpha,spk_cell,'o-')
                ax.grid(True)
                ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
                ax = plt.subplot(grid[0,1])
                ax.set_title('max DSI = %f'%np.max(dsi_kim_all_E1[:,c]))
                ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
                ax.axis('off')
                ax = plt.subplot(grid[1,:])
                ax.plot(preff_mp[0:100],color='steelblue',label='pref')
                ax.plot(null_mp[0:100],color='tomato',label='null')
                ax.legend() 

                ax = plt.subplot(grid[2,:])
                ax.plot(preff_gE[0:100],color='seagreen',label='gEx')
                ax.plot( preff_gE[0:100] - preff_gI[0:100],color='steelblue',label='gEx - gInh')
                ax.plot(-preff_gI[0:100],color='tomato',label='gInh')
                ax.hlines(0,0,100,ls = '--',color='slategray')
                ax.set_ylim(-100,100)
                ax.legend() 

                ax = plt.subplot(grid[3,:])
                ax.plot(null_gE[0:100],color='seagreen',label='gEx')
                ax.plot(null_gE[0:100]-null_gI[0:100],color='steelblue',label='gEx - gInh')
                ax.plot(-null_gI[0:100],color='tomato',label='gInh')
                ax.hlines(0,0,100,ls = '--',color='slategray')
                ax.set_ylim(-100,100)
                ax.legend() 

                plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/E1/direct_Polar_%i'%(c),bbox_inches='tight', pad_inches = 0.1)
                plt.close()

            for c in range(n_cellsI1):
                spk_cell = mean_overR_spkC_I1[:,c ]
                spk_cell = np.append(spk_cell,spk_cell[0])

                preff_mp = np.mean(preff_mp_I1[i,c],axis=0)
                null_mp = np.mean(null_mp_I1[i,c],axis=0)

                preff_gE = np.mean(preff_gE_I1[i,c],axis=0)
                null_gE = np.mean(null_gE_I1[i,c],axis=0)

                preff_gI = np.mean(preff_gI_I1[i,c],axis=0)
                null_gI = np.mean(null_gI_I1[i,c],axis=0)

                preff_mp = np.reshape(preff_mp,(1000,3))
                null_mp = np.reshape(null_mp,(1000,3))

                preff_gE = np.reshape(preff_gE,(1000,3))
                null_gE = np.reshape(null_gE,(1000,3))

                preff_gI = np.reshape(preff_gI,(1000,3))
                null_gI = np.reshape(null_gI,(1000,3))

                preff_mp = np.mean(preff_mp,axis=1)
                null_mp = np.mean(null_mp,axis=1)

                preff_gE = np.mean(preff_gE,axis=1)
                null_gE = np.mean(null_gE,axis=1)

                preff_gI = np.mean(preff_gI,axis=1)
                null_gI = np.mean(null_gI,axis=1)

                grid = plt.GridSpec(4,2,wspace=0.4,hspace=0.3)
                plt.figure(figsize=(15,12))
                ax = plt.subplot(grid[0,0],projection='polar')
                ax.plot( alpha,spk_cell,'o-')
                ax.grid(True)
                ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
                ax = plt.subplot(grid[0,1])
                ax.set_title('max DSI = %f'%np.max(dsi_kim_all_I1[:,c]))
                ax.imshow(rf_I1[c],cmap='gray',aspect='equal',interpolation='none')
                ax.axis('off')
                ax = plt.subplot(grid[1,:])
                ax.plot(preff_mp[0:100],color='steelblue',label='pref')
                ax.plot(null_mp[0:100],color='tomato',label='null')
                ax.legend() 
                ax = plt.subplot(grid[2,:])
                ax.plot(preff_gE[0:100],color='seagreen',label='gEx')
                ax.plot( preff_gE[0:100] - preff_gI[0:100],color='steelblue',label='gEx - gInh')
                ax.plot(-preff_gI[0:100],color='tomato',label='gInh')
                ax.hlines(0,0,100,ls = '--',color='slategray')
                ax.legend() 
                ax = plt.subplot(grid[3,:])
                ax.plot(null_gE[0:100],color='seagreen',label='gEx')
                ax.plot(null_gE[0:100]-null_gI[0:100],color='steelblue',label='gEx - gInh')
                ax.plot(-null_gI[0:100],color='tomato',label='gInh')
                ax.hlines(0,0,100,ls = '--',color='slategray')
                ax.legend() 
                plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/I1/direct_Polar_%i'%(c),bbox_inches='tight', pad_inches = 0.1)
                plt.close()




    plt.figure(figsize=(14,10))
    plt.subplot(1,2,1)
    plt.hist(dsi_kim_all_E1[0])
    plt.ylabel('number exc. cells')
    plt.xlabel('DSI')
    plt.subplot(1,2,2)
    plt.hist(dsi_kim_all_I1[0])
    plt.ylabel('number inh. cells')
    plt.xlabel('DSI')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/hist_DSI_max_E1',bbox_inches='tight', pad_inches = 0.1)


    preff_mp_E1_oR = np.mean(preff_mp_E1[0],axis=1) 
    null_mp_E1_oR = np.mean(null_mp_E1[0],axis=1) 
    mean_pref_mp_E1_oR = np.reshape(preff_mp_E1_oR,(n_cellsE1,1000,3))
    mean_pref_mp_E1_oR = np.mean(mean_pref_mp_E1_oR,axis=2)
    mean_null_mp_E1_oR = np.reshape(null_mp_E1_oR,(n_cellsE1,1000,3))
    mean_null_mp_E1_oR = np.mean(mean_null_mp_E1_oR,axis=2)
    
    diff_pref_mp_E1 = np.zeros(n_cellsE1)
    diff_null_mp_E1 = np.zeros(n_cellsE1)

    for i in range(n_cellsE1):
        #diff_pref_mp_E1[i] = np.abs(np.max(mean_pref_mp_E1_oR[i]) - np.min(mean_pref_mp_E1_oR[i]))
        #diff_null_mp_E1[i] = np.abs(np.max(mean_null_mp_E1_oR[i]) - np.min(mean_null_mp_E1_oR[i]))
        diff_pref_mp_E1[i] = np.abs(np.max(preff_mp_E1_oR[i]) - np.min(preff_mp_E1_oR[i]))
        diff_null_mp_E1[i] = np.abs(np.max(null_mp_E1_oR[i]) - np.min(null_mp_E1_oR[i]))


    plt.figure()
    plt.scatter(diff_pref_mp_E1,diff_null_mp_E1, c = dsi_kim_all_E1[0])
    plt.plot(np.linspace(100,600,5),np.linspace(100,600,5),'k--')
    plt.xlim(200,550)
    plt.ylim(200,550)
    plt.colorbar()
    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/scatter_min_maxVM_DSI',bbox_inches='tight', pad_inches = 0.1)

    high_dsi = np.where(dsi_kim_all_E1[0] > 0.9)[0]
    low_dsi = np.where(dsi_kim_all_E1[0] < 0.4)[0]

    print(high_dsi)
    print(low_dsi)

    plt.figure()
    plt.scatter(diff_pref_mp_E1[high_dsi],diff_null_mp_E1[high_dsi], color='seagreen', label='high DSI')
    plt.scatter(diff_pref_mp_E1[low_dsi],diff_null_mp_E1[low_dsi], color='tomato', label='low DSI')
    plt.plot(np.linspace(100,600,5),np.linspace(100,600,5),'k--')
    plt.xlim(200,550)
    plt.ylim(200,550)
    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/scatter_min_maxVM_DSI_lowHigh',bbox_inches='tight', pad_inches = 0.1)

    preff_gE_E1_oR = np.mean(preff_gE_E1[0],axis=1) 
    null_gE_E1_oR = np.mean(null_gE_E1[0],axis=1) 

    preff_gI_E1_oR = np.mean(preff_gI_E1[0],axis=1) 
    null_gI_E1_oR = np.mean(null_gI_E1[0],axis=1) 

    print(np.shape(preff_gE_E1_oR))
    plt.figure()
    plt.plot(preff_gE_E1_oR[313,350:750], color='seagreen')
    plt.plot(preff_gE_E1_oR[313,350:750] - preff_gI_E1_oR[313,350:750], color='steelblue')
    plt.plot(-preff_gI_E1_oR[313,350:750], color='tomato')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/currents_313',bbox_inches='tight', pad_inches = 0.1)


    mp_highDSI_preff = preff_mp_E1_oR[high_dsi]
    mp_highDSI_null = null_mp_E1_oR[high_dsi]

    mp_lowDSI_preff = preff_mp_E1_oR[low_dsi]
    mp_lowDSI_null = null_mp_E1_oR[low_dsi]

    gE_highDSI_preff = preff_gE_E1_oR[high_dsi]
    gI_highDSI_preff = preff_gI_E1_oR[high_dsi]

    gE_lowDSI_preff = preff_gE_E1_oR[low_dsi]
    gI_lowDSI_preff = preff_gI_E1_oR[low_dsi]


    gE_highDSI_null = null_gE_E1_oR[high_dsi]
    gI_highDSI_null = null_gI_E1_oR[high_dsi]

    gE_lowDSI_null = null_gE_E1_oR[low_dsi]
    gI_lowDSI_null = null_gI_E1_oR[low_dsi]




    print(np.shape(gE_highDSI_preff))
    plt.figure(figsize=(15,15))
    plt.subplot(2,2,1)
    plt.title('HighDSI preff')
    plt.plot(np.mean(gE_highDSI_preff[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gE_highDSI_preff[:,350:750],axis=0) - np.mean(gI_highDSI_preff[:,350:750], axis=0), color='steelblue', label='gEx - gI')
    plt.plot(-np.mean(gI_highDSI_preff[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(-50,50)
    plt.legend()
    plt.ylabel('Input current')
    plt.subplot(2,2,2)
    plt.title('HighDSI null')
    plt.plot(np.mean(gE_highDSI_null[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gE_highDSI_null[:,350:750],axis=0) - np.mean(gI_highDSI_null[:,350:750], axis=0), color='steelblue', label='gEx - gI')
    plt.plot(-np.mean(gI_highDSI_null[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(-50,50)
    plt.legend()
    plt.ylabel('Input current')
    plt.subplot(2,2,3)
    plt.title('HighDSI preff')
    plt.plot(np.mean(gE_highDSI_preff[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gI_highDSI_preff[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(0,50)
    plt.legend()
    plt.ylabel('Input current')
    plt.subplot(2,2,4)
    plt.title('HighDSI null')
    plt.plot(np.mean(gE_highDSI_null[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gI_highDSI_null[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(0,50)
    plt.legend()
    plt.ylabel('Input current')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/currentsMeanHigh',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(np.mean(gE_highDSI_preff[:,350:750],axis=0), color='darkgreen',ls="solid", label='gEx pref')
    plt.plot(np.mean(gE_highDSI_preff[:,350:750],axis=0) - np.mean(gI_highDSI_preff[:,350:750], axis=0), color='teal',ls="solid", label='gEx - gI pref')
    plt.plot(-np.mean(gI_highDSI_preff[:,350:750], axis=0), color='maroon',ls="solid", label='- gIn pref')

    plt.plot(np.mean(gE_highDSI_null[:,350:750],axis=0), color='mediumseagreen',ls=':', label='gEx null')
    plt.plot(np.mean(gE_highDSI_null[:,350:750],axis=0) - np.mean(gI_highDSI_null[:,350:750], axis=0), color='dodgerblue',ls=':', label='gEx - gI null')
    plt.plot(-np.mean(gI_highDSI_null[:,350:750], axis=0), color='coral',ls=':', label='- gIn null')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(-50,50)
    plt.legend()
    plt.ylabel('Input current')
    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/currentsMeanHigh_both',bbox_inches='tight', pad_inches = 0.1,dpi=300)


    plt.figure(figsize=(15,15))
    plt.subplot(2,2,1)
    plt.title('LowDSI preff')
    plt.plot(np.mean(gE_lowDSI_preff[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gE_lowDSI_preff[:,350:750],axis=0) - np.mean(gI_lowDSI_preff[:,350:750], axis=0), color='steelblue', label='gEx - gI')
    plt.plot(-np.mean(gI_lowDSI_preff[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.legend()
    plt.ylim(-50,50)
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylabel('Input current')
    plt.subplot(2,2,2)
    plt.title('LowDSI null')
    plt.plot(np.mean(gE_lowDSI_null[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gE_lowDSI_null[:,350:750],axis=0) - np.mean(gI_lowDSI_null[:,350:750], axis=0), color='steelblue', label='gEx - gI')
    plt.plot(-np.mean(gI_lowDSI_null[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(-50,50)
    plt.legend()
    plt.ylabel('Input current')

    plt.subplot(2,2,3)
    plt.title('LowDSI preff')
    plt.plot(np.mean(gE_lowDSI_preff[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gI_lowDSI_preff[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.legend()
    plt.ylim(0,50)
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylabel('Input current')
    plt.subplot(2,2,4)
    plt.title('LowDSI null')
    plt.plot(np.mean(gE_lowDSI_null[:,350:750],axis=0), color='seagreen', label='gEx')
    plt.plot(np.mean(gI_lowDSI_null[:,350:750], axis=0), color='tomato', label='- gIn')
    plt.hlines(0,0,400,ls = '--',color='slategray')
    plt.ylim(0,50)
    plt.legend()
    plt.ylabel('Input current')

    plt.savefig('./Output/direct_gratingSinus/maxDSI/maxDSI/currentsMeanLow',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def calcDSI(data, totalT, name,c_lvl,sf_lvl):
    print(data[:,0,0])
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
    plt.hist(dsi_maz,10)
    plt.axvline(np.mean(dsi_maz),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/dsi_maz_hist_'%(c_lvl,sf_lvl)+(name))
    plt.close()

    plt.figure()
    plt.hist(dsi_will,10)
    plt.axvline(np.mean(dsi_will),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/dsi_will_hist_'%(c_lvl,sf_lvl)+(name))
    plt.close()

    plt.figure()
    plt.hist(dsi_kim,10)
    plt.axvline(np.mean(dsi_kim),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/dsi_kim_hist_'%(c_lvl,sf_lvl)+(name))
    plt.close()

def evalDSI(totalT,cont_LVL,sf_LVL):

    #spkC_E2 = np.load('./work/directGrating_Sinus_SpikeCount_E2.npy')
    spkC_I1 = np.load('./work/directGrating_Sinus_SpikeCount_I1.npy') 
    print(np.shape(spkC_I1))
    #spkC_I2 = np.load('./work/directGrating_Sinus_SpikeCount_I2.npy')

    #spkC_E2 = spkC_E2[cont_LVL]
    spkC_I1 = spkC_I1[cont_LVL,sf_LVL]
    #spkC_I2 = spkC_I2[cont_LVL]

    #calcDSI(spkC_E2,totalT,'E2',cont_LVL)
    print(totalT)
    calcDSI(spkC_I1,totalT,'I1',cont_LVL,sf_LVL)

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

    print(np.shape(vm_E1_all))
        

    spkC_LGN_all = np.load('./work/directGrating_Sinus_SpikeCount_LGN.npy')

    print('Mean FR LGN = ',np.mean(spkC_LGN_all))
    print(np.shape(spkC_E1_all))
    n_contrast, n_SF = np.shape(spkC_E1_all)[0:2]


    sim_param = np.loadtxt('./work/directGrating_Sinus_parameter.txt')


    spat_Freq = sim_param[1]   
       

    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]


    w_I1 = np.loadtxt('./Input_network/InhibW.txt')
    n_post,n_pre = np.shape(w_I1)
    rf_I1 = np.reshape(w_I1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_I1 = rf_I1[:,:,:,0] - rf_I1[:,:,:,1]


    for i in range(n_contrast):
        if not os.path.exists('Output/direct_gratingSinus/LVL_%i'%(i)):
            os.mkdir('Output/direct_gratingSinus/LVL_%i'%(i))
        if not os.path.exists('Output/direct_gratingSinus/LVL_%i/E1'%(i)):
            os.mkdir('Output/direct_gratingSinus/LVL_%i/E1'%(i))

        if not os.path.exists('Output/direct_gratingSinus/LVL_%i'%(i)):
            os.mkdir('Output/direct_gratingSinus/LVL_%i'%(i))
        if not os.path.exists('Output/direct_gratingSinus/LVL_%i/I1'%(i)):
            os.mkdir('Output/direct_gratingSinus/LVL_%i/I1'%(i))

        for s in range(n_SF):
            if not os.path.exists('Output/direct_gratingSinus/LVL_%i/SF_%i'%(i,s)):
                os.mkdir('Output/direct_gratingSinus/LVL_%i/SF_%i'%(i,s))
            if not os.path.exists('Output/direct_gratingSinus/LVL_%i/SF_%i/E1'%(i,s)):
                os.mkdir('Output/direct_gratingSinus/LVL_%i/SF_%i/E1'%(i,s))
            if not os.path.exists('Output/direct_gratingSinus/LVL_%i/SF_%i/I1'%(i,s)):
                os.mkdir('Output/direct_gratingSinus/LVL_%i/SF_%i/I1'%(i,s))


    for i in range(n_contrast):
        for s in range(n_SF):
            spkC_E1 = spkC_E1_all[i,s] 
            vm_E1 = vm_E1_all[i,s]
            gE_E1 = gE_E1_all[i,s]
            gI_E1 = gI_E1_all[i,s]

            print(np.mean(spkC_E1))



            spkC_I1 = spkC_I1_all[i,s] 
            vm_I1 = vm_I1_all[i,s]
            gE_I1 = gE_I1_all[i,s]
            gI_I1 = gI_I1_all[i,s]

            spkC_LGN = spkC_LGN_all[i,s]


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


            evalDSI(totalT,i,s)

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


            cycleT = int(1000/(spat_Freq*2))
            n_cycles = int(totalT/cycleT)

            preff_vm_resh = np.reshape(preff_vm,(n_cellsE1,n_cycles,cycleT))
            null_vm_resh = np.reshape(null_vm,(n_cellsE1,n_cycles,cycleT))
            
            preff_vm_I1_resh = np.reshape(preff_vm_I1,(n_cellsI1,n_cycles,cycleT))
            null_vm_I1_resh = np.reshape(null_vm_I1,(n_cellsI1,n_cycles,cycleT))

            plt.figure()
            plt.hist(np.mean(preff_gE,axis=1),24,label='gExc', alpha = 0.5)
            plt.hist(np.mean(preff_gI,axis=1),24,label='gInh', alpha = 0.5)
            plt.legend()
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/gEx_gInh_preff_E1_hist.png'%(i,s))
            plt.close()

            plt.figure()
            plt.hist(np.mean(null_gE,axis=1),24,label='gExc', alpha = 0.5)
            plt.hist(np.mean(null_gI,axis=1),24,label='gInh', alpha = 0.5)
            plt.legend()
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/gEx_gInh_null_E1_hist.png'%(i,s))
            plt.close()


            plt.figure()
            plt.hist(np.mean(preff_gE_I1,axis=1),24,label='gExc', alpha = 0.5)
            plt.hist(np.mean(preff_gI_I1,axis=1),24,label='gInh', alpha = 0.5)
            plt.legend()
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/gEx_gInh_pref_I1_hist.png'%(i,s))

            plt.figure()
            plt.hist(np.mean(null_gE_I1,axis=1),24,label='gExc', alpha = 0.5)
            plt.hist(np.mean(null_gI_I1,axis=1),24,label='gInh', alpha = 0.5)
            plt.legend()
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/gEx_gInh_null_I1_hist.png'%(i,s))




            print(np.shape(preff_vm))

            plt.figure()
            plt.plot(np.mean(preff_vm_resh[100],axis=0),color='steelblue',label='pref')
            plt.plot(np.mean(null_vm_resh[100],axis=0),color='tomato',label='null')
            plt.legend()
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/E1/preff_null_VM_100_resh.png'%(i,s))

            plt.figure()
            plt.plot(preff_vm[100,0:250],color='steelblue',label='pref')
            plt.plot(null_vm[100,0:250],color='tomato',label='null')
            plt.legend()
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/E1/preff_null_VM_100.png'%(i,s))



            plt.figure()
            plt.hist(dsi_maz,10)
            plt.axvline(np.mean(dsi_maz),color='black',linestyle='--',linewidth=4)
            plt.xlim(0,1)
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/dsi_maz_hist_E1'%(i,s))
            plt.close()

            plt.figure()
            plt.hist(dsi_will,10)
            plt.axvline(np.mean(dsi_will),color='black',linestyle='--',linewidth=4)
            plt.xlim(0,1)
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/dsi_will_hist_E1'%(i,s))
            plt.close()

            plt.figure()
            plt.hist(dsi_kim,10)
            plt.axvline(np.mean(dsi_kim),color='black',linestyle='--',linewidth=4)
            plt.xlim(0,1)
            plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/dsi_kim_hist_E1'%(i,s))
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
                plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/E1/direct_Polar_%i'%(i,s,c))
                plt.close()


            alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
            alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
            alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
            alpha = alpha/180. * np.pi #change in radian
                
            for c in range(n_cellsI1):
                spk_cell = mean_overR_spkC_I1[:,c]
                spk_cell = np.append(spk_cell,spk_cell[0])

                grid = plt.GridSpec(2,2,wspace=0.4,hspace=0.3)
                plt.figure()
                ax = plt.subplot(grid[0,0],projection='polar')
                ax.plot( alpha,spk_cell,'o-')
                ax.grid(True)
                ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
                ax = plt.subplot(grid[0,1])
                ax.imshow(rf_I1[c],cmap='gray',aspect='equal',interpolation='none')
                ax.axis('off')
                ax = plt.subplot(grid[1,:])
                ax.plot(np.mean(preff_vm_I1_resh[c],axis=0),color='steelblue',label='pref')
                ax.plot(np.mean(null_vm_I1_resh[c],axis=0),color='tomato',label='null')
                ax.legend() 
                plt.savefig('./Output/direct_gratingSinus/LVL_%i/SF_%i/I1/direct_Polar_%i'%(i,s,c))
                plt.close()
#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def calcDSI(data, duration, name):

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
    plt.savefig('./Output/direct_gratingBar/dsi_maz_hist_'+(name))
    plt.close()

    plt.figure()
    plt.hist(dsi_will,10)
    plt.axvline(np.mean(dsi_will),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingBar/dsi_will_hist_'+(name))
    plt.close()

    plt.figure()
    plt.hist(dsi_kim,10)
    plt.axvline(np.mean(dsi_kim),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingBar/dsi_kim_hist_'+(name))
    plt.close()

def evalDSI(duration):

    spkC_E2 = np.load('./work/directGrating_Bar_SpikeCount_E2.npy')
    spkC_I1 = np.load('./work/directGrating_Bar_SpikeCount_I1.npy') 
    spkC_I2 = np.load('./work/directGrating_Bar_SpikeCount_I2.npy')

    calcDSI(spkC_E2,duration,'E2')

    calcDSI(spkC_I1,duration,'I1')

    calcDSI(spkC_I2,duration,'I2')

    return()

def main():
    print('Analyse direction selectivity by preseting a moving bar')

    spkC_E1 = np.load('./work/directGrating_Bar_SpikeCount_E1.npy')
    vm_E1 = np.load('./work/directGrating_Bar_MemnranPot_E1.npy')

    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]

    n_steps,n_repeats,duration,n_cellsE1 = np.shape(vm_E1)
    step_degree = int(360/n_steps)
    

    print(np.shape(spkC_E1))
    mean_overR_vm_E1 = np.mean(vm_E1,axis=1)
    mean_overR_spkC_E1 = np.mean(spkC_E1,axis=1)



    print('Calculate DSI')
    dsi_maz = np.zeros(n_cellsE1) #DSI after Mazurek et al. (2014) // (R_pref - R_null)/(R_pref)
    dsi_will = np.zeros(n_cellsE1)#DSI after Willson et al. (2018)// (R_pref - R_null)/(R_pref + R_null)
    dsi_kim = np.zeros(n_cellsE1)#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)


    evalDSI(duration)
    preff_vm=np.zeros((n_cellsE1,duration))
    null_vm =np.zeros((n_cellsE1,duration))
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

    plt.figure()
    plt.plot(preff_vm[100],color='steelblue',label='pref')
    plt.plot(null_vm[100],color='tomato',label='null')
    plt.legend()
    plt.savefig('./Output/direct_gratingBar/preff_null_VM_100.png')

    plt.figure()
    plt.hist(dsi_maz,10)
    plt.axvline(np.mean(dsi_maz),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingBar/dsi_maz_hist_E1')
    plt.close()

    plt.figure()
    plt.hist(dsi_will,10)
    plt.axvline(np.mean(dsi_will),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingBar/dsi_will_hist_E1')
    plt.close()

    plt.figure()
    plt.hist(dsi_kim,10)
    plt.axvline(np.mean(dsi_kim),color='black',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.savefig('./Output/direct_gratingBar/dsi_kim_hist_E1')
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
        ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        ax.axis('off')
        ax = plt.subplot(grid[1,:])
        ax.plot(preff_vm[c],color='steelblue',label='pref')
        ax.plot(null_vm[c],color='tomato',label='null')
        ax.legend() 
        plt.savefig('./Output/direct_gratingBar/direct_Polar_%i'%c)
        plt.close()



#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

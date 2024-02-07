import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy import signal

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

    preff_E1 = np.load('./work/dsi_preff_E1.npy')
    print(np.shape(preff_E1))

    dsi_kim = np.load('./work/dsi_kim_Cells.npy')
    print(np.shape(dsi_kim))

    n_cells = n_post
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

    gEx_E1_all = np.load('./work/directGrating_Sinus_gExc_E1.npy')
    gIn_E1_all = np.load('./work/directGrating_Sinus_gInh_E1.npy')
    vm_E1_all = np.load('./work/directGrating_Sinus_MembranPot_E1.npy')


    vm_pref_E1 = []
    vm_null_E1 = []
    gEx_pref_E1 = []
    gEx_null_E1 = []
    gIn_pref_E1 = []
    gIn_null_E1 = []

    

    print(np.shape(vm_E1_all))
    plot_t = 200
    for c in range(n_cells):
        spk_cell = spkC_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,c]
        spk_cell = np.mean(spk_cell,axis=1)
        spk_cell = np.append(spk_cell,spk_cell[0])

        vm_E1_all_cell = vm_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,:,c ]
        vm_E1_all_cell = np.mean(vm_E1_all_cell,axis=1)
        preff_vm_resh= vm_E1_all_cell[int(preff_E1[0,c,2])]
        vm_pref_E1.append(preff_vm_resh)
        total_t = len(preff_vm_resh)
        preff_vm_resh= np.reshape(preff_vm_resh,(int(total_t//plot_t),plot_t))
        #print(np.shape(preff_vm_resh))

        preff_D = int(preff_E1[0,c,2])*step_degree
        null_D = (preff_D+180)%360
        null_arg = int(null_D/step_degree)

        null_vm_resh = vm_E1_all_cell[null_arg]
        vm_null_E1.append(null_vm_resh)
        null_vm_resh = np.reshape(null_vm_resh,(int(total_t//plot_t),plot_t))


        gEx_E1_all_cell = gEx_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,:,c ]
        gEx_E1_all_cell = np.mean(gEx_E1_all_cell,axis=1)
        preff_gEx_resh= gEx_E1_all_cell[int(preff_E1[0,c,2])]
        total_t = len(preff_gEx_resh)
        gEx_pref_E1.append(preff_gEx_resh)
        preff_gEx_resh= np.reshape(preff_gEx_resh,(int(total_t//plot_t),plot_t))
        #print(np.shape(preff_vm_resh))

        null_gEx_resh = gEx_E1_all_cell[null_arg]
        gEx_null_E1.append(null_gEx_resh)
        null_gEx_resh = np.reshape(null_gEx_resh,(int(total_t//plot_t),plot_t))


        gIn_E1_all_cell = gIn_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,:,c ]
        gIn_E1_all_cell = np.mean(gIn_E1_all_cell,axis=1)
        preff_gIn_resh= gIn_E1_all_cell[int(preff_E1[0,c,2])]
        total_t = len(preff_gIn_resh)
        gIn_pref_E1.append(preff_gIn_resh)
        preff_gIn_resh= np.reshape(preff_gIn_resh,(int(total_t//plot_t),plot_t))
        #print(np.shape(preff_vm_resh))

        null_gIn_resh = gIn_E1_all_cell[null_arg]
        gIn_null_E1.append(null_gIn_resh)
        null_gIn_resh = np.reshape(null_gIn_resh,(int(total_t//plot_t),plot_t))


        grid = plt.GridSpec(5,2,wspace=0.4,hspace=0.3)
        plt.figure(figsize=(14,16))
        ax = plt.subplot(grid[0,0],projection='polar')
        ax.plot( alpha,spk_cell,'o-')
        ax.grid(True)
        ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
        ax = plt.subplot(grid[0,1])
        ax.set_title('DSI = %f'%dsi_kim[0,c])
        ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        ax.axis('off')
        ax = plt.subplot(grid[1,:])
        ax.plot(preff_vm_resh[5],color='steelblue',label='pref')#ax.plot(np.mean(preff_vm_resh,axis=0),color='steelblue',label='pref')#
        ax.plot(null_vm_resh[5],color='tomato',label='null')#ax.plot(np.mean(null_vm_resh,axis=0),color='tomato',label='null')#
        ax.legend() 
        ax = plt.subplot(grid[2,:])
        ax.plot(preff_gEx_resh[5],'--',color='steelblue',label='pref gEx')#ax.plot(np.mean(preff_gEx_resh,axis=0),'--',color='steelblue',label='pref gEx')#
        ax.plot(null_gEx_resh[5],'--',color='tomato',label='null gEx')#ax.plot(np.mean(null_gEx_resh,axis=0),'--',color='tomato',label='null gEx')#
        ax.legend() 
        ax = plt.subplot(grid[3,:])
        ax.plot(preff_gIn_resh[5],'.-',color='steelblue',label='pref gIn')#ax.plot(np.mean(preff_gIn_resh,axis=0),'.-',color='steelblue',label='pref gIn')#
        ax.plot(null_gIn_resh[5],'.-',color='tomato',label='null gIn')#ax.plot(np.mean(null_gIn_resh,axis=0),'.-',color='tomato',label='null gIn')#
        ax.legend() 
        ax = plt.subplot(grid[4,:])
        #ax.plot(np.mean(preff_gEx_resh,axis=0) - np.mean(preff_gIn_resh,axis=0) ,color='steelblue',label='pref gEx - gIn')
        ax.plot(preff_gEx_resh[5] - preff_gIn_resh[5] ,color='steelblue',label='pref gEx - gIn')
        #ax.plot(np.mean(null_gEx_resh,axis=0) - np.mean(null_gIn_resh,axis=0) ,color='tomato',label='null gEx - gIn')
        ax.plot(null_gEx_resh[5] - null_gIn_resh[5] ,color='tomato',label='null gEx - gIn')
        ax.hlines(0,0,plot_t,color='black')
        ax.set_ylim(-60,60)
        ax.legend() 
        plt.savefig('./Output/DSI/E1/direct_Polar_%i'%(c))
        plt.close()
        

    gEx_pref_E1 = np.asarray(gEx_pref_E1)
    gEx_null_E1 = np.asarray(gEx_null_E1)
    gIn_pref_E1 = np.asarray(gIn_pref_E1)
    gIn_null_E1 = np.asarray(gIn_null_E1)

    print(np.shape(gEx_pref_E1))

    crossCorr_pref_valid = []
    crossCorr_null_valid = []

    crossCorr_pref_same = []
    crossCorr_null_same = []


    for c in range(n_cells):
        crossCorr = signal.correlate(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),'valid')
        crossCorr_pref_valid.append(crossCorr)

        crossCorr = signal.correlate(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),'valid')
        crossCorr_null_valid.append(crossCorr)


        crossCorr = signal.correlate(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),'same')
        crossCorr = calcCrossCorr(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),50 )
        crossCorr_pref_same.append(crossCorr)#/np.max(crossCorr))

        crossCorr = signal.correlate(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),'same')
        crossCorr = calcCrossCorr(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),50 )
        crossCorr_null_same.append(crossCorr)#/np.max(crossCorr))

    crossCorr_pref_valid = np.asarray(crossCorr_pref_valid)
    crossCorr_null_valid = np.asarray(crossCorr_null_valid)

    crossCorr_pref_same = np.asarray(crossCorr_pref_same)
    crossCorr_null_same = np.asarray(crossCorr_null_same)

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.scatter(crossCorr_pref_valid[:,0], crossCorr_null_valid[:,0],c=dsi_kim[0])
    plt.xlabel('pref')
    plt.ylabel('null')
    plt.subplot(1,3,2)
    plt.scatter(dsi_kim[0], crossCorr_pref_valid[:,0])
    plt.xlabel('DSI')
    plt.ylabel('null')
    plt.subplot(1,3,3)
    plt.scatter(dsi_kim[0], crossCorr_null_valid[:,0])
    plt.xlabel('DSI')
    plt.ylabel('null')
    plt.savefig('./Output/DSI/crossCor_valid')

    t_wi = 51
    t_total = len(gEx_pref_E1[0])
    lags = signal.correlation_lags(t_wi, t_wi)

    print(len(lags))

    plt.figure()
    plt.plot(lags,crossCorr_pref_same[0], label='pref') #[int(t_total/2)-t_wi:int(t_total/2)+t_wi-1]
    plt.plot(lags,crossCorr_null_same[0], label='null') #[int(t_total/2)-t_wi:int(t_total/2)+t_wi-1]
    plt.vlines(0,ymin=0.5,ymax=1.1, linestyles = 'dashed', color='black')
    plt.legend()
    plt.savefig('./Output/DSI/crossCorr')
    plt.close()

    plt.figure()
    plt.plot(np.mean(crossCorr_pref_same,axis=0), label='pref')#[1400:1600]
    plt.plot(np.mean(crossCorr_null_same,axis=0), label='null')#[1400:1600]
    plt.vlines(100,ymin=0.5,ymax=1.1)
    plt.legend()
    plt.savefig('./Output/DSI/crossCorr_MeanALLC')
    plt.close()


    low_dsi_arg = np.where(dsi_kim[0] < 0.2)[0]
    mid_low_dsi_arg = np.where((dsi_kim[0] >= 0.2) & (dsi_kim[0] < 0.4))[0]
    mid_dsi_arg = np.where((dsi_kim[0] >= 0.4) & (dsi_kim[0] < 0.6))[0]
    mid_high_dsi_arg = np.where((dsi_kim[0] >= 0.6) & (dsi_kim[0] < 0.8))[0]
    high_dsi_arg = np.where(dsi_kim[0] >= 0.8)[0]

    print(len(low_dsi_arg), len(mid_dsi_arg),len(high_dsi_arg))

    crossCorr_pref_low = crossCorr_pref_same[low_dsi_arg]
    crossCorr_pref_mid_low = crossCorr_pref_same[mid_low_dsi_arg]
    crossCorr_pref_mid = crossCorr_pref_same[mid_dsi_arg]
    crossCorr_pref_mid_high = crossCorr_pref_same[mid_high_dsi_arg]
    crossCorr_pref_high = crossCorr_pref_same[high_dsi_arg]

    crossCorr_null_low = crossCorr_null_same[low_dsi_arg]
    crossCorr_null_mid_low = crossCorr_null_same[mid_low_dsi_arg]
    crossCorr_null_mid = crossCorr_null_same[mid_dsi_arg]
    crossCorr_null_mid_high = crossCorr_null_same[mid_high_dsi_arg]
    crossCorr_null_high = crossCorr_null_same[high_dsi_arg]


    plt.figure(figsize=(12,4))
    plt.subplot(1,5,1)
    plt.plot(lags,np.mean(crossCorr_pref_low,axis=0), label='pref low') #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_low,axis=0), label='null low') #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.subplot(1,5,2)
    plt.plot(lags,np.mean(crossCorr_pref_mid_low,axis=0), label='pref mid low') #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid_low,axis=0), label='null mid low') #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.subplot(1,5,3)
    plt.plot(lags,np.mean(crossCorr_pref_mid,axis=0), label='pref mid') #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid,axis=0), label='null mid') #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.subplot(1,5,4)
    plt.plot(lags,np.mean(crossCorr_pref_mid_high,axis=0), label='pref mid high') #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid_high,axis=0), label='null mid high') #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.subplot(1,5,5)
    plt.plot(lags,np.mean(crossCorr_pref_high,axis=0), label='pref high') #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_high,axis=0), label='null high') #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.savefig('./Output/DSI/crossCorr_Low_Mid_High')
    plt.close()


    plt.figure(figsize=(25,20))
    for i in range(5):
        plt.subplot(5,5,i+1)
        plt.plot(lags,crossCorr_pref_low[i], label='pref')
        plt.plot(lags,crossCorr_null_low[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('low')
    for i in range(5):
        plt.subplot(5,5,5+i+1)
        plt.plot(lags, crossCorr_pref_mid_low[i], label='pref')
        plt.plot(lags, crossCorr_null_mid_low[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('mid low')
    for i in range(5):
        plt.subplot(5,5,10+i+1)
        plt.plot(lags, crossCorr_pref_mid[i], label='pref')
        plt.plot(lags, crossCorr_null_mid[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('mid')
    for i in range(5):
        plt.subplot(5,5,15+i+1)
        plt.plot(lags, crossCorr_pref_mid_high[i], label='pref')
        plt.plot(lags, crossCorr_null_mid_high[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('mid high')
    for i in range(5):
        plt.subplot(5,5,20+i+1)
        plt.plot(lags, crossCorr_pref_high[i], label='pref')
        plt.plot(lags, crossCorr_null_high[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('high')
    plt.savefig('./Output/DSI/crossCorr_Low_Mid_High_singleCells')
    plt.close()


    """
    plt.figure(figsize=(25,20))
    for i in range(5):
        plt.subplot(5,5,i+1)
        plt.plot(lags,crossCorr_pref_low[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='pref')
        plt.plot(lags,crossCorr_null_low[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='null')
        plt.vlines(0,ymin=0.5,ymax=1.1,linestyles = 'dashed', color='black')
        plt.ylabel('low')
    for i in range(5):
        plt.subplot(5,5,5+i+1)
        plt.plot(lags, crossCorr_pref_mid_low[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='pref')
        plt.plot(lags, crossCorr_null_mid_low[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='null')
        plt.vlines(0,ymin=0.5,ymax=1.1,linestyles = 'dashed', color='black')
        plt.ylabel('mid low')
    for i in range(5):
        plt.subplot(5,5,10+i+1)
        plt.plot(lags, crossCorr_pref_mid[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='pref')
        plt.plot(lags, crossCorr_null_mid[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='null')
        plt.vlines(0,ymin=0.5,ymax=1.1,linestyles = 'dashed', color='black')
        plt.ylabel('mid')
    for i in range(5):
        plt.subplot(5,5,15+i+1)
        plt.plot(lags, crossCorr_pref_mid_high[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='pref')
        plt.plot(lags, crossCorr_null_mid_high[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='null')
        plt.vlines(0,ymin=0.5,ymax=1.1,linestyles = 'dashed', color='black')
        plt.ylabel('mid high')
    for i in range(5):
        plt.subplot(5,5,20+i+1)
        plt.plot(lags, crossCorr_pref_high[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='pref')
        plt.plot(lags, crossCorr_null_high[i][int(t_total/2)-t_wi:int(t_total/2)+t_wi-1], label='null')
        plt.vlines(0,ymin=0.5,ymax=1.1,linestyles = 'dashed', color='black')
        plt.ylabel('high')
    plt.savefig('./Output/DSI/crossCorr_Low_Mid_High_singleCells')
    plt.close()
    """

    sum_gEx_pref = np.mean(gEx_pref_E1,axis=1)
    sum_gEx_null = np.mean(gEx_null_E1,axis=1)

    sum_gIn_pref = np.mean(gIn_pref_E1,axis=1)
    sum_gIn_null = np.mean(gIn_null_E1,axis=1)

    print(np.shape(dsi_kim[0]), np.shape(sum_gEx_pref))

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.scatter(sum_gEx_pref,dsi_kim[0])
    plt.xlabel('Sum gEx pref')
    plt.ylabel('DSI')
    plt.ylim(0,1.1)
    plt.xlim(0,60)
    plt.subplot(2,2,2)
    plt.scatter(sum_gEx_null,dsi_kim[0])
    plt.xlabel('Sum gEx null')
    plt.ylabel('DSI')
    plt.ylim(0,1.1)
    plt.xlim(0,60)
    plt.subplot(2,2,3)
    plt.scatter(sum_gIn_pref,dsi_kim[0])
    plt.xlabel('Sum gIn pref')
    plt.ylabel('DSI')
    plt.ylim(0,1.1)
    plt.xlim(0,60)
    plt.subplot(2,2,4)
    plt.scatter(sum_gIn_null,dsi_kim[0])
    plt.xlabel('Sum gIn null')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI/scatter_currents')
    plt.close()


    plt.figure(figsize=(10,12))
    plt.subplot(3,1,1)
    plt.scatter(np.mean(gEx_pref_E1 - gIn_pref_E1,axis=1), dsi_kim[0])
    plt.ylabel('DSI')
    plt.xlabel('gEx - gIn pref')
    plt.subplot(3,1,2)
    plt.scatter(np.mean(gEx_null_E1 - gIn_null_E1,axis=1), dsi_kim[0])
    plt.ylabel('DSI')
    plt.xlabel('gEx - gIn null')
    plt.subplot(3,1,3)
    plt.scatter(np.mean(gEx_null_E1 - gIn_null_E1,axis=1), np.mean(gEx_pref_E1 - gIn_pref_E1,axis=1),c=dsi_kim[0])
    plt.ylabel('gEx - gIn pref')
    plt.xlabel('gEx - gIn null')
    plt.ylim(-18,0.5)
    plt.xlim(-18,0.5)
    plt.savefig('./Output/DSI/scatter_currents_gEx_gInh')
    plt.close()

#------------------------------------------------------------------------------
if __name__=="__main__":

    main()
    moreDSI()

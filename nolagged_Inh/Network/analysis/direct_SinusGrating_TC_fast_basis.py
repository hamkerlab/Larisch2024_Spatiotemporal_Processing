import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

#import seaborn as sns
#import pandas as pd
import numpy as np
import os
import pickle
from scipy import signal
from tqdm import tqdm

###
# Analyse DS, but use the preff. O from TC to determine preff D
###

def load_Spk_Data():

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)

    ## initialize LGN data
    test_data = np.load('./work/directGrating_Sinus_SpikeCount_LGN_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells = np.shape(test_data)    
    spkC_LGN_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells))

    ## initialize E1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeCount_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells = np.shape(test_data)    
    spkC_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells))

    ## initialize I1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeCount_I1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells = np.shape(test_data)    
    spkC_I1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                spkC_LGN_all[a,sf,tf] = np.load('./work/directGrating_Sinus_SpikeCount_LGN_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                spkC_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeCount_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                spkC_I1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeCount_I1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(spkC_E1_all, spkC_I1_all,spkC_LGN_all)

def load_Time_Data():

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)

    #np.load('./work/directGrating_Sinus_SpikeTimes_E1.npy')
    ## initialize E1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    print(np.shape(test_data))
    n_degree, repeats, n_cells, total_t  = np.shape(test_data)    
    spkT_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells, total_t ))

    ## initialize I1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeTimes_I1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells,total_t  = np.shape(test_data)    
    spkT_I1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells, total_t ))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                spkT_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                spkT_I1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeTimes_I1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(spkT_E1_all, spkT_I1_all)

def load_Current_Data():

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)


    ## initialize current data
    test_data = np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    print(np.shape(test_data))
    n_degree, repeats, total_t, n_cells = np.shape(test_data)    
    gEx_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, total_t, n_cells))
    gIn_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, total_t, n_cells))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                gEx_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                gIn_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_gInh_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(gEx_E1_all,gIn_E1_all)


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


def reject_outliers(data, m=3.):
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def moreDSI():


    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    


    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]

    preff_E1 = np.load('../../work_basis/dsi_preff_E1_TC_0p5_50ms.npy')
    print(np.shape(preff_E1))

    dsi_kim = np.load('../../work_basis/dsi_kim_Cells_TC_0p5_50ms.npy')
    print(np.shape(dsi_kim))


    mse_list = np.load('./work/rmseFFT2D.npy')
    ampl_list = np.load('./work/ampDiffFFT2D.npy')
    print(np.shape(mse_list), np.shape(ampl_list))

    plt.figure()
    plt.scatter(mse_list,dsi_kim[0])
    plt.xlabel('RMSE')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI_TC/RMSE_DSI')
    plt.close()

    plt.figure()
    plt.scatter(ampl_list,dsi_kim[0])
    plt.xlabel('TFD')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI_TC/TFD_DSI')
    plt.close()



    mse_min = np.where(mse_list<0.035)[0]
    dsi_min = dsi_kim[0, mse_min]
    mse_min = mse_list[mse_min]
    mse_max = np.where(mse_list>0.17)[0]
    dsi_max = dsi_kim[0, mse_max]
    mse_max = mse_list[mse_max]

    plt.figure()
    plt.scatter(mse_min,dsi_min, label='min')
    plt.scatter(mse_max,dsi_max, label='max')
    plt.legend()
    plt.xlabel('RMSE')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI_TC/RMSE_DSI_min_max')
    plt.close()
    

    tfd_min = np.where(ampl_list<0.01)[0]
    dsi_min = dsi_kim[0, tfd_min]
    tfd_min = ampl_list[tfd_min]
    tfd_max = np.where(ampl_list>0.3)[0]
    print(tfd_max)
    dsi_max = dsi_kim[0, tfd_max]
    tfd_max = ampl_list[tfd_max]

    plt.figure()
    plt.scatter(tfd_min,dsi_min, label='min')
    plt.scatter(tfd_max,dsi_max, label='max')
    plt.legend()
    plt.xlabel('TFD')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI_TC/TFD_DSI_min_max')
    plt.close()


    #delta_fitt = np.load('./work/1DGabor_Fitt_Delta.npy')
    #print(np.shape(delta_fitt))

 

    #dsi_kim_fitt = dsi_kim[0,0:16]
    #delta_phi = np.max(delta_fitt[:,:,4], axis=1)
    #plt.figure()
    #plt.scatter(delta_phi, dsi_kim_fitt)
    #plt.savefig('./Output/DSI_TC/phi_to_dsi')
    #plt.close()
    #print(np.where(dsi_kim_fitt > 0.7))
    #print(np.where(dsi_kim_fitt < 0.4))

    #plt.figure()
    #plt.subplot(121)
    #plt.plot(dsi_kim_fitt,'o')
    #plt.ylabel('DSI')
    #plt.subplot(122)
    #plt.plot(delta_phi,'o')
    #plt.ylabel('phi')
    #plt.savefig('./Output/DSI_TC/phi_and_dsi')
    #plt.close()


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
    plt.savefig('./Output/DSI_TC/sF_to_speed_imshow')
    plt.close()

    ## load the STRFs
    strf_E1 = np.load('./work/STRF_list.npy')
    print(np.shape(strf_E1))
    dsi_arg = np.argsort(dsi_kim[0][0:25])
    plt.figure(figsize=(12,12))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.title(np.round(dsi_kim[0][dsi_arg[i]],3))
        plt.imshow(strf_E1[dsi_arg[i]],cmap=plt.get_cmap('RdBu',7), interpolation='none', aspect='auto', vmin=-np.max(np.abs(strf_E1[dsi_arg[i]])), vmax=np.max(np.abs(strf_E1[dsi_arg[i]])))
        plt.axis('off')
    plt.savefig('./Output/DSI_TC/STRF')
    plt.close()



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
    plt.savefig('./Output/DSI_TC/hist_PreffD.png')
    plt.close()

    ### compare the pref. orientation with the pref. direction
    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')
    params_E1 = params_E1[:,0,5]
    orienSteps = 8
    preffO_E1 = params_E1*orienSteps

    plt.figure()
    plt.scatter(preffO_E1%180,max_prefD%180)
    plt.xlabel('preffO')
    plt.ylabel('preffD')
    plt.savefig('./Output/DSI_TC/preffO_preffD')
    plt.close()

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
    plt.savefig('./Output/DSI_TC/preffO_minus_preffD')
    plt.close()

    if not os.path.exists('Output/DSI_TC/E1'):
        os.mkdir('Output/DSI_TC/E1')

    # make the direct plots for preff DSI 

    # print spikes per cell and degree for max_DSI
    alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
    alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
    alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
    alpha = alpha/180. * np.pi #change in radian
                        
    #spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')
    spkC_E1_all, _, _= load_Spk_Data()
    
    gEx_E1_all, gIn_E1_all = load_Current_Data()
    #gEx_E1_all = np.load('./work/directGrating_Sinus_gExc_E1.npy')
    #gIn_E1_all = np.load('./work/directGrating_Sinus_gInh_E1.npy')



    gEx_pref_E1 = []
    gEx_null_E1 = []
    gIn_pref_E1 = []
    gIn_null_E1 = []

    

    plot_t = 200
    for c in range(n_cells):
        spk_cell = spkC_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,c]
        spk_cell = np.mean(spk_cell,axis=1)
        spk_cell = np.append(spk_cell,spk_cell[0])


        preff_D = int(preff_E1[0,c,2])*step_degree
        null_D = (preff_D+180)%360
        null_arg = int(null_D/step_degree)

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

        
        #grid = plt.GridSpec(5,2,wspace=0.4,hspace=0.3)
        #plt.figure(figsize=(14,16))
        #ax = plt.subplot(grid[0,0],projection='polar')
        #ax.plot( alpha,spk_cell,'o-')
        #ax.grid(True)
        #ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
        #ax = plt.subplot(grid[0,1])
        #ax.set_title('DSI = %f'%dsi_kim[0,c])
        #ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        #ax.axis('off')
        #ax = plt.subplot(grid[1,:])
        #ax.plot(preff_vm_resh[5],color='steelblue',label='pref')#ax.plot(np.mean(preff_vm_resh,axis=0),color='steelblue',label='pref')#
        #ax.plot(null_vm_resh[5],color='tomato',label='null')#ax.plot(np.mean(null_vm_resh,axis=0),color='tomato',label='null')#
        #ax.legend() 
        #ax = plt.subplot(grid[2,:])
        #ax.plot(preff_gEx_resh[5],'--',color='steelblue',label='pref gEx')#ax.plot(np.mean(preff_gEx_resh,axis=0),'--',color='steelblue',label='pref gEx')#
        #ax.plot(null_gEx_resh[5],'--',color='tomato',label='null gEx')#ax.plot(np.mean(null_gEx_resh,axis=0),'--',color='tomato',label='null gEx')#
        #ax.legend() 
        #ax = plt.subplot(grid[3,:])
        #ax.plot(preff_gIn_resh[5],'.-',color='steelblue',label='pref gIn')#ax.plot(np.mean(preff_gIn_resh,axis=0),'.-',color='steelblue',label='pref gIn')#
        #ax.plot(null_gIn_resh[5],'.-',color='tomato',label='null gIn')#ax.plot(np.mean(null_gIn_resh,axis=0),'.-',color='tomato',label='null gIn')#
        #ax.legend() 
        #ax = plt.subplot(grid[4,:])
        #ax.plot(preff_gEx_resh[5] - preff_gIn_resh[5] ,color='steelblue',label='pref gEx - gIn')
        #ax.plot(null_gEx_resh[5] - null_gIn_resh[5] ,color='tomato',label='null gEx - gIn')
        #ax.hlines(0,0,plot_t,color='black')
        #ax.set_ylim(-60,60)
        #ax.legend() 
        #plt.savefig('./Output/DSI_TC/E1/direct_Polar_%i'%(c))
        #plt.close()
        

    gEx_pref_E1 = np.asarray(gEx_pref_E1)
    gEx_null_E1 = np.asarray(gEx_null_E1)
    gIn_pref_E1 = np.asarray(gIn_pref_E1)
    gIn_null_E1 = np.asarray(gIn_null_E1)

    n_cells, n_steps = np.shape(gEx_pref_E1)

    gEx_pref_E1_smal = np.copy(gEx_pref_E1)
    gEx_null_E1_smal = np.copy(gEx_null_E1)
    gIn_pref_E1_smal = np.copy(gIn_pref_E1)
    gIn_null_E1_smal = np.copy(gIn_null_E1)

    gEx_pref_E1_smal = np.reshape(gEx_pref_E1_smal,(n_cells,30,100))
    gEx_null_E1_smal = np.reshape(gEx_null_E1_smal,(n_cells,30,100))
    gIn_pref_E1_smal = np.reshape(gIn_pref_E1_smal,(n_cells,30,100))
    gIn_null_E1_smal = np.reshape(gIn_null_E1_smal,(n_cells,30,100))

    ### calculate the E/I-ratio for preff and null direction
    ratio_pref = np.nanmean(np.mean(gEx_pref_E1_smal,axis=2)/np.mean(gIn_pref_E1_smal,axis=2),axis=1)
    
    ratio_null = np.nanmean(np.mean(gEx_null_E1_smal,axis=2)/np.mean(gIn_null_E1_smal,axis=2),axis=1) #np.mean(gEx_null_E1,axis=1)/np.mean(gIn_null_E1,axis=1)

    low_dsi_arg = np.where(dsi_kim[0] < 0.2)[0]
    mid_low_dsi_arg = np.where((dsi_kim[0] >= 0.2) & (dsi_kim[0] < 0.4))[0]
    mid_dsi_arg = np.where((dsi_kim[0] >= 0.4) & (dsi_kim[0] < 0.6))[0]
    mid_high_dsi_arg = np.where((dsi_kim[0] >= 0.6) & (dsi_kim[0] < 0.8))[0]
    high_dsi_arg = np.where(dsi_kim[0] >= 0.8)[0]


    plt.figure()
    plt.plot([1,2], [np.mean(ratio_pref[low_dsi_arg]),np.mean(ratio_null[low_dsi_arg])], marker='^', markersize = 8 ,label='0.0-0.2')
    plt.plot([1,2], [np.mean(ratio_pref[mid_low_dsi_arg]),np.mean(ratio_null[mid_low_dsi_arg])], marker='X', markersize = 8,label='0.2-0.4')
    plt.plot([1,2], [np.mean(ratio_pref[mid_dsi_arg]),np.mean(ratio_null[mid_dsi_arg])], marker='*', markersize = 8,label='0.4-0.6')
    plt.plot([1,2], [np.mean(ratio_pref[mid_high_dsi_arg]),np.mean(ratio_null[mid_high_dsi_arg])], marker='s', markersize = 8,label='0.6-0.8')
    plt.plot([1,2], [np.mean(ratio_pref[high_dsi_arg]),np.mean(ratio_null[high_dsi_arg])], marker='o', markersize = 8,label='0.8-1.0')
    plt.xlim(0.75,2.25)
    #plt.ylim(0.7,0.925)
    plt.legend(ncol=2)
    plt.ylabel('E/I ratio')
    plt.xticks([1,2],['Pref','Null'])
    
    plt.savefig('./Output/DSI_TC/EI_ratio_scatter')


    crossCorr_pref_valid = []
    crossCorr_null_valid = []

    crossCorr_pref_same = []
    crossCorr_null_same = []


    for c in range(n_cells):
        crossCorr = signal.correlate(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),'valid')
        crossCorr_pref_valid.append(crossCorr)

        crossCorr = signal.correlate(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),'valid')
        crossCorr_null_valid.append(crossCorr)


        #crossCorr = signal.correlate(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),'same')
        crossCorr = calcCrossCorr(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),20 )
        crossCorr_pref_same.append(crossCorr)#/np.max(crossCorr))

        #crossCorr = signal.correlate(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),'same')
        crossCorr = calcCrossCorr(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),20 )
        crossCorr_null_same.append(crossCorr)#/np.max(crossCorr))

    crossCorr_pref_valid = np.asarray(crossCorr_pref_valid)
    crossCorr_null_valid = np.asarray(crossCorr_null_valid)

    crossCorr_pref_same = np.asarray(crossCorr_pref_same)
    crossCorr_null_same = np.asarray(crossCorr_null_same)

    np.save('./work/dircection_corrCorr_Currents_pref_valid_basis',crossCorr_pref_valid )
    np.save('./work/dircection_corrCorr_Currents_null_valid_basis',crossCorr_null_valid )
    np.save('./work/dircection_corrCorr_Currents_pref_same_basis',crossCorr_pref_same )
    np.save('./work/dircection_corrCorr_Currents_null_same_basis',crossCorr_null_same )

    return -1

    t_wi = 51
    t_total = len(gEx_pref_E1[0])
    lags = signal.correlation_lags(t_wi, t_wi)

    idx_0 = np.where(lags == 0)[0]
    plt.figure()
    plt.hist(crossCorr_pref_same[:,idx_0], histtype='step', label='pref')
    plt.hist(crossCorr_null_same[:,idx_0], histtype='step', label='null')
    plt.legend()
    plt.xlabel('Correlation gEx - gInh')
    plt.ylabel('# Cells')
    plt.savefig('./Output/DSI_TC/hist_Correlation')
    plt.close()

    idx_0 = np.where(lags == 0)[0]
    plt.figure()
    plt.scatter(crossCorr_pref_same[:,idx_0],dsi_kim[0], label='pref')
    plt.scatter(crossCorr_null_same[:,idx_0],dsi_kim[0], label='null')
    plt.legend()
    plt.xlabel('Correlation gEx - gInh')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI_TC/hist_Correlation_scatter')
    plt.close()

    crossCorr_argmax = np.argmax(crossCorr_pref_same,axis=1)    
    crossCorr_pref_maxT =(lags[crossCorr_argmax])

    crossCorr_argmax = np.argmax(crossCorr_null_same,axis=1)    
    crossCorr_nullmaxT = (lags[crossCorr_argmax])

    plt.figure()
    plt.scatter(crossCorr_pref_maxT,dsi_kim[0], label='pref')
    plt.scatter(crossCorr_nullmaxT,dsi_kim[0], label='null')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.ylabel('DSI')
    plt.savefig('./Output/DSI_TC/delta_T_dsi')
    plt.close()



    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.scatter(crossCorr_pref_valid[:,0], crossCorr_null_valid[:,0],c=dsi_kim[0])
    plt.xlabel('pref')
    plt.ylabel('null')
    plt.subplot(1,3,2)
    plt.scatter(dsi_kim[0], crossCorr_pref_valid[:,0])
    plt.xlabel('DSI')
    plt.ylabel('pref')
    plt.subplot(1,3,3)
    plt.scatter(dsi_kim[0], crossCorr_null_valid[:,0])
    plt.xlabel('DSI')
    plt.ylabel('null')
    plt.savefig('./Output/DSI_TC/crossCor_valid')
    plt.close()

    plt.figure()
    plt.plot(lags,crossCorr_pref_same[0], label='pref') #[int(t_total/2)-t_wi:int(t_total/2)+t_wi-1]
    plt.plot(lags,crossCorr_null_same[0], label='null') #[int(t_total/2)-t_wi:int(t_total/2)+t_wi-1]
    plt.vlines(0,ymin=0.5,ymax=1.1, linestyles = 'dashed', color='black')
    plt.legend()
    plt.savefig('./Output/DSI_TC/crossCorr')
    plt.close()

    plt.figure()
    plt.plot(np.mean(crossCorr_pref_same,axis=0), label='pref')#[1400:1600]
    plt.plot(np.mean(crossCorr_null_same,axis=0), label='null')#[1400:1600]
    plt.vlines(100,ymin=0.5,ymax=1.1)
    plt.legend()
    plt.savefig('./Output/DSI_TC/crossCorr_MeanALLC')
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



    ## plot something about the deltaT 
    crossCorr_pref_low_maxT      = reject_outliers(lags[np.argmax(crossCorr_pref_low,axis=1)])
    crossCorr_pref_mid_low_maxT  = reject_outliers(lags[np.argmax(crossCorr_pref_mid_low,axis=1)])
    crossCorr_pref_mid_maxT      = reject_outliers(lags[np.argmax(crossCorr_pref_mid,axis=1)])
    crossCorr_pref_mid_high_maxT = reject_outliers(lags[np.argmax(crossCorr_pref_mid_high,axis=1)])
    crossCorr_pref_high_maxT     = reject_outliers(lags[np.argmax(crossCorr_pref_high,axis=1)])

    crossCorr_null_low_maxT      = reject_outliers(lags[np.argmax(crossCorr_null_low,axis=1)])
    crossCorr_null_mid_low_maxT  = reject_outliers(lags[np.argmax(crossCorr_null_mid_low,axis=1)])
    crossCorr_null_mid_maxT      = reject_outliers(lags[np.argmax(crossCorr_null_mid,axis=1)])
    crossCorr_null_mid_high_maxT = reject_outliers(lags[np.argmax(crossCorr_null_mid_high,axis=1)])
    crossCorr_null_high_maxT     = reject_outliers(lags[np.argmax(crossCorr_null_high,axis=1)])



    data_plot_pref = [crossCorr_pref_low_maxT, crossCorr_pref_mid_low_maxT, crossCorr_pref_mid_maxT, crossCorr_pref_mid_high_maxT, crossCorr_pref_high_maxT]
    data_plot_null = [crossCorr_null_low_maxT, crossCorr_null_mid_low_maxT, crossCorr_null_mid_maxT, crossCorr_null_mid_high_maxT, crossCorr_null_high_maxT]



    data_plot_pref_len = [len(crossCorr_pref_low_maxT), len(crossCorr_pref_mid_low_maxT), len(crossCorr_pref_mid_maxT), len(crossCorr_pref_mid_high_maxT), len(crossCorr_pref_high_maxT)]
    data_plot_null_len = [len(crossCorr_null_low_maxT), len(crossCorr_null_mid_low_maxT), len(crossCorr_null_mid_maxT), len(crossCorr_null_mid_high_maxT), len(crossCorr_null_high_maxT)]


    fig, ax = plt.subplots()
    bp1 = ax.violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),showmeans=False)
    bp2 = ax.violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)),showmeans=False)
    plt.hlines(0,0,11,colors='black',linestyles='dashed')
    #plt.ylim(-15.5,15.5)
    plt.xlabel('DSI')
    plt.ylabel(r'$\Delta$T')
    legend_elements = [Line2D([0], [0], color='steelblue', lw=4, label='pref'), 
                       Line2D([0], [0], color='tomato', lw=4, label='null')]
    ax.legend(handles=legend_elements)
    plt.xticks(np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    plt.savefig('./Output/DSI_TC/violon_deltaT')
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 16))
    bp1 = axes[0].violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),showmeans=False)
    bp2 = axes[0].violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)),showmeans=False)
    axes[0].hlines(0,0,11,colors='black',linestyles='dashed')
    #plt.ylim(-15.5,15.5)
    axes[0].set_xlabel('DSI')
    axes[0].set_ylabel(r'$\Delta$T')
    legend_elements = [Line2D([0], [0], color='steelblue', lw=4, label='pref'), 
                       Line2D([0], [0], color='tomato', lw=4, label='null')]
    axes[0].legend(handles=legend_elements)
    axes[0].set_xticks(np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    bar1 = axes[1].bar([0,1,2,3,4],data_plot_pref_len, alpha=0.4, color='gray')
    plt.savefig('./Output/DSI_TC/violon_deltaT_bar',dpi=300,bbox_inches='tight')
    plt.close()


    ## plot something about the deltaT - do it again, but with abs(deltaT)
    crossCorr_pref_low_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_low,axis=1)]))
    crossCorr_pref_mid_low_maxT  = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_mid_low,axis=1)]))
    crossCorr_pref_mid_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_mid,axis=1)]))
    crossCorr_pref_mid_high_maxT = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_mid_high,axis=1)]))
    crossCorr_pref_high_maxT     = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_high,axis=1)]))

    crossCorr_null_low_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_low,axis=1)]))
    crossCorr_null_mid_low_maxT  = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_mid_low,axis=1)]))
    crossCorr_null_mid_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_mid,axis=1)]))
    crossCorr_null_mid_high_maxT = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_mid_high,axis=1)]))
    crossCorr_null_high_maxT     = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_high,axis=1)]))



    data_plot_pref = [crossCorr_pref_low_maxT, crossCorr_pref_mid_low_maxT, crossCorr_pref_mid_maxT, crossCorr_pref_mid_high_maxT, crossCorr_pref_high_maxT]
    data_plot_null = [crossCorr_null_low_maxT, crossCorr_null_mid_low_maxT, crossCorr_null_mid_maxT, crossCorr_null_mid_high_maxT, crossCorr_null_high_maxT]

    data_plot_pref_len = [len(crossCorr_pref_low_maxT), len(crossCorr_pref_mid_low_maxT), len(crossCorr_pref_mid_maxT), len(crossCorr_pref_mid_high_maxT), len(crossCorr_pref_high_maxT)]
    data_plot_null_len = [len(crossCorr_null_low_maxT), len(crossCorr_null_mid_low_maxT), len(crossCorr_null_mid_maxT), len(crossCorr_null_mid_high_maxT), len(crossCorr_null_high_maxT)]



    fig, ax = plt.subplots()
    bp1 = ax.violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)))
    bp2 = ax.violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)))
    #plt.hlines(0,0,11,colors='black',linestyles='dashed')
    plt.ylim(0)
    plt.xlabel('DSI')
    plt.ylabel(r'$\Delta$T')
    plt.xticks(0.25+np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    plt.savefig('./Output/DSI_TC/violon_deltaT_abs',bbox_inches='tight',dpi=300)
    plt.close()


    fig, ax = plt.subplots()
    bp1 = ax.violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)))
    bp2 = ax.violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)))
    ax.set_ylabel(r'$\Delta$T')
    plt.ylim(0.0, 17.5)
    #plt.hlines(0,0,11,colors='black',linestyles='dashed')
    color = 'gray'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.bar(np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),data_plot_pref_len, alpha=0.3, color=color,width=0.3)
    ax2.bar(0.5+np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),data_plot_null_len, alpha=0.3, color=color,width=0.3)
    ax2.set_ylabel('# cells',color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.ylim(0.0, 17.5)
    plt.xlabel('DSI')
    plt.xticks(0.25+np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    plt.savefig('./Output/DSI_TC/violon_deltaT_abs_bar',bbox_inches='tight',dpi=300)
    plt.close()


    plt.figure(figsize=(16,4))
    plt.subplot(1,5,1)
    plt.title('DSI 0.0 - 0.2')
    plt.plot(lags,np.mean(crossCorr_pref_low,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_low,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.ylabel('cross correlation', fontsize=13)
    plt.xlabel(r'$\Delta$T')
    plt.subplot(1,5,2)
    plt.title('DSI 0.2 - 0.4')
    plt.plot(lags,np.mean(crossCorr_pref_mid_low,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid_low,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.subplot(1,5,3)
    plt.title('DSI 0.4 - 0.6')
    plt.plot(lags,np.mean(crossCorr_pref_mid,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.subplot(1,5,4)
    plt.title('DSI 0.6 - 0.8')
    plt.plot(lags,np.mean(crossCorr_pref_mid_high,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid_high,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.subplot(1,5,5)
    plt.title('DSI 0.8 - 1.0')
    plt.plot(lags,np.mean(crossCorr_pref_high,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_high,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=0.7,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.savefig('./Output/DSI_TC/crossCorr_Low_Mid_High',dpi=300, bbox_inches='tight')
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
        plt.yticks([],[])
    for i in range(5):
        plt.subplot(5,5,10+i+1)
        plt.plot(lags, crossCorr_pref_mid[i], label='pref')
        plt.plot(lags, crossCorr_null_mid[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('mid')
        plt.yticks([],[])
    for i in range(5):
        plt.subplot(5,5,15+i+1)
        plt.plot(lags, crossCorr_pref_mid_high[i], label='pref')
        plt.plot(lags, crossCorr_null_mid_high[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('mid high')
        plt.yticks([],[])
    for i in range(5):
        plt.subplot(5,5,20+i+1)
        plt.plot(lags, crossCorr_pref_high[i], label='pref')
        plt.plot(lags, crossCorr_null_high[i], label='null')
        plt.vlines(0,ymin=-1,ymax=1,linestyles = 'dashed', color='black')
        plt.ylabel('high')
        plt.yticks([],[])
    plt.savefig('./Output/DSI_TC/crossCorr_Low_Mid_High_singleCells')
    plt.close()
    

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
    plt.savefig('./Output/DSI_TC/scatter_currents')
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
    plt.savefig('./Output/DSI_TC/scatter_currents_gEx_gInh')
    plt.close()


def calcTM(rf1,rf2):

    n_pop1 = len(rf1)
    n_pop2 = len(rf2)

    template_match = np.zeros((n_pop1, n_pop2))

    for p1 in range(n_pop1):
        c1 = rf1[p1].flatten()
        for p2 in range(n_pop2):
            c2 = rf2[p2].flatten()
            template_match[p1,p2] = np.dot(c1,c2)/(np.linalg.norm(c1)*np.linalg.norm(c2))

    return(template_match)

def calcPSTH(data,deltaT):
    ## calculate the peristimulus time histogram
    repeats, n_steps = np.shape(data)
    n_bins = int(n_steps//deltaT)
    psth = np.zeros((repeats,n_bins))
    for r in range(repeats):
        spk_times = np.where(data[r] == 1)[0]
        for b in range(n_bins):
            idx = np.where((spk_times>=(0+deltaT*b) ) & (spk_times<(deltaT+deltaT*b) ))[0]
            psth[r,b] = len(idx)
    return(np.mean(psth,axis=0))


def calcDiff(arr_1,arr_2):
    n_1 = len(arr_1)
    n_2 = len(arr_2)
    diff = np.zeros((n_1,n_2))
    for i in range(n_1):
        diff[i,:] = np.abs(arr_1[i] - arr_2)
    return(diff)  


def calcCrossE1(spkC_times_E1, spkC_times_I1, preff_E1, null_idx):


    n_E1 = np.shape(spkC_times_E1)[-2]
    n_I1 = np.shape(spkC_times_I1)[-2]

    max_t_pref_all = np.zeros((n_E1,n_I1))
    max_t_null_all = np.zeros((n_E1,n_I1))
    ### go through all pyr-cells and calculate the cross-correlation between the 
    ## actual pyr-cell and all inhibitory cells for pref and null direction
    
    for cc in tqdm(range(n_E1),ncols=80): #n_E1

        cell = cc

        ### calculate the cross correlation between the actual pyramidial cell and all other inhibitory cells
        ### for null direction
        spk_times_c1 = spkC_times_E1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),int(preff_E1[cell,2]),:,cell,:]
        spk_times_inhib = spkC_times_I1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),int(preff_E1[cell,2]),:,:,:]
        cross_cor_all = []
        correlate_cell_pref = np.zeros(len(spk_times_inhib[0]))
        for c_i in range(len(spk_times_inhib[0])):
            cross_cor = calcCrossCorr(spk_times_c1[0], spk_times_inhib[0,c_i],100)
            cross_cor_all.append(cross_cor)
            idx_st = np.where(spk_times_inhib[0,c_i] == 1)[0]
            correlate_cell_pref[c_i] = np.corrcoef(spk_times_c1[0], spk_times_inhib[0,c_i])[0,1]



        ### calculate the cross correlation between the actual pyramidial cell and all other inhibitory cells
        ### for null direction
        spk_null_c1 = spkC_times_E1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),null_idx[cell],:,cell,:]
        spk_null_inhib = spkC_times_I1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),null_idx[cell],:,:,:]
        cross_cor_all_null = []
        correlate_cell_null = np.zeros(len(spk_times_inhib[0]))
        for c_i in range(len(spk_null_inhib[0])):
            cross_cor = calcCrossCorr(spk_null_c1[0], spk_null_inhib[0,c_i],100)
            cross_cor_all_null.append(cross_cor)
            idx_st = np.where(spk_null_inhib[0,c_i] == 1)[0]
            correlate_cell_null[c_i] = np.corrcoef(spk_null_c1[0], spk_null_inhib[0,c_i])[0,1]

        cross_cor_all = np.asarray(cross_cor_all)
        cross_cor_all_null = np.asarray(cross_cor_all_null)

        ### get the max delta_t 
        t_steps = np.linspace(-99,99,200, dtype='int32')
        
        max_t_pref = np.argmax(cross_cor_all[:,1:-1],axis=1)
        #print(max_t_pref)
        max_t_pref = t_steps[max_t_pref]


        max_t_null = np.argmax(cross_cor_all_null[:,1:-1],axis=1)
        max_t_null = t_steps[max_t_null]


        max_t_pref_all[cc] = max_t_pref
        max_t_null_all[cc] = max_t_null


    np.save('./work/DSI_spike_CrossCorr_pref_short',max_t_pref_all)
    np.save('./work/DSI_spike_CrossCorr_null_short',max_t_null_all)


def calcCycleT(psth_pref_c1, psth_window, total_time):

    fft = np.fft.fft(psth_pref_c1)
    fft_real = fft.real # get the real value part
    fft_img = fft.imag # get the imaginary part
    fft_magn = np.sqrt(fft_real**2 +fft_img**2 ) # calculate the magnitude for each frequency

    fft_phase = np.arctan2(fft_img,fft_real) # calculate the phases in radians!!

    fft_freq = np.fft.fftfreq(len(psth_pref_c1),d= psth_window/total_time )
    #print(fft)
    #print(fft_freq)
    #fft_freq = np.fft.fftshift(fft_freq)

    idx_max = np.argmax(np.abs(fft_magn[1:]))

    max_freq =np.abs(fft_freq[idx_max+1])
    #print(idx_max, fft_freq[idx_max+1])
    ### if Frequency is known -> calculate the time for one cycle -> create corresponding many bins and look, for each bin, when is the peak response (!)
    cycl_len = len(psth_pref_c1)/max_freq
    n_cycles = int(len(psth_pref_c1)/cycl_len)
    #print('cycl_len: ',cycl_len, 'n_cycles: ', n_cycles, 'time: ',int(n_cycles*int(cycl_len)))
    cycles = np.linspace(0,len(psth_pref_c1)-1,n_cycles)
    if (n_cycles*int(cycl_len))%len(psth_pref_c1) == 0: # it fits without a cutoff
        sub_psth = np.reshape(psth_pref_c1, (n_cycles,int(cycl_len)))
        cycl_t = np.argmax(sub_psth,axis=1)
    else:
        cut_off = (n_cycles*int(cycl_len))%len(psth_pref_c1) # time steps that did not fitt into a complete cycle
        sub_psth = np.reshape(psth_pref_c1[0:int(n_cycles*int(cycl_len))], (n_cycles,int(cycl_len)))
        #print(cut_off, np.shape(sub_psth), len(psth_pref_c1[0:int(n_cycles*int(cycl_len))]))
        cycl_t = np.argmax(sub_psth,axis=1)

    #print('cycl_t: ', cycl_t, 'mean_t: ', np.mean(cycl_t), 'median_t: ', np.median(cycl_t))

    
    return(fft_freq, fft_magn, fft_phase, cycl_t, cycl_len)




#-----------------------------------------------------------------------------
if __name__=="__main__":

    moreDSI()


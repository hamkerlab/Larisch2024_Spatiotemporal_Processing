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

def main():
    print('Analyse direction selectivity by preseting a moving Sinus')

    if not os.path.exists('Output/DSI_TC_maxDSI/'):
        os.mkdir('Output/DSI_TC_maxDSI/')


    spkC_E1_all, spkC_I1_all, spkC_LGN_all= load_Spk_Data()


    ### get preff O. 
    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')
    params_E1 = params_E1[:,0,5]
    orienSteps = 8
    preffO_E1 = params_E1*orienSteps

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

        if not os.path.exists('Output/DSI_TC/LVL_%i'%(i)):
            os.mkdir('Output/DSI_TC/LVL_%i'%(i))

        step_degree = int(360/n_degress) 

        ## get through each cell and look, where the cells has spiked most
        for c in range(n_cellsE1):
            spkC_E1 = spkC_E1_all[i,:,:,:,:,c]
            spkC_E1 = np.mean(spkC_E1,axis=3)
            ## get the direction wich is close as possible to pref O + 90°
            
            pref_O = preffO_E1[c]
            ## each pref_0 can be combined with two pref D -> choose the one with higher activity!
            pref_D1 = pref_O
            pref_D2 = (pref_O+180)%360            

            pref_D1_arg = int(pref_D1//step_degree)
            spkC_E1_pref1 = spkC_E1[:,:,pref_D1_arg]

            pref_D2_arg = int(pref_D2//step_degree)
            spkC_E1_pref2 = spkC_E1[:,:,pref_D2_arg]

            pref_speed1, pref_spatF1 = np.where(spkC_E1_pref1 == np.max(spkC_E1_pref1))
            pref_speed2, pref_spatF2 = np.where(spkC_E1_pref2 == np.max(spkC_E1_pref2))

            spk_cell1 = spkC_E1[int(pref_speed1[0]),int(pref_spatF1[0])]
            spk_cell2 = spkC_E1[int(pref_speed2[0]),int(pref_spatF2[0])]
            if np.max(spk_cell1)>np.max(spk_cell2):
                preff_E1[i,c,:] = pref_speed1[0], pref_spatF1[0], pref_D1_arg #pref_degree[0]
            else:
                preff_E1[i,c,:] = pref_speed2[0], pref_spatF2[0], pref_D2_arg


            ##calculate the DSI
            spk_cell = spkC_E1[int(preff_E1[i,c,0]),int(preff_E1[i,c,1])]
            #spk_cell = spk_cell[0]
            preff_arg = int(preff_E1[i,c,2])
            preff_D = preff_arg*step_degree
            null_D = (preff_D+180)%360
            null_arg = int(null_D/step_degree)
            if spk_cell[preff_arg] < spk_cell[null_arg]:
                preff_t = np.copy(null_arg)
                null_arg = np.copy(preff_arg)
                preff_arg = preff_t

            dsi_maz[i, c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg])
            dsi_will[i,c] = (spk_cell[preff_arg] - spk_cell[null_arg])/(spk_cell[preff_arg] + spk_cell[null_arg] )
            dsi_kim[i,c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])            

        print(dsi_maz[i])
        plt.figure()
        plt.hist(dsi_maz[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Mazurek')
        plt.savefig('Output/DSI_TC/LVL_%i/Hist_dsi_maz.png'%(i))
        plt.close()

        plt.figure()
        plt.hist(dsi_will[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Willson')
        plt.savefig('Output/DSI_TC/LVL_%i/Hist_dsi_will.png'%(i))
        plt.close()

        plt.figure()
        plt.hist(dsi_kim[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Kim')
        plt.savefig('Output/DSI_TC/LVL_%i/Hist_dsi_kim.png'%(i))
        plt.close()                

    np.save('./work/dsi_kim_Cells_TC',dsi_kim,allow_pickle=False)
    np.save('./work/dsi_preff_E1_TC',preff_E1,allow_pickle=False)
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

    preff_E1 = np.load('./work/dsi_preff_E1_TC_maxDSI.npy')
    print(np.shape(preff_E1))



    spkC_E1_all, _, _= load_Spk_Data()
    spkC_E1_all = np.squeeze(spkC_E1_all)
    print(preff_E1[:,0,:])
    print( lvl_speed[int(preff_E1[0,0,0])], lvl_spFrq[int(preff_E1[0,0,1])]   )
    print(sim_param)

    print(np.shape(spkC_E1_all))
    print(spkC_E1_all[2,4,:,0,0])

    c = 3
    for i in range(5):
        spatF = i
        spk_cell = np.mean(spkC_E1_all[int(preff_E1[0,c,0]),spatF,:,:,c],axis=1)
        spk_cell = np.append(spk_cell,spk_cell[0])
        step_degree = int(360/n_steps)

        alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
        alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
        alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
        alpha = alpha/180. * np.pi #change in radian

        grid = plt.GridSpec(1,1,wspace=0.4,hspace=0.3)
        plt.figure(figsize=(6,6))
        ax = plt.subplot(grid[0,0],projection='polar')
        ax.plot( alpha,spk_cell,'o-')
        ax.grid(True)
        plt.savefig('./Output/DSI_TC_maxDSI/direct_Polar_spatF_%i'%(spatF))
        plt.close()



    dsi_kim = np.load('./work/dsi_kim_Cells_TC.npy')
    print(np.shape(dsi_kim))

    """
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
    """


    max_prefD = preff_E1[0,:,2]
    step_degree = int(360/n_steps)
    max_prefD = max_prefD*step_degree
    ### the moving direction is orthogonal to the stimulus orientation so add 90 degree 
    max_prefD = (max_prefD+90)%360
    """
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
    """
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
    
    #gEx_E1_all, gIn_E1_all = load_Current_Data()
    #gEx_E1_all = np.load('./work/directGrating_Sinus_gExc_E1.npy')
    #gIn_E1_all = np.load('./work/directGrating_Sinus_gInh_E1.npy')

    n_cells = 324

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


        
        grid = plt.GridSpec(5,2,wspace=0.4,hspace=0.3)
        plt.figure(figsize=(14,16))
        ax = plt.subplot(grid[0,0],projection='polar')
        ax.plot( alpha,spk_cell,'o-')
        ax.grid(True)
        #ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
        ax = plt.subplot(grid[0,1])
        ax.set_title('DSI = %f'%dsi_kim[0,c])
        ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        ax.axis('off')
        plt.savefig('./Output/DSI_TC/E1/direct_Polar_%i'%(c))
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

def plotCurrents():

    if not os.path.exists('Output/DSI_TC/single_currents/'):
        os.mkdir('Output/DSI_TC/single_currents/')

    gEx_E1_all, gIn_E1_all = load_Current_Data()

    gEx_E1_all = np.asarray(np.squeeze(gEx_E1_all))
    gIn_E1_all = np.asarray(np.squeeze(gIn_E1_all))

    n_tF, n_sF, n_degrees, n_rep, t_total, n_cells = np.shape(gEx_E1_all)

    preff_E1 = np.load('../../work_basis/dsi_preff_E1_TC_0p5_50ms.npy')#np.load('./work/dsi_preff_E1_TC.npy')
    preff_E1 = np.squeeze(preff_E1)

    cell_list = [1, 23, 34,54]

    for c in cell_list:
        pref_idx = preff_E1[c]
        pref_tF = int(pref_idx[0])
        pref_sF = int(pref_idx[1])
        print(pref_sF)
        pref_D = int(pref_idx[2])
        null_D = (pref_D+10)%n_degrees 

        gEx_pref = gEx_E1_all[pref_tF,pref_sF,pref_D,:,:,c]
        gEx_pref = np.mean(gEx_pref,axis=0)    

        gIn_pref = gIn_E1_all[pref_tF,pref_sF,pref_D,:,:,c]
        gIn_pref = np.mean(gIn_pref,axis=0)    

        gEx_null = gEx_E1_all[pref_tF,pref_sF,null_D,:,:,c]
        gEx_null = np.mean(gEx_null,axis=0)    

        gIn_null = gIn_E1_all[pref_tF,pref_sF,null_D,:,:,c]
        gIn_null = np.mean(gIn_null,axis=0)   




        offset = 400#t_total - (n_win*t_win)

        n_win = 10#600
        t_win = int((t_total-offset)/n_win)   

        gEx_pref = gEx_pref[offset:]
        gIn_pref = gIn_pref[offset:]
        gEx_null = gEx_null[offset:]
        gIn_null = gIn_null[offset:]
        



        sp = np.fft.fft(gEx_pref)
        freq = np.fft.fftfreq(t_win)
        print(np.argmax(freq))

        gEx_pref = np.reshape(gEx_pref,(t_win, n_win))
        gEx_pref = np.mean(gEx_pref,axis=1)


        gIn_pref = np.reshape(gIn_pref,(t_win, n_win))
        gIn_pref = np.mean(gIn_pref,axis=1)


        gEx_null = np.reshape(gEx_null,(t_win, n_win))
        gEx_null = np.mean(gEx_null,axis=1)    


 
        gIn_null = np.reshape(gIn_null,(t_win, n_win))
        gIn_null = np.mean(gIn_null,axis=1)    

        print(np.mean(gEx_pref), np.mean(gEx_null))
        print(np.mean(gIn_pref), np.mean(gIn_null))
        print('#############')

        plt.figure(figsize=(5,6))
        plt.subplot(311)
        plt.title('Preferred direction')
        plt.plot(gEx_pref[50:150],'--', color='steelblue', label='g_Ex')# [offset:offset+t_win]
        plt.plot(gIn_pref[50:150],'--', color='tomato', label='g_In')
        plt.legend()
        plt.xlabel('Time step [ms]')
        plt.ylabel('Input current [nA]')
        plt.subplot(312)
        plt.title('Null direction')
        plt.plot(gEx_null[50:150],'--', color='steelblue', label='g_Ex')
        plt.plot(gIn_null[50:150],'--', color='tomato', label='g_In')
        plt.legend()
        plt.xlabel('Time step [ms]')
        plt.ylabel('Input current [nA]')
        plt.subplot(313)
        plt.plot(gEx_pref[50:150] - gIn_pref[50:150],'--', color='teal', label='pref')
        plt.plot(gEx_null[50:150] - gIn_null[50:150],'--', color='orange', label='null')
        plt.hlines(0,0,100,colors='black', linestyles='--')
        plt.legend()
        plt.xlabel('Time step [ms]')
        plt.ylabel('gExc - gInh')
        plt.savefig('./Output/DSI_TC/single_currents/currents_c_%i.png'%c,dpi=300,bbox_inches='tight')
        plt.close()



    mean_gEx_pref = np.zeros(n_cells)
    mean_gIn_pref = np.zeros(n_cells)
    mean_gEx_null = np.zeros(n_cells)
    mean_gIn_null = np.zeros(n_cells)

    for c in range(n_cells):
        pref_idx = preff_E1[c]
        pref_tF = int(pref_idx[0])
        pref_sF = int(pref_idx[1])

        pref_D = int(pref_idx[2])
        null_D = (pref_D+10)%n_degrees 


        gEx_pref = gEx_E1_all[pref_tF,pref_sF,pref_D,:,:,c]
        mean_gEx_pref[c] = np.mean(gEx_pref)    

        gIn_pref = gIn_E1_all[pref_tF,pref_sF,pref_D,:,:,c]
        mean_gIn_pref[c] = np.mean(gIn_pref)    

        gEx_null = gEx_E1_all[pref_tF,pref_sF,null_D,:,:,c]
        mean_gEx_null[c] = np.mean(gEx_null)    

        gIn_null = gIn_E1_all[pref_tF,pref_sF,null_D,:,:,c]
        mean_gIn_null[c] = np.mean(gIn_null)   


    print(np.sum(mean_gEx_pref), np.sum(mean_gIn_pref))
    print(np.sum(mean_gEx_null), np.sum(mean_gIn_null))

#-----------------------------------------------------------------------------
if __name__=="__main__":

    #main()
    #moreDSI()
    plotCurrents()

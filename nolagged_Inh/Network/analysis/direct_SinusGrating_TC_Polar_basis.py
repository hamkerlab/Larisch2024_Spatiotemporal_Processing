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

def moreDSI(preff_E1, path):

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    

    spkC_E1_all, _, _= load_Spk_Data()
    spkC_E1_all = np.squeeze(spkC_E1_all)

    n_speeds, n_spatF,n_degress,repeats,n_cellsE1 = np.shape(spkC_E1_all)

    ### get preff orientation via tuning curves 
    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')


    params_E1 = params_E1[:,0,5] # what are the other two parameters ? And why not take them? [not the same range!!]
    orienSteps = 8
    preffO_E1 = params_E1*orienSteps


    ## calculate the DSI again, based on the pref - configuration from the basis model
    dsi_maz = np.zeros(n_cellsE1) #DSI after Mazurek et al. (2014) // (R_pref - R_null)/(R_pref)
    dsi_will = np.zeros(n_cellsE1)#DSI after Willson et al. (2018)// (R_pref - R_null)/(R_pref + R_null)
    dsi_kim = np.zeros(n_cellsE1)#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)    

    step_degree = int(360/n_degress) 
    for c in range(n_cellsE1):
        pref_TF = int(preff_E1[c,0])
        pref_SF = int(preff_E1[c,1])
        spk_cell = spkC_E1_all[pref_TF,pref_SF,:,:,c]
        spk_cell = np.mean(spk_cell,axis=1)

        pref_O = preffO_E1[c]        
        ## each pref_0 can be combined with two pref D -> choose the one with higher activity!
        pref_D1 = pref_O
        pref_D2 = (pref_O+180)%360            

        pref_D1_arg = int(pref_D1//step_degree)
        pref_D2_arg = int(pref_D2//step_degree)



        ## decide what is the preferred direction and what the null direction via spike count
        if spk_cell[pref_D1_arg] > spk_cell[pref_D2_arg]:
            pref_arg = pref_D1_arg
            null_arg = pref_D2_arg
        else:
            pref_arg = pref_D2_arg
            null_arg = pref_D1_arg

        #pref_arg = np.argmax(spk_cell)
        #null_arg = (pref_arg+10)%20


        #print(pref_arg, null_arg)

        dsi_maz[ c] = (spk_cell[pref_arg] - spk_cell[null_arg])/(spk_cell[pref_arg])
        dsi_will[c] = (spk_cell[pref_arg] - spk_cell[null_arg])/(spk_cell[pref_arg] + spk_cell[null_arg] )
        dsi_kim[c] = 1 - (spk_cell[null_arg]/spk_cell[pref_arg])   


    np.save('./work/dsi_TC_basis_'+path+'_maz',dsi_maz)
    np.save('./work/dsi_TC_basis_'+path+'_will',dsi_will)
    np.save('./work/dsi_TC_basis_'+path+'_kim',dsi_kim)

    print(len(np.where(dsi_kim>=0.8)[0]))

    plt.figure()
    plt.hist(dsi_maz)
    plt.ylabel('# of cells')
    plt.xlabel('DSI Mazurek')
    plt.savefig('Output/DSI_TC_basis/'+path+'/DSI_hist/Hist_dsi_maz.png')
    plt.close()

    plt.figure()
    plt.hist(dsi_will)
    plt.ylabel('# of cells')
    plt.xlabel('DSI Willson')
    plt.savefig('Output/DSI_TC_basis/'+path+'/DSI_hist/Hist_dsi_will.png')
    plt.close()

    plt.figure()
    plt.hist(dsi_kim)
    plt.ylabel('# of cells')
    plt.xlabel('DSI Kim')
    plt.savefig('Output/DSI_TC_basis/'+path+'/DSI_hist/Hist_dsi_kim.png')
    plt.close()                


    ### make some polar plots
    c = 1
    for i in range(5):
        spatF = i
        spk_cell = np.mean(spkC_E1_all[4,spatF,:,:,c],axis=1)
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
        plt.savefig('./Output/DSI_TC_basis/'+path+'/direct_Polar_spatF_%i'%(spatF))
        plt.close()




def DSI_STRF():

    if not os.path.exists('Output/DSI_TC_basis'):
        os.mkdir('Output/DSI_TC_basis')

    if not os.path.exists('Output/DSI_TC_basis/STRF'):
        os.mkdir('Output/DSI_TC_basis/STRF')

    if not os.path.exists('Output/DSI_TC_basis/STRF/DSI_hist'):
        os.mkdir('Output/DSI_TC_basis/STRF/DSI_hist')

    preff_E1 = np.load('../../work_basis/dsi_preff_E1_TC_0p5_50ms.npy')
    preff_E1 = np.squeeze(preff_E1)
    print(np.shape(preff_E1))

    moreDSI(preff_E1, 'STRF')


def DSI_noSTRF():


    if not os.path.exists('Output/DSI_TC_basis'):
        os.mkdir('Output/DSI_TC_basis')

    if not os.path.exists('Output/DSI_TC_basis/noSTRF'):
        os.mkdir('Output/DSI_TC_basis/noSTRF')

    if not os.path.exists('Output/DSI_TC_basis/noSTRF/DSI_hist'):
        os.mkdir('Output/DSI_TC_basis/noSTRF/DSI_hist')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    

    preff_E1 = np.load('../../work_basis/dsi_preff_E1_TC_0p5.npy')
    preff_E1 = np.squeeze(preff_E1)
    print(np.shape(preff_E1))

    moreDSI(preff_E1, 'noSTRF')


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

    DSI_STRF()
    DSI_noSTRF()


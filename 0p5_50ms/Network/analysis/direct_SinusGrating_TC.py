import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#import seaborn as sns
#import pandas as pd
import numpy as np
import os
import pickle
from scipy import signal
from tqdm import tqdm
from load_data import *

###
# Analyse DS, but use the preff. O from TC to determine preff D
###


def main():
    print('Analyse direction selectivity by preseting a moving Sinus')

    if not os.path.exists('Output/DSI_TC/'):
        os.mkdir('Output/DSI_TC/')


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

            dsi_kim[i,c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])            

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
    for i in range(t_win,0,-1): # first shifts
        y_calc = y[i:]
        x_calc = x[:len(y_calc)]
        crossCorr[c] = np.corrcoef(x_calc,y_calc)[0,1]
        c+=1

    ## identic ##
    crossCorr[c] = np.corrcoef(x,y)[0,1]
    c+=1

    ## for the next shifts
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
    
    print('Calculate CrossCorrelation between spikes')
    print('####################')

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

#-----------------------------------------------------------------------------
if __name__=="__main__":

    main()

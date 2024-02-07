import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
import os
import pickle
from scipy import signal

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


def main():
    print('Analyse direction selectivity by preseting a moving Sinus')

    if not os.path.exists('Output/DSI_TC/'):
        os.mkdir('Output/DSI_TC/')

    spkC_E1_all, spkC_I1_all,spkC_LGN_all = load_Spk_Data()

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

        plt.figure()
        plt.hist(dsi_will[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Willson')
        plt.savefig('Output/DSI_TC/LVL_%i/Hist_dsi_will.png'%(i))

        plt.figure()
        plt.hist(dsi_kim[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Kim')
        plt.savefig('Output/DSI_TC/LVL_%i/Hist_dsi_kim.png'%(i))
                    

    np.save('./work/dsi_kim_Cells_TC',dsi_kim,allow_pickle=False)
    np.save('./work/dsi_preff_E1_TC',preff_E1,allow_pickle=False)

#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

import numpy as np
import pickle

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


    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)

    #np.load('./work/directGrating_Sinus_SpikeTimes_E1.npy')
    ## initialize E1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 

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


    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)


    ## initialize current data
    test_data = np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 

    n_degree, repeats, total_t, n_cells = np.shape(test_data)    
    gEx_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, total_t, n_cells))
    gIn_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, total_t, n_cells))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                gEx_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                gIn_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_gInh_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(gEx_E1_all,gIn_E1_all)


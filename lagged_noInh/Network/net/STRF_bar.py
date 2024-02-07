from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import os.path
from scipy.special import factorial
from net_EB_LGN_DoG import *


def setInput(filter_size):

    patch = np.ones((s_input,s_input))# create a "neutral" patch with the size patchsize x patchsize -> make it bigger (+boarder) depending on size of the filter
    p_h,p_w = np.shape(patch)
    # show a dark or bright bar
    bar_w = np.random.randint(1,8)
    bar_h = np.random.randint(1,8)
    x_pos = np.random.randint(0,s_input-bar_w) #np.random.randint(boarder,patchsize-bar_w)
    y_pos = np.random.randint(0,s_input-bar_h)#np.random.randint(boarder,patchsize-bar_h)
    value = np.random.rand()

    if np.random.rand()<=0.5:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 2.0  # bright bar
    else:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 0.0  # dark bar

    popIMG.r = patch


    return(patch)

def start():
    print('Showing bright and dark bars with a DoG input layer')

    duration = 50 #ms
    n_stim = 400000
    compile()
    loadWeights()

    print(popRGC.geometry)
    print(popLGN.geometry)
    # use population views for monitoring to reduce the memory usage
    pop_viewRGC = popRGC[0:16,0,:16]
    pop_viewLGN = popLGN[0:16,0,:16]
    pop_viewE1 = popE1#[0:16]
    pop_viewI1 = popIL1[0:16]
    pop_viewE2 = popE2[0:16]
    pop_viewI2 = popIL2[0:16]
    


    rgc_mon = Monitor(pop_viewRGC,['spike']) #Monitor(popRGC,['spike'])
    lgn_mon = Monitor(pop_viewLGN,['spike']) #Monitor(popLGN,['spike'])#,'current','v'])
    mon_E1 = Monitor(pop_viewE1,['spike']) #Monitor(popE1,['spike'])
    mon_I1 = Monitor(pop_viewI1,['spike']) #Monitor(popIL1,['spike'])

    mon_E2 = Monitor(pop_viewE2,['spike']) #Monitor(popE2,['spike'])
    mon_I2 = Monitor(pop_viewI2,['spike']) #Monitor(popIL2,['spike'])

    rec_RGC_spikes = []
    rec_LGN_spikes = []
    rec_E1_spikes = [] 
    rec_I1_spikes = []
    rec_E2_spikes = [] 
    rec_I2_spikes = []

    input_list = []
    
    ### lagged LGN cells over syn-Delays
    # add some extra delays to implement "lagged" LGN-Cells -> additional delay depends on t_delay !
    projInput_LGN_ON.delay = np.load('./work/LGN_ON_delay.npy')
    projInput_LGN_OFF.delay = np.load('./work/LGN_OFF_delay.npy')


    print('Start simulation')
    
    for i in range(n_stim):


        patch = setInput(filter_size)
        input_list.append(patch)

        simulate(duration)


        spikes_RGC = rgc_mon.get('spike')
        spikes_LGN = lgn_mon.get('spike')
        spikes_E1 = mon_E1.get('spike')
        spikes_I1 = mon_I1.get('spike')
        spikes_E2 = mon_E2.get('spike')
        spikes_I2 = mon_I2.get('spike')


        rec_RGC_spikes.append(spikes_RGC)
        rec_LGN_spikes.append(spikes_LGN)
        rec_E1_spikes.append(spikes_E1)
        rec_I1_spikes.append(spikes_I1)
        rec_E2_spikes.append(spikes_E2)
        rec_I2_spikes.append(spikes_I2)

        if((i%(n_stim/10)) == 0):
            print("Round %i of %i" %(i,n_stim))

    print('Simulation is finished')

    np.save('./work/STRF_Input',input_list)

    np.save('./work/STRF_Spk_RGC',rec_RGC_spikes, allow_pickle=True)
    np.save('./work/STRF_Spk_LGN',rec_LGN_spikes, allow_pickle=True)
    np.save('./work/STRF_Spk_E1',rec_E1_spikes, allow_pickle=True)
    np.save('./work/STRF_Spk_I1',rec_I1_spikes, allow_pickle=True)
    np.save('./work/STRF_Spk_E2',rec_E2_spikes, allow_pickle=True)
    np.save('./work/STRF_Spk_I2',rec_I2_spikes, allow_pickle=True)

if __name__=="__main__":
    start()

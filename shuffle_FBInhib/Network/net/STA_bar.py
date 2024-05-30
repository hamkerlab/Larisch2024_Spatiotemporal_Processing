from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import os.path
from scipy.special import factorial
from net_EB_LGN_DoG import *


def setInput(filter_size):

    patch = np.zeros((s_input,s_input))# create a "neutral" patch with the size patchsize x patchsize -> make it bigger (+boarder) depending on size of the filter
    p_h,p_w = np.shape(patch)
    # show a dark or bright bar
    bar_w = np.random.randint(1,8)
    bar_h = np.random.randint(1,8)
    x_pos = np.random.randint(0,s_input-bar_w) #np.random.randint(boarder,patchsize-bar_w)
    y_pos = np.random.randint(0,s_input-bar_h)#np.random.randint(boarder,patchsize-bar_h)
    value = np.random.rand()

    if np.random.rand()<=0.5:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 0.5  # bright bar
    else:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = -0.5  # dark bar

    popIMG.r = patch


    return(patch)

def start():
    print('Showing bright and dark bars with a DoG input layer')

    duration = 100 #ms
    n_stim = 100000
    compile()
    loadWeights()


    pop_viewRGC = popRGC[0:9,0,:9]
    pop_viewLGN = popLGN[0:9,0,:9]
    pop_viewE1 = popE1[0:9]
    pop_viewI1 = popIL1[0:9]

    rgc_mon = Monitor(popRGC,['spike'])
    lgn_mon = Monitor(popLGN,['spike'])#,'current','v'])
    mon_E1 = Monitor(popE1,['spike'])
    mon_I1 = Monitor(popIL1,['spike'])

    mon_E2 = Monitor(popE2,['spike'])
    mon_I2 = Monitor(popIL2,['spike'])

    rec_RGC_count = np.zeros((n_stim, popRGC.size))
    rec_LGN_count = np.zeros((n_stim, popLGN.size))
    rec_E1_count = np.zeros((n_stim, popE1.size)) 
    rec_I1_count = np.zeros((n_stim, popIL1.size))
    rec_E2_count = np.zeros((n_stim, popE2.size)) 
    rec_I2_count = np.zeros((n_stim, popIL2.size))

    maxN = np.max([popRGC.size, popLGN.size, popE1.size])

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

        for n in range(maxN):
            if n < popRGC.size:
                rec_RGC_count[i,n] = len(spikes_RGC[n])*1000/duration
            if n < popLGN.size:
                rec_LGN_count[i,n] = len(spikes_LGN[n])*1000/duration
            if n < popE1.size:
                rec_E1_count[i,n] = len(spikes_E1[n])*1000/duration
            if n < popIL1.size:
                rec_I1_count[i,n] = len(spikes_I1[n])*1000/duration
            if n < popE2.size:
                rec_E2_count[i,n] = len(spikes_E2[n])*1000/duration
            if n < popIL2.size:
                rec_I2_count[i,n] = len(spikes_I2[n])*1000/duration

        if((i%(n_stim/10)) == 0):
            print("Round %i of %i" %(i,n_stim))

    print('Simulation is finished')

    np.save('./work/STRF_Count_Input',input_list)

    np.save('./work/STRF_SpkCount_RGC',rec_RGC_count, allow_pickle=True)
    np.save('./work/STRF_SpkCount_LGN',rec_LGN_count, allow_pickle=True)
    np.save('./work/STRF_SpkCount_E1',rec_E1_count, allow_pickle=True)
    np.save('./work/STRF_SpkCount_I1',rec_I1_count, allow_pickle=True)
    np.save('./work/STRF_SpkCount_E2',rec_E2_count, allow_pickle=True)
    np.save('./work/STRF_SpkCount_I2',rec_I2_count, allow_pickle=True)

if __name__=="__main__":
    start()

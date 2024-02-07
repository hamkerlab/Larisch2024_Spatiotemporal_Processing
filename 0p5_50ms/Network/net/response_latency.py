from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import os.path
from scipy.special import factorial
from net_EB_LGN_DoG import *


def setInput(filter_size):


    patch = np.ones((s_input,s_input))*0.5 # create a "neutral" patch with the size patchsize x patchsize -> make it bigger depending on size of the filter
    p_h,p_w = np.shape(patch)
    # show a dark or bright bar
    bar_w = np.random.randint(2,9)
    bar_h = np.random.randint(2,9)
    x_pos = np.random.randint(0,p_w-bar_w) #np.random.randint(boarder,patchsize-bar_w)
    y_pos = np.random.randint(0,p_h-bar_h)#np.random.randint(boarder,patchsize-bar_h)
    value = np.random.rand()

    if np.random.rand()<=0.5:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 1  # bright bar
    else:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 0  # dark bar

    """
    patch = np.ones((s_input,s_input))*0.5
    patch[5:int((s_input)/2),:] = 1


    plt.figure()
    plt.imshow(patch,cmap='gray',vmin=0.0,vmax=1.0)
    plt.colorbar()
    plt.savefig('input_'+str((x_pos,y_pos))+'.jpg')    
    """

    popIMG.r = patch


    return(patch)

def start():
    print('Showing bright and dark bars with a DoG input layer')

    duration = 50 #ms
    n_stim =100# 30000
    compile()

    loadWeights()

    ### lagged LGN cells over syn-Delays
    # add some extra delays to implement "lagged" LGN-Cells -> additional delay depends on t_delay !
    projInput_LGN_ON.delay = np.load('./work/LGN_ON_delay.npy')
    projInput_LGN_OFF.delay = np.load('./work/LGN_OFF_delay.npy')

    rgc_mon = Monitor(popRGC,['spike','v'])
    lgn_mon = Monitor(popLGN,['spike','v'])#,'current','v'])
    mon_E1 = Monitor(popE1,['spike','g_Exc'])#,'vm','g_Exc','g_Inh'])
    mon_IL1 = Monitor(popIL1,['spike'])#,'vm','g_Exc','g_Inh'])
    #mon_E2 = Monitor(popE2,['spike'])#,'vm','g_Exc','g_Inh'])
    #mon_IL2 = Monitor(popIL2,['spike'])#,'vm','g_Exc','g_Inh'])


    rec_RGC_spikes = []
    rec_LGN_spikes = []
    rec_E1_spikes = []#np.zeros((numberOfNeurons,nbrOfPatchesSTA))
    rec_IL1_spikes = []    
    #rec_E2_spikes = []
    #rec_IL2_spikes = []

#    rec_LGN_current = []
    rec_RGC_mV = []
    rec_LGN_mV = []
 
    blank_patch = patch = np.ones((s_input,s_input))*0.5

    print('Start simulation')
    
    for i in range(n_stim):


        patch = setInput(filter_size)


        simulate(duration)
        popIMG.r = blank_patch
        simulate(200) # additional time to let the neurons "cool down"

        spikes_RGC = rgc_mon.get('spike')
        spikes_LGN = lgn_mon.get('spike')
        spikes_E1 = mon_E1.get('spike')
        spikes_IL1 = mon_IL1.get('spike')
        #spikes_E2 = mon_E2.get('spike')
        #spikes_IL2 = mon_IL2.get('spike')    

        rec_RGC_spikes.append(spikes_RGC)
        rec_LGN_spikes.append(spikes_LGN)
        rec_E1_spikes.append(spikes_E1)
        rec_IL1_spikes.append(spikes_IL1)
        #rec_E2_spikes.append(spikes_E2)
        #rec_IL2_spikes.append(spikes_IL2)


#        current_LGN = lgn_mon.get('current')
        mV_LGN = lgn_mon.get('v')
        mV_RGC = rgc_mon.get('v')
        
#        rec_LGN_current.append(current_LGN)
        rec_LGN_mV.append(mV_LGN)


        rec_RGC_mV.append(mV_RGC)

        if((i%(n_stim/10)) == 0):
            print("Round %i of %i" %(i,n_stim))

    print('Simulation is finished')

    np.save('./work/responceLat_RGC',rec_RGC_spikes)
    np.save('./work/responceLat_LGN',rec_LGN_spikes)
    np.save('./work/responceLat_Spk_E1',rec_E1_spikes)
    np.save('./work/responceLat_Spk_IL1',rec_IL1_spikes)
    #np.save('./work/responceLat_Spk_E2',rec_E2_spikes)
    #np.save('./work/responceLat_Spk_IL2',rec_IL2_spikes)

    exc_E1 = mon_E1.get('g_Exc')
    np.save('./work/responceLat_gExc_E1',exc_E1)

    np.save('./work/responceLat_v_RGC',rec_RGC_mV)
    

    #np.save('./work/STRF_membrPotExc',rec_membPotEx)
    #np.save('./work/STRF_gExc_Ex',rec_gExcEx)
    #np.save('./work/STRF_gInh_Ex',rec_gInhEx)


    print(np.shape(rec_LGN_mV))

    mv_LGN = np.reshape(rec_LGN_mV,(n_stim*(duration+200),n_LGN))

    plt.figure()
    plt.plot(mv_LGN[:,0])
    plt.plot(mv_LGN[:,1])
    plt.plot(mv_LGN[:,2])
    plt.plot(mv_LGN[:,3])
    plt.savefig('Output/vm_LGN.png')


    print(rec_LGN_spikes)

if __name__=="__main__":
    start()

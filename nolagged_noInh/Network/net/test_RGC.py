from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import os.path
from net_EB_LGN_DoG import *
def setInput(filter_size):

    boarder = 1
    patch = np.ones((s_input,s_input))*0.5 # create a "neutral" patch with the size patchsize x patchsize -> make it bigger (+boarder) depending on size of the filter
    p_h,p_w = np.shape(patch)
    # show a dark or bright bar
    bar_w = 1#np.random.randint(2,9)
    bar_h = 6#np.random.randint(2,9)
    x_pos = 1#np.random.randint(0,p_w-bar_w) #np.random.randint(boarder,patchsize-bar_w)
    y_pos = 0#np.random.randint(0,p_h-bar_h)#np.random.randint(boarder,patchsize-bar_h)
    

    if np.random.rand()<=0.5:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 1  # bright bar
    else:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 0  # dark bar

    popIMG.r = patch


    return(patch)


def start():

    duration = 100 #ms
    n_stim =2# 30000
    compile()

    loadWeights()

    rgc_mon = Monitor(popInput,['spike','v'])
    lgn_mon = Monitor(popLGN,['spike'])
    e1_mon = Monitor(popE1,['spike'])


    blank_patch = patch = np.ones((s_input,s_input))*0.5

    for i in range(n_stim):
        patch = setInput(filter_size)

        simulate(duration)
        popIMG.r = blank_patch
        simulate(200) # additional time to let the neurons "cool down"

    spikes_RGC = rgc_mon.get('spike')
    v_rgc = rgc_mon.get('v')

    plt.figure()
    plt.plot(v_rgc[:,0])
    plt.savefig('./Output/test_v.png')


    spk_lgn=lgn_mon.get('spike')
    

    spk_E1 = e1_mon.get('spike')

    print(spikes_RGC)
if __name__=="__main__":
    start()


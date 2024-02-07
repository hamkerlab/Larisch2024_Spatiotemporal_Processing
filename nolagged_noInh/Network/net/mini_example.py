from ANNarchy import *
from net_mini import *
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def setInput(filter_size):

    boarder = 1
    patch = np.ones((s_input,s_input))*0.5 # create a "neutral" patch with the size patchsize x patchsize -> make it bigger (+boarder) depending on size of the filter
    p_h,p_w = np.shape(patch)
    # show a dark or bright bar
    bar_w = np.random.randint(1,4)
    bar_h = np.random.randint(1,4)
    x_pos = np.random.randint(0,p_w-bar_w) #np.random.randint(boarder,patchsize-bar_w)
    y_pos = np.random.randint(0,p_h-bar_h)#np.random.randint(boarder,patchsize-bar_h)
    value = np.random.rand()

    """
    if np.random.rand()<=0.5:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 1  # bright bar
    else:
        patch[x_pos:x_pos+bar_w,y_pos:y_pos+bar_h] = 0  # dark bar
    """
    plt.figure()
    plt.imshow(patch,cmap='gray',vmin=0,vmax=1)
    plt.colorbar()
    plt.savefig('./Output/input')

    popIMG.r = patch

    
    return(patch)

def start():
    print('Showing bright and dark bars with a DoG input layer')

    duration = 100 #ms
    n_stim = 2
    compile()

    dogON_W = projDoG_ON.w
    print(np.shape(dogON_W))

    dogOFF_W = projDoG_OFF.w
    
    w_inject = projCurrInject_ON.w
    print(w_inject)

#    popInput_OFF = popInput[:,:,1]
    popInput_OFF.current = 2
    
#    popInput_OFF = popInput[:,:,0]
    popInput_OFF.current = 5

    plt.figure()
    plt.imshow(np.reshape(dogON_W[0],(3,3)),cmap='gray')
    plt.colorbar()
    plt.savefig('./dog_W_cell0_popON')
    
    plt.figure()
    plt.imshow(np.reshape(dogOFF_W[0],(3,3)),cmap='gray')
    plt.colorbar()
    plt.savefig('./dog_W_cell0_popOFF')
    

    on_mon = Monitor(popON,['r'])
    off_mon = Monitor(popOFF,['r'])

    lgn_mon_ON = Monitor(popInput,['spike','current','v'])
    #lgn_mon_OFF = Monitor(popInput_OFF,['spike','current','v'])

    print('Start simulation')
    
    for i in range(n_stim):

        patch = setInput(filter_size)

        simulate(duration)


        if((i%(n_stim/10)) == 0):
            print("Round %i of %i" %(i,n_stim))

    print('Simulation is finished')


    current_LGN = lgn_mon_ON.get('current')
    #current_LGN_OFF = lgn_mon_OFF.get('current')
    #vm_LGN = lgn_mon.get('v')

    on_r = on_mon.get('r')
    off_r = off_mon.get('r')


    print(np.shape(on_r))
    print(on_r[10,:])
    print('-----')
    
    plt.figure()
    plt.hist(on_r[10,:],24)
    plt.savefig('./hist_FR_popON_stim5')

    plt.figure()
    plt.hist(off_r[10,:],24)
    plt.savefig('./hist_FR_popOFF_stim5')


    print(np.shape(current_LGN))
    print(current_LGN[10,:])
    print('--------')

#    print(np.shape(current_LGN))
    #print(current_LGN_OFF[10,:])
    #print('--------')

if __name__=="__main__":
    start()

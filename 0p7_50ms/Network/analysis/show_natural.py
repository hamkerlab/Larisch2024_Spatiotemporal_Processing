import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():

    spk_RGC = np.load('./work/show_natScenes_SpikeCount_RGC.npy')
    spk_LGN = np.load('./work/show_natScenes_SpikeCount_LGN.npy')
    spk_E1 = np.load('./work/show_natScenes_SpikeCount_E1.npy')
    spk_I1 = np.load('./work/show_natScenes_SpikeCount_I1.npy')

    vm_E1 = np.load('./work/show_natScenes_MembranPot_E1.npy')
    gEx_E1 = np.load('./work/show_natScenes_gExc_E1.npy')
    gIn_E1 = np.load('./work/show_natScenes_gInh_E1.npy')

    vm_I1 = np.load('./work/show_natScenes_MembranPot_I1.npy')
    gEx_I1 = np.load('./work/show_natScenes_gExc_I1.npy')
    gIn_I1 = np.load('./work/show_natScenes_gInh_I1.npy')


    n_patches,n_repeats,n_LGN = np.shape(spk_LGN)

    n_E1 = int(n_LGN/2)
    n_I1 = int(n_E1/4)
    print(np.shape(vm_E1))

    print('Mean FR RGC = ',np.mean(spk_RGC))
    print('Mean FR LGN = ',np.mean(spk_LGN))
    print('Mean FR E1 = ',np.mean(spk_E1))
    print('Mean FR I1 = ',np.mean(spk_I1))   


    duration = np.shape(vm_E1)[2]
    vmMean_oR_E1 = np.mean(vm_E1,axis=1) #vm_E1[:,2]#
    vmMean_oR_I1 = np.mean(vm_I1,axis=1) #vm_I1[:,2]#

    gEMean_oR_E1 =np.mean(gEx_E1,axis=1) #gEx_E1[:,2] #
    gEMean_oR_I1 =np.mean(gEx_I1,axis=1) #gEx_I1[:,2] #

    gIMean_oR_E1 =np.mean(gIn_E1,axis=1) #gIn_E1[:,2]#
    gIMean_oR_I1 =np.mean(gIn_I1,axis=1) #gIn_I1[:,2] #



    vmMean_oR_E1 = np.reshape(vmMean_oR_E1,(n_patches*duration,n_E1))
    vmMean_oR_I1 = np.reshape(vmMean_oR_I1,(n_patches*duration,n_I1))

    gEMean_oR_E1 = np.reshape(gEMean_oR_E1,(n_patches*duration,n_E1))
    gEMean_oR_I1 = np.reshape(gEMean_oR_I1,(n_patches*duration,n_I1))

    gIMean_oR_E1 = np.reshape(gIMean_oR_E1,(n_patches*duration,n_E1))
    gIMean_oR_I1 = np.reshape(gIMean_oR_I1,(n_patches*duration,n_I1))



    endt = 200

    plt.figure()
    plt.subplot(211)
    plt.plot(vmMean_oR_E1[0:endt,0])
    plt.ylabel('vm')
    plt.xlabel('time')
    plt.subplot(212)
    plt.plot(gEMean_oR_E1[0:endt,0], label='gExc')
    plt.plot(-gIMean_oR_E1[0:endt,0], label='gInh')
    plt.ylabel('currents')
    plt.xlabel('time')
    plt.savefig('./Output/showScenes/vm_cellE0')
    
    plt.figure()
    plt.subplot(211)
    plt.plot(vmMean_oR_I1[0:endt,0])
    plt.ylabel('vm')
    plt.xlabel('time')
    plt.subplot(212)
    plt.plot(gEMean_oR_I1[0:endt,0], label='gExc')
    plt.plot(-gIMean_oR_I1[0:endt,0], label='gInh')
    plt.ylabel('currents')
    plt.xlabel('time')
    plt.savefig('./Output/showScenes/vm_cellI0')

#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

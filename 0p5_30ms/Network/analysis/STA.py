import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def calculateSTA():
    
    if not os.path.exists('Output/STA/'):
        os.mkdir('Output/STA/')


    w_E1 = np.loadtxt('Input_network/V1weight.txt')

    input_data = np.load('./work/STRF_Count_Input.npy',allow_pickle=True)
    n_stim, h, w = np.shape(input_data)

    spkC_RGC = np.load('./work/STRF_SpkCount_RGC.npy',allow_pickle=True)
    n_cRGC = np.shape(spkC_RGC)[1]
    spkC_LGN = np.load('./work/STRF_SpkCount_LGN.npy',allow_pickle=True)
    n_cLGN = np.shape(spkC_LGN)[1]
    spkC_E1 = np.load('./work/STRF_SpkCount_E1.npy',allow_pickle=True)
    n_cE1 = np.shape(spkC_E1)[1]
    spkC_I1 = np.load('./work/STRF_SpkCount_I1.npy',allow_pickle=True)
    n_cI1 = np.shape(spkC_I1)[1]
    spkC_E2 = np.load('./work/STRF_SpkCount_E2.npy',allow_pickle=True)
    n_cE2 = np.shape(spkC_E2)[1]
    spkC_I2 = np.load('./work/STRF_SpkCount_I2.npy',allow_pickle=True)
    n_cI2 = np.shape(spkC_I2)[1]
   

    sX = int(np.ceil(np.sqrt(n_cRGC)))
    plt.figure(figsize=(sX,sX))
    for i in range(n_cRGC):
        plt.subplot(sX,sX,i+1)
        STA = spkC_RGC[:,i, None,None] * input_data
        plt.axis('off')
        plt.imshow(np.sum(STA,axis=0), cmap='gray')
    plt.savefig('Output/STA/STA_RGC')
    plt.close()

    sX = int(np.ceil(np.sqrt(n_cLGN)))
    plt.figure(figsize=(sX,sX))
    for i in range(n_cLGN):
        plt.subplot(sX,sX,i+1)
        STA = spkC_LGN[:,i, None,None] * input_data
        plt.axis('off')
        plt.imshow(np.sum(STA,axis=0), cmap='gray')
    plt.savefig('Output/STA/STA_LGN')
    plt.close()

    sX = int(np.ceil(np.sqrt(n_cE1)))
    plt.figure(figsize=(sX,sX))
    for i in range(n_cE1):
        plt.subplot(sX,sX,i+1)
        STA = spkC_E1[:,i, None,None] * input_data
        plt.axis('off')
        plt.imshow(np.sum(STA,axis=0), cmap='gray')
    plt.savefig('Output/STA/STA_E1')
    plt.close()

    plt.figure(figsize=(sX,sX))
    for i in range(n_cE1):
        plt.subplot(sX,sX,i+1)
        rf = w_E1[i]
        rf = np.reshape(rf, (18,18,2))
        plt.axis('off')
        plt.imshow(rf[:,:,0] - rf[:,:,1], cmap='gray')
    plt.savefig('Output/STA/RF_E1')
    plt.close()

    sX = int(np.ceil(np.sqrt(n_cI1)))
    plt.figure(figsize=(sX,sX))
    for i in range(n_cI1):
        plt.subplot(sX,sX,i+1)
        STA = spkC_I1[:,i, None,None] * input_data
        plt.axis('off')
        plt.imshow(np.sum(STA,axis=0), cmap='gray')
    plt.savefig('Output/STA/STA_I1')
    plt.close()

    sX = int(np.ceil(np.sqrt(n_cE2)))
    plt.figure(figsize=(sX,sX))
    for i in range(n_cE2):
        plt.subplot(sX,sX,i+1)
        STA = spkC_E2[:,i, None,None] * input_data
        plt.axis('off')
        plt.imshow(np.sum(STA,axis=0), cmap='gray')
    plt.savefig('Output/STA/STA_E2')
    plt.close()

    sX = int(np.ceil(np.sqrt(n_cI2)))
    plt.figure(figsize=(sX,sX))
    for i in range(n_cI2):
        plt.subplot(sX,sX,i+1)
        STA = spkC_I2[:,i, None,None] * input_data
        plt.axis('off')
        plt.imshow(np.sum(STA,axis=0), cmap='gray')
    plt.savefig('Output/STA/STA_I2')
    plt.close()
#------------------------------------------------------------------------------
if __name__=="__main__":
    calculateSTA()

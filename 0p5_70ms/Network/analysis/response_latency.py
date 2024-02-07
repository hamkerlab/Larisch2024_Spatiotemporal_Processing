import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_gExc(gExc_E1):
    totalT,n_cells = np.shape(gExc_E1)
    presenT = 250
    n_ele = int(totalT/presenT)

    gExc_E1 = np.reshape(gExc_E1,(n_ele,presenT,n_cells))

    plt.figure()
    plt.plot(gExc_E1[0,:,0])
    plt.savefig('Output/responseLat/gExcCell0_test.png')

    plt.figure()
    plt.plot(np.mean(gExc_E1[0,:,:],axis=1))
    plt.savefig('Output/responseLat/gExcCellAll_mean.png')
    
    gExc_mean_oC = np.mean(gExc_E1,axis=2)

    gExc_mean_oC_oEle = np.mean(gExc_mean_oC,axis=0)

    plt.figure()
    plt.plot(gExc_mean_oC_oEle)
    plt.savefig('Output/responseLat/gExcCellAll_meanOC_meanOEle.png')

    
def startAnalyze():
    
    if not os.path.exists('Output/responseLat/'):
        os.mkdir('Output/responseLat/')

    spk_E1 = np.load('work/responceLat_Spk_E1.npy',allow_pickle=True)
    gExc_E1 = np.load('work/responceLat_gExc_E1.npy',allow_pickle=True)
    
    analyze_gExc(gExc_E1)

    totalT,n_cells = np.shape(gExc_E1)
    presenT = 250
    n_ele = int(totalT/presenT)

    print(np.shape(spk_E1))    
    spk_relative = np.zeros((n_ele,n_cells))
    cc = 0
    for ele in spk_E1:
        inputOffset = 0+presenT*cc
        for cell in ele:
            if len(ele[cell]) == 0:
                spk_relative[cc,cell] = np.nan
            else:
                spkT = ele[cell]
                spk_relative[cc,cell] = spkT[0] - inputOffset
        cc+=1

    meanSpikeOnset = (np.nanmean(spk_relative,axis=0))

    spikeOnset = np.reshape(spk_relative,(n_ele*n_cells))

    plt.figure()
    plt.hist(meanSpikeOnset,24)
    plt.savefig('Output/responseLat/meanOnsetHist.png')

    plt.figure()
    plt.hist(spikeOnset[~np.isnan(spikeOnset)],24)
    plt.savefig('Output/responseLat/OnsetHist.png')

    print('Median Response Latency = ', np.nanmedian(spikeOnset))
    print('Mean Response Latency = ', np.nanmean(spikeOnset))
    print('Max Response Latency = ', np.nanmax(spikeOnset))
    print('Min Response Latency = ', np.nanmin(spikeOnset))

#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyze()


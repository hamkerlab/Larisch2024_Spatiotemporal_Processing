import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import trange

def startAnalyze():

    if not os.path.exists('Output/RGC/'):
        os.mkdir('Output/RGC/')


    time_back = 300 #ms time to look back

    spk_RGC = np.load('./work/STRF_Spk_RGC.npy',allow_pickle=True)
    n_stim = len(spk_RGC)
    n_cells = len(spk_RGC[0])
    isi_l = []
    for i in trange(int(n_stim/2),ncols=50):
        stim = spk_RGC[i]
        for c in stim:
            if len(stim[c]) >=2:
                isi = np.diff(stim[c])
                isi_l = isi_l + isi.tolist()
                #print(stim[c], isi)
    isi_l = np.asarray(isi_l, dtype="int32")   
    print(np.shape(isi_l))         
    print('Create histogram')            
    plt.figure()
    plt.hist(isi_l,30)
    plt.savefig('Output/RGC/rgc_ISI.png')


if __name__=="__main__":
    startAnalyze()

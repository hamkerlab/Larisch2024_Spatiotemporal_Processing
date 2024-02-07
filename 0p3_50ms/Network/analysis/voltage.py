import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path
#------------------------------------------------------------------------------
def startAnalyse():
    if not os.path.exists('Output/Activ/Membrane/'):
        os.mkdir('Output/Activ/Membrane/')

    memExc = np.load('./work/voltage_Exc_V1VM.npy')
    memN   = np.load('./work/voltage_N_V1VM.npy')
    memInh = np.load('./work/voltage_Inh_V1VM.npy')

    idx = np.where(memExc > 0.0)

    plt.figure()
    plt.plot(memExc[1],'r-')
    plt.plot(memN[1],'g-')
    plt.plot(memInh[1],'b-')
    plt.savefig('./Output/Activ/Membrane/test.png')
#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyse()

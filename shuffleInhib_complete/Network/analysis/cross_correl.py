import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import spiking as spk
import math
#def correlogram(T1, T2, width=20*ms , bin=1*ms , T=None):
def correlogram(T1, T2, width=20 , bin=1 , T=None):
# Module to compute the cross-correlogram
# from  Brian 1.4.3 
    '''
    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).
    N.B.: units are discarded.
    TODO: optimise?
    '''
    if (T1==[]) or (T2==[]): # empty spike train
        return np.nan
    # Remove units
    width = float(width)
    T1 = np.array(T1)
    T2 = np.array(T2)
    i = 0
    j = 0
    n = int(math.ceil(width / bin)) # Histogram length
    l = []
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        l.extend(T2[i:j] - t)
    H, _ = np.histogram(l, bins=np.arange(2 * n + 1) * bin - n * bin) #, new = True)

    # Divide by time to get rate
    if T is None:
        T = max(T1[-1], T2[-1]) - min(T1[0], T2[0])
    # Windowing function (triangle)
    W = np.zeros(2 * n)
    W[:n] = T - bin * np.arange(n - 1, -1, -1)
    W[n:] = T - bin * np.arange(n)

    return (H/W)
#------------------------------------------------------------------------------#
def test():
    ms = 1/1000.0
    tmRF = np.load('./work/TemplateMatch.npy')
    correlRF = np.load('./work/correlationRF_V1.npy')
    spikings = np.load('./work/frSingleSpkes.npy')
    spikings = spikings.item()
    nbrOfNeurons = np.shape(correlRF)[0]
    #set primary diagonal to zero to get the maximums between the receptive fields
    for i in range(nbrOfNeurons):
        correlRF[i,i] = 0.0
    #get maximum correlation
    idxMax = np.where(correlRF == np.max(correlRF))
    print(idxMax)

    #set primary diagonal to one to get the minimum between the receptive fields
    for i in range(nbrOfNeurons):
        correlRF[i,i] = 1.0
    #get maximum correlation
    idxMin = np.where(correlRF == np.min(correlRF))
    print(correlRF[idxMin[0][0],idxMin[0][1]])

    
    spkMax1 = np.asarray(spikings[idxMax[0][0]])
    spkMax2 = np.asarray(spikings[idxMax[0][1]])

    spkMin1 = np.asarray(spikings[idxMin[0][0]])
    spkMin2 = np.asarray(spikings[idxMin[0][1]])
    width =100.0*ms

    corMax = correlogram(spkMax1*ms,spkMax2*ms,width=width,bin=1*ms, T = 10000.0*ms)
    print(np.shape(corMax))
    corMin = correlogram(spkMin1*ms,spkMin2*ms,width=width,bin=1*ms, T = 10000.0*ms)
    print(np.shape(corMin))

    plt.figure()
    plt.plot(corMax,'r-+',label='max corr')
    plt.plot(corMin,'b-o',label='min corr')
    plt.ylim(ymin = -0.2)
    plt.xticks(np.linspace(0,len(corMax),5),np.linspace(-width/ms,width/ms,5))
    plt.xlabel('Lag [ms]')
    plt.ylabel('Coincidences')
    plt.legend()
    plt.savefig('./Output/Activ/Correlogram.png')
#------------------------------------------------------------------------------
def calcCrossCorrels(spikings,tmRF):
    nbrOfNeurons = np.shape(tmRF)[0]
    correlM = np.zeros((nbrOfNeurons,nbrOfNeurons,100*2))
    ms = 1/1000.0
    width =100.0*ms
    binSize = 0.25
    minTM = -1.0
    maxTM = 1.0    
    nbrOfBins = int((abs(minTM)+maxTM)/binSize)
    correlCells = np.zeros((nbrOfNeurons,nbrOfNeurons,100*2)) #cross-correl between each neuron pair
    binCorrel = np.zeros((nbrOfBins,100*2))#cross-correl pooled to bin over their similarity
    for i in xrange(nbrOfBins):
        idx = np.where((tmRF>minTM+(i*binSize)) & (tmRF<minTM+binSize+(i*binSize)))
        postIdx = idx[0]
        preIdx = idx[1]
        for j in xrange(len(postIdx)):
            spkPost = np.asarray(spikings[postIdx[j]])
            spkPre = np.asarray(spikings[preIdx[j]])
            correl = correlogram(spkPost*ms,spkPre*ms,width=width,bin=1*ms, T = 10000.0*ms)
            correlCells[postIdx[j],preIdx[j] ] = correl
            binCorrel[i] +=correl
        binCorrel[i] /= len(postIdx)
    np.save('./work/correlCells.npy',correlCells)
    np.save('./work/binCorrel.npy',binCorrel)
    return(correlCells,binCorrel)
#------------------------------------------------------------------------------
def start():
    #need array with time points !
    tmRF = np.load('./work/TemplateMatch.npy')
    correlRF = np.load('./work/correlationRF_V1.npy')
    spikings = np.load('./work/frSingleSpkes.npy')
    spikings = spikings.item()

    nbrCells = len(spikings)
    duration = 125
    nbrOfInputs = 30000
    nbrOfPatches = 5000
    maxSteps = nbrOfPatches*125
    nbrOfTimeSteps = duration*nbrOfInputs
    spikeMatrix = np.zeros((nbrCells,nbrOfTimeSteps))
    print(nbrOfTimeSteps)
    for i in range(nbrCells):
        array = spikings[i]
        spikeMatrix[i,array] = 1
    print('Calc spike counts.')
    cellSpks = spk.calcSpikeCount(spikeMatrix[:,0:maxSteps],nbrOfPatches)
    print('Spike counts are finish')
    #corrT = np.correlate(spikeMatrix[0,0:duration],spikeMatrix[1,0:duration],'same')
    corrT = np.correlate(cellSpks[0],cellSpks[1],'full')
    plt.figure()
    plt.plot(corrT)
    plt.savefig('test.png')

    print('Blubb')

#    for i in xrange(nbrOfNeurons):
#        spkPost = np.asarray(spikings[i])
#        for j in xrange(nbrOfNeurons):
#            spkPre = np.asarray(spikings[j])
#            correl = correlogram(spkPost*ms,spkPre*ms,width=width,bin=1*ms, T = 1000.0*ms)
#            correlM[i,j] = correl

    calcCrossCorrels(spikings,tmRF)

    correlCells = np.load('./work/correlCells.npy')
    binCorrel = np.load('./work/binCorrel.npy')
    nbr,size = np.shape(binCorrel)

    dataPoints = np.shape(binCorrel[0,size/2-10:size/2+10])[0]
    minTM = -1.0
    maxTM = 1.0
    binStep = maxTM/(nbr/2)
    cmap = plt.get_cmap('brg')
    colors = [cmap(i) for i in np.linspace(0,1,nbr)]
    markers = ['+-','<-','*-','^-','>-','.-','o-','x-']
    plt.figure()
    #plt.ylim(ymin = -0.2)
    for i in xrange(nbr):
        plt.plot(binCorrel[i,size/2-10:size/2+11],markers[i],label='From '+str(minTM+ i*binStep)+' to '+str(minTM+binStep+ i*binStep),color=colors[i])#,marker=markers[i] )
    plt.xticks(np.linspace(0,dataPoints,5),np.linspace(-dataPoints/2,(dataPoints/2),5))
    plt.xlabel('Lag [ms]')
    plt.ylabel('Coincidences')
    plt.legend()
    plt.savefig('./Output/Activ/Correlogram_BIN.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    
#------------------------------------------------------------------------------
def startCrossCorelo():
    #test()
    start()
#------------------------------------------------------------------------------
if __name__=="__main__":
    startCrossCorelo()

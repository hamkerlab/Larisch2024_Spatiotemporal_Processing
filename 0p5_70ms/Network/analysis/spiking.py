import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mutualInformation as mI

def calcFR(frMatrix,nbrOfPatches):
    #calculate the fire rates of every neuron for ervery input patch
    nbrOfneuronsExc,nbrOfSteps = np.shape(frMatrix)
    duration = nbrOfSteps/nbrOfPatches
    fr = np.zeros((nbrOfneuronsExc,nbrOfPatches))
    for i in range(nbrOfneuronsExc):
        for j in range(nbrOfPatches):
            fr[i,j] = np.sum(frMatrix[i,0+(duration*j):duration+(duration*j)])*1000/duration
    return(fr)
#------------------------------------------------------------------------------
def calcSpikeCount(frMatrix,nbrOfPatches):
    #calculate the fire rates of every neuron for ervery input patch
    nbrOfneuronsExc,nbrOfSteps = np.shape(frMatrix)
    duration = nbrOfSteps/nbrOfPatches
    fr = np.zeros((nbrOfneuronsExc,nbrOfPatches))
    for i in range(nbrOfneuronsExc):
        for j in range(nbrOfPatches):
            fr[i,j] = np.sum(frMatrix[i,0+(duration*j):duration+(duration*j)])
    return(fr)
#------------------------------------------------------------------------------
def calcFirstSpk(frMatrix,nbrOfPatches):
    #calculate a wighted value, dependent on the time step of the first spike
    nbrOfneuronsExc,nbrOfSteps = np.shape(frMatrix)
    duration = nbrOfSteps/nbrOfPatches
    spkFirst = np.zeros((nbrOfneuronsExc,nbrOfPatches))
    tau = 125.0
    for i in range(nbrOfneuronsExc):
        for j in range(nbrOfPatches):
            inx = np.where(frMatrix[i,0+(duration*j):duration+(duration*j)] == 1) # search at witch time step the neuron fired at the patch
            if len(inx[0]) == 0: 
                spkFirst[i,j] = 0 # if the neuron not fired, zero
            else:
                # wighted Value dependent on the time step of the first encounterd spike; earlier the spike, higher the value (t_0 =1)
                spkFirst[i,j] = (np.exp(-inx[0][0]/tau)) 
    return(spkFirst)
#------------------------------------------------------------------------------
def calcSumSpk(frMatrix,nbrOfPatches):
    #calculate sum over a wighted value, dependent on the time steps of the spikes
    nbrOfneuronsExc,nbrOfSteps = np.shape(frMatrix)
    duration = nbrOfSteps/nbrOfPatches
    spkSum = np.zeros((nbrOfneuronsExc,nbrOfPatches))
    tau = 125.0
    for i in range(nbrOfneuronsExc):
        for j in range(nbrOfPatches):
            inx = np.where(frMatrix[i,0+(duration*j):duration+(duration*j)] == 1) # search at witch time step the neuron fired at the patch
            if len(inx[0]) == 0: 
                spkSum[i,j] = 0 # if the neuron not fired, zero
            else:
                spkSum[i,j] = np.sum(np.exp(-inx[0]/tau)) # sum over the timestep dependent values 
    return(spkSum)
#------------------------------------------------------------------------------
def estimateOnFirstSpike(frMatrix,nbrOfPatches, tau = 125): # with a tau of 125 ms
    print('estimade fire rate over time between on set and first spike')
    nbrOfNeurons,nbrOfSteps = np.shape(frMatrix)
    estimatedFR = np.zeros((nbrOfNeurons,nbrOfPatches))
    for i in range(nbrOfNeurons):
        for j in range(nbrOfPatches):
            inx = np.where(frMatrix[i,0+(tau*j):tau+(tau*j)] == 1) # search at witch time step the neuron fired at the patch
            if len(inx[0]) > 0: 
                if (inx[0][0] == 0):
                    estimatedFR[i,j] = tau
                else:
                    estimatedFR[i,j] = tau/inx[0][0]
    return(estimatedFR)
#------------------------------------------------------------------------------
def estimateOnFirstSpike2(frMatrix,nbrOfPatches, tau):
    #second version with variable tau: not working !
    nbrOfNeurons,nbrOfSteps = np.shape(frMatrix)
    estimatedFR = np.zeros((nbrOfNeurons,nbrOfPatches))
    bins = int(nbrOfSteps/tau)
    duration = int(nbrOfSteps/nbrOfPatches)
    binSize = int(tau/duration)
    for i in range(nbrOfNeurons):
        c = 0
        for j in range(bins):
            bin = frMatrix[i,0+(tau*j):tau+(tau*j)]
            for k in range(binSize):
                activity =  bin[0+(duration*k):duration+(duration*k)]
                inx = np.where(activity == 1)
                if len(inx[0] >0):
                    estimatedFR[i,c] = inx[0][0]+(duration*k)
                c+=1
    return(estimatedFR)
#------------------------------------------------------------------------------
def estimateOnFirstTwoSpike(frMatrix,nbrOfPatches):
    print('estimade fire rate over time between first and second spike')
    nbrOfNeurons,nbrOfSteps = np.shape(frMatrix)
    duration = 125.0
    estimatedFR = np.zeros((nbrOfNeurons,nbrOfPatches))
    for i in range(nbrOfNeurons):
        for j in range(nbrOfPatches):
            inx = np.where(frMatrix[i,0+(duration*j):duration+(duration*j)] == 1)
            if len(inx[0]) == 0: 
                estimatedFR[i,j] = 0 # if the neuron not fired, zero
            if len(inx[0]) == 1:
                estimatedFR[i,j] = 1 # if only one spike occurred, fire rate is one
            if len(inx[0]) >= 2:
                estimatedFR[i,j] = duration/(inx[0][1] - inx[0][0]) 
    return(estimatedFR)
#------------------------------------------------------------------------------
def help():
    print('Methods to calculate the fire rates and the weighted values dependent on time step relativ to stimulus onset. Need a matrix in shape=((NumberOfNeurons,NumberOfAllSimulationSteps)) and the number of input Patches')

#------------------------------------------------------------------------------
def startAnalyseSpiking():
    print('Start to Analyse the time spiking behavior')
    excSpks = np.load('work/frSingleSpkes.npy')
    excSpks = excSpks.item()
    nbrOfNeurons = len(excSpks)
    duration = 125
    nbrOfInputs = 30000
    nbrOfTimeSteps = duration*nbrOfInputs
    spikeMatrix = np.zeros((nbrOfNeurons,nbrOfTimeSteps))
    for i in range(nbrOfNeurons):
        array = excSpks[i]
        spikeMatrix[i,array] = 1
    #timeStep = 1000.0
    estimatedFRFirst = estimateOnFirstSpike(spikeMatrix,nbrOfInputs,duration) #*1000/duration
    neuronFR = calcFR(spikeMatrix,nbrOfInputs)# 

    #miSpikes = np.zeros((nbrOfNeurons,nbrOfNeurons))
    #binSize = 1000    
    #print('Start to calculate the mutual information over spikes')
    #for i in range(nbrOfNeurons):
    #    for j in range(nbrOfNeurons):
    #       miSpikes[i,j] = mI.calc_MI_HistBit(spikeMatrix[i],spikeMatrix[j],binSize,False)
    #print(i)
    #np.save('work/MI_Spikes',miSpikes)

    plt.figure()
    plt.plot(np.reshape(estimatedFRFirst, nbrOfNeurons*nbrOfInputs),np.reshape(neuronFR, nbrOfNeurons*nbrOfInputs),'o')
    plt.xlabel('estimated FR [Hz]')
    plt.ylabel('measured FR [Hz]')
    plt.savefig('Output/MeasuredToEstimated_FirstSpike.png')
    
    estimatedFRTwo = estimateOnFirstTwoSpike(spikeMatrix,nbrOfInputs) *1000/duration
    plt.figure()
    plt.plot(np.reshape(estimatedFRTwo, nbrOfNeurons*nbrOfInputs),np.reshape(neuronFR, nbrOfNeurons*nbrOfInputs),'o')
    plt.xlabel('estimated FR [Hz]')
    plt.ylabel('measured FR [Hz]')
    plt.savefig('Output/MeasuredToEstimated_SecondSpike.png')
    

    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,10), sharey=True)
    ax1.hist(np.reshape(neuronFR, nbrOfNeurons*nbrOfInputs),10,log=True)
    ax1.set_title('measured firing rates')
    ax2.hist(np.reshape(estimatedFRFirst, nbrOfNeurons*nbrOfInputs),10,log=True)
    ax2.set_title('estimated over first spike')
    ax3.hist(np.reshape(estimatedFRTwo, nbrOfNeurons*nbrOfInputs),10,log=True)
    ax3.set_title('estimated over first two spikes')
    plt.savefig('Output/FireRates_Hists.png')
#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyseSpiking()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mutualInformation as mi

#------------------------------------------------------------------------------
def calcMutualInformation(frExc,binSize,desc):
    # calculates the mutual information between the fire rates of every neuron
    # binSize determine the number of simulation steps, for them the MI are calculated
    print('Calculate Mutual Information')
    nbrOfPatches,nbrOfNeurons = (np.shape(frExc))   
    spkMI = np.zeros((nbrOfPatches/binSize,nbrOfNeurons,nbrOfNeurons))
    for i in range(nbrOfPatches/binSize):
        for j in range(nbrOfNeurons):
            for k in range(nbrOfNeurons):
                start = 0+(i*binSize)
                end = binSize+(i*binSize)
                spkMI[i,j,k] = mi.calc_MI_Hist(frExc[start:end,j],frExc[start:end,k],binSize,True)
        if((i%(nbrOfPatches/binSize/10)) == 0):
            print("Round %i of %i by MI" %(i,nbrOfPatches/binSize))
    #save the calculated MI
    np.save('./work/learning_MI_'+desc,spkMI)
#------------------------------------------------------------------------------
def calcCorrelation(frExc,binSize,desc):
    print('Calculate the Correlation between the fire rates during the time')
    nbrOfPatches,nbrOfNeurons = (np.shape(frExc))
    corrM = np.zeros((nbrOfPatches/binSize,nbrOfNeurons,nbrOfNeurons))
    for i in range(nbrOfPatches/binSize):
        start = 0+(i*binSize)
        end = binSize+(i*binSize)
        for j in range(nbrOfNeurons):
            frPost = frExc[start:end,j]
            for k in range(nbrOfNeurons):
                frPre = frExc[start:end,k]
                corrM[i,j,k] =  np.corrcoef(frPost,frPre)[0,1]
        if((i%(nbrOfPatches/binSize/10)) == 0):
            print("Round %i of %i by Corr" %(i,nbrOfPatches/binSize))
    np.save('./work/learning_Corr_'+desc,corrM)
#------------------------------------------------------------------------------
def plotMeanMI(frMI,maxPatch,duration,desc):
    # plot the mean curve of the mean MI over the time
    # first, delete the MI of a neuron with themselve
    nbrOfValues,nbrPost,nbrPre =np.shape(frMI)
    # set the correlated values np.nan
    for i in range(nbrPost):
        frMI[:,i,i] = np.nan
    frMIArray = np.reshape(frMI, (nbrOfValues,nbrPost*nbrPre))
    frMIClean = np.zeros((nbrOfValues,nbrPost*nbrPre-nbrPre))

    # reshape the MI values for every step in array
    # and sort out the nan values
    for i in range(nbrOfValues):
        frMIClean[i] = frMIArray[i,~np.isnan(frMIArray[i])]
    #plot
    plt.figure(figsize=(10,5))
    plt.plot((np.mean(frMIClean,axis=1)))
    plt.ylabel('average MI')
    plt.xlabel('Time in s')
    plt.xticks(np.linspace(0,nbrOfValues,5),np.linspace(0,maxPatch*duration/1000.0,5))
    plt.ylim(ymin=0.0,ymax=0.2)
    plt.savefig('./Output/meanMI_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def plotMeanCorr(frCorr,maxPatch,duration,desc):
    nbrOfValues,nbrPost,nbrPre =np.shape(frCorr)
    for i in range(nbrPost):
        frCorr[:,i,i] = np.nan
    frCorrArray = np.reshape(frCorr, (nbrOfValues,nbrPost*nbrPre))
    frCorrClean = np.zeros((nbrOfValues,nbrPost*nbrPre-nbrPre))
    for i in range(nbrOfValues):
        frCorrClean[i] = frCorrArray[i,~np.isnan(frCorrArray[i])]

    plt.figure(figsize=(10,5))
    plt.plot((np.mean(frCorrClean,axis=1)))
    plt.ylabel('average Correlation')
    plt.xlabel('Time in s')
    plt.xticks(np.linspace(0,nbrOfValues,5),np.linspace(0,maxPatch*duration/1000.0,5))
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.savefig('./Output/meanCorr_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotFireRate(frExc,desc):
    meanFR = np.mean(frExc,axis=0)
    plt.figure()
    plt.hist(meanFR,15)
    plt.savefig('./Output/meanFR_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
    
    plt.figure()
    plt.plot(frExc[:,0])
    plt.savefig('./Output/fr_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(np.max(frExc,axis=1))
    plt.savefig('./Output/maxFr_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(frExc[:,0],15)
    plt.savefig('./Output/frHist_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def startAnalysis():
    print('Start to analyise the correlation and mutual information between the spiking activities')
    frExc = np.load('./Input_network/frExc.npy')
    frInh = np.load('./Input_network/frInh.npy')
    maxPatch = 400000#100000# # 10000*1250ms = 1250s
    duration = 125
    frExc = frExc[0:maxPatch,:]
    frInh = frInh[0:maxPatch,:]
    binSize = 1000 #bin size of 100 Patches = 100*125ms = 12,5 s
    #plotFireRate(frExc,'exc')
    #plotFireRate(frInh,'inh')
    #calcMutualInformation(frExc,binSize,'exc')
    calcCorrelation(frExc,binSize,'exc')
    frMI = np.load('./work/learning_MI_exc.npy')
    #print(np.shape(frMI))
    frCorr= np.load('./work/learning_Corr_exc.npy')
    #print(np.shape(frCorr))
    #plotMeanMI(frMI,maxPatch,duration,'exc')
    plotMeanCorr(frCorr,maxPatch,duration,'exc')
    
#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalysis()

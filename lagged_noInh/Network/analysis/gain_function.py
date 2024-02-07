import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path
#------------------------------------------------------------------------------
def plotGain(fr,current,manner):

    plt.figure()
    plt.plot(current[0,:],fr[0,:],'-o')
    plt.savefig('./Output/Gain/Single_'+manner+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(np.mean(current,axis=0),np.mean(fr,axis=0))
    plt.xlabel(manner)
    plt.ylabel('firing rate')
    #plt.xlim(0.0,0.7)
    #plt.ylim(0.0,60)
    plt.savefig('./Output/Gain/Mean_'+manner+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def getCurveValues(frTC,gETC,gITC):
    nbrOfNeurons,sizeX,sizeY,nbrOfPatches = np.shape(frTC)
    curveFR = np.zeros((nbrOfNeurons,nbrOfPatches))
    curvegE = np.zeros((nbrOfNeurons,nbrOfPatches))
    curvegI = np.zeros((nbrOfNeurons,nbrOfPatches))
    for i in range(nbrOfNeurons):
        goodIndx = (np.where(frTC[i,:,:,:] ==np.max(frTC[i,:,:,:])))
        sumFR = np.sum(frTC[i,goodIndx[0],goodIndx[1]],axis=1)
        probBestIdx = np.where(sumFR ==np.max(sumFR)) # Index with probaly the best tuning Curve
        backIdX = goodIndx[0][probBestIdx[0][0]]
        backIdY = goodIndx[1][probBestIdx[0][0]]
        nbrOfCoords = (np.shape(goodIndx))[1]
        curveFR[i] = frTC[i,backIdX,backIdY,:]
        curvegE[i] = gETC[i,backIdX,backIdY,:]
        curvegI[i] = gITC[i,backIdX,backIdY,:]
    return(curveFR,curvegE,curvegI)
#------------------------------------------------------------------------------
def startAnalyseGainFunction():

    if not os.path.exists('./Output/Gain'):
        os.mkdir('./Output/Gain')

#################################################
#     over data from Gabor-function TC          #
#################################################

    frExcTC = np.load('./work/TuningCurve_frEx.npy')
    frInhTC = np.load('./work/TuningCurve_frInhib.npy')
    gE_ExcTC = np.load('./work/TuningCurve_gExcEx.npy')
    gI_ExcTC = np.load('./work/TuningCurve_gInhEx.npy')

    contrastLVL = 4#np.shape(frExcTC)[4]
    frExcTC = np.mean(frExcTC[:,:,:,:,contrastLVL-1],axis=4)
    gE_ExcTC = np.mean(gE_ExcTC[:,:,:,:,contrastLVL-1],axis=4)
    frInhTC = np.mean(frInhTC[:,:,:,:,contrastLVL-1],axis=4)
    gI_ExcTC = np.mean(gI_ExcTC[:,:,:,:,contrastLVL-1],axis=4)


    curveFR,curvegE,curvegI = getCurveValues(frExcTC,gE_ExcTC,gI_ExcTC)
    nbrCells,nbrPatches = np.shape(curveFR)

    sortedFR = np.zeros((nbrCells,nbrPatches))
    sortedGExc=np.zeros((nbrCells,nbrPatches))
    sortedGInh=np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):
        idx = np.argsort(curvegE[i,:])
        sortedFR[i,:]= curveFR[i,idx]
        sortedGExc[i,:] = curvegE[i,idx]

    plotGain(sortedFR,sortedGExc,'g_Exc_TC')
    for i in range(nbrCells):
        idx = np.argsort(curvegI[i,:])
        sortedFR[i,:]= curveFR[i,idx]
        sortedGInh[i,:] = curvegI[i,idx]

    plotGain(sortedFR,sortedGInh,'g_Inh_TC')
#------------------------------------------------------------------------------
#################################################
#       over data from sinus TC task            #
#################################################

    frExcTC = np.load('./work/TuningCurve_sinus_frEx.npy')
    frInhTC = np.load('./work/TuningCurve_sinus_frInhib.npy')
    gE_ExcTC= np.load('./work/TuningCurve_sinus_gExc.npy')
    gI_ExcTC= np.load('./work/TuningCurve_sinus_gInh.npy')

    contrastLVL = 4#np.shape(frExcTC)[4]
    phaseStep = 0
    frequencyStep = 0 

    frExcTC = frExcTC[:,:,phaseStep,frequencyStep,contrastLVL-1]
    gE_ExcTC = gE_ExcTC[:,:,phaseStep,frequencyStep,contrastLVL-1]
    frInhTC = frInhTC[:,:,phaseStep,frequencyStep,contrastLVL-1]
    gI_ExcTC = gI_ExcTC[:,:,phaseStep,frequencyStep,contrastLVL-1]

    #curveFR,curvegE,curvegI = getCurveValues(frExcTC,gE_ExcTC,gI_ExcTC)
    curveFR = np.mean(frExcTC,axis=2)
    curvegE = np.mean(gE_ExcTC,axis=2)
    curvegI = np.mean(gI_ExcTC,axis=2)
    nbrCells,nbrPatches = np.shape(curveFR)

    sortedFR = np.zeros((nbrCells,nbrPatches))
    sortedGExc=np.zeros((nbrCells,nbrPatches))
    sortedGInh=np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):
        idx = np.argsort(curvegE[i,:])
        sortedFR[i,:]= curveFR[i,idx]
        sortedGExc[i,:] = curvegE[i,idx]

    plotGain(sortedFR,sortedGExc,'g_Exc_sinus')
    for i in range(nbrCells):
        idx = np.argsort(curvegI[i,:])
        sortedFR[i,:]= curveFR[i,idx]
        sortedGInh[i,:] = curvegI[i,idx]

    plotGain(sortedFR,sortedGInh,'g_Inh_TC_Sinus')

#------------------------------------------------------------------------------
#################################################
#      over data from fluctuation task          #
#################################################

    fr_Exc = np.load('./work/fluctuation_frExc.npy')
    gE_Exc = np.load('./work/fluctuation_V1_gExc.npy')
    gI_Exc = np.load('./work/fluctuation_V1_gInh.npy')

    nbrCells,nbrPatches,nbrSamples = np.shape(fr_Exc)
    
    meanFR = np.mean(fr_Exc,axis=2)  #mean over singel samples of one patch
    meangExc = np.mean(gE_Exc,axis=2)
    meangInh = np.mean(gI_Exc,axis=2)

    sortedFR = np.zeros((nbrCells,nbrPatches))
    sortedGExc=np.zeros((nbrCells,nbrPatches))
    sortedGInh=np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):
        idx = np.argsort(meangExc[i,:])
        sortedFR[i,:]= meanFR[i,idx]
        sortedGExc[i,:] = meangExc[i,idx]
    
    plotGain(sortedFR,sortedGExc,'g_Exc_Scene')
    for i in range(nbrCells):
        idx = np.argsort(meangInh[i,:])
        sortedFR[i,:]= meanFR[i,idx]
        sortedGInh[i,:] = meangInh[i,idx]

    plotGain(sortedFR,sortedGInh,'g_Inh_Scene')
#------------------------------------------------------------------------------
#################################################
#        over data from contrast task           #
#################################################

    fr_Exc = np.load('./work/contrast_V1FR.npy')
    gE_Exc = np.load('./work/contrast_V1gE.npy')
    gI_Exc = np.load('./work/contrast_V1gI.npy')

    nbrCells,nbrPatches,nbrSamples = np.shape(fr_Exc)
    
    meanFR = np.mean(fr_Exc,axis=2)  #mean over singel samples of one patch
    meangExc = np.mean(gE_Exc,axis=2)
    meangInh = np.mean(gI_Exc,axis=2)

    sortedFR = np.zeros((nbrCells,nbrPatches))
    sortedGExc=np.zeros((nbrCells,nbrPatches))
    sortedGInh=np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):
        idx = np.argsort(meangExc[i,:])
        sortedFR[i,:]= meanFR[i,idx]
        sortedGExc[i,:] = meangExc[i,idx]
    
    plotGain(sortedFR,sortedGExc,'g_Exc_Contrast')
    for i in range(nbrCells):
        idx = np.argsort(meangInh[i,:])
        sortedFR[i,:]= meanFR[i,idx]
        sortedGInh[i,:] = meangInh[i,idx]

    plotGain(sortedFR,sortedGInh,'g_Inh_Contrast')
#------------------------------------------------------------------------------
def startGainNatural():
#################################################
#            over natural scenes                #
#################################################

    fr_Exc = np.load('./work/Active_fr.npy')
    print(np.shape(fr_Exc))
#------------------------------------------------------------------------------
if __name__ == "__main__":
    startAnalyseGainFunction()
    #startGainNatural()

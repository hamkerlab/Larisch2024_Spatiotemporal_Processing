import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os.path
#------------------------------------------------------------------------------
def plotMeanToVar(meanSPC,varSPC,layer):

    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(meanSPC, varSPC)
    regrs_Line = intercept + slope*meanSPC
    mp.pyplot.figure()
    mp.pyplot.plot(meanSPC,regrs_Line,'k-')
    mp.pyplot.plot(meanSPC,varSPC,'o')
    mp.pyplot.xlabel('mean')
    mp.pyplot.ylabel('variance')
    #mp.pyplot.xlim(1.0,2.5)
    #mp.pyplot.ylim(1.0,11.0)
    mp.pyplot.savefig('./Output/MeanToVar/MeanToVar1_'+layer+'.png',bbox_inches='tight', pad_inches = 0.1)


    grad = 1
    regrs_Line = np.polyfit(meanSPC,varSPC,grad)
    x = np.linspace(np.min(meanSPC),np.max(meanSPC),grad+1)
    regrs = regrs_Line[0]*x + regrs_Line[1]
    mp.pyplot.figure()
    mp.pyplot.plot(x,regrs,'k-')
    mp.pyplot.plot(meanSPC,varSPC,'o')
    mp.pyplot.xlabel('mean')
    mp.pyplot.ylabel('variance')
    #mp.pyplot.xlim(1.0,2.5)
    #mp.pyplot.ylim(1.0,11.0)
    mp.pyplot.savefig('./Output/MeanToVar/MeanToVar2'+layer+'.png',bbox_inches='tight', pad_inches = 0.1)

    meanSPC /=np.max(meanSPC)
    varSPC /=np.max(varSPC)
    grad = 1
    regrs_Line = np.polyfit(meanSPC,varSPC,grad)
    x = np.linspace(np.min(meanSPC),np.max(meanSPC),grad+1)
    regrs_Line = regrs_Line[0]*x + regrs_Line[1]
    mp.pyplot.figure()
    mp.pyplot.plot(x,regrs_Line,'k-')
    mp.pyplot.plot(meanSPC,varSPC,'o')
    mp.pyplot.xlabel('mean')
    mp.pyplot.ylabel('variance')
    mp.pyplot.xlim(0.0,1.2)
    mp.pyplot.ylim(0.0,1.2)
    mp.pyplot.savefig('./Output/MeanToVar/MeanToVarNormed'+layer+'.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def plotDataPoints(meanV1,varV1,meanLGN,varLGN):

    nbrCellsLGN,nbrPatches = np.shape(meanLGN)
    nbrCellsV1 = np.shape(meanV1)[0]

    slopesLGN = np.zeros(nbrCellsLGN) 
    offsetsLGN= np.zeros(nbrCellsLGN)

    plt.figure()
    for i in xrange(nbrCellsLGN):
        idx = np.argsort(meanLGN[i])
        mean = meanLGN[i,idx]
        var = varLGN[i,idx] 
        slopesLGN[i], offsetsLGN[i] = np.polyfit(mean,var,1)
        plt.plot(mean,var,'bo')
        plt.xlabel('mean spike count')
        plt.ylabel('variance spike count')
    plt.savefig('./Output/MeanToVar/LGN_ALL')
    print(np.mean(slopesLGN))

    slopesV1 = np.zeros(nbrCellsV1) 
    offsetsV1= np.zeros(nbrCellsV1)

    plt.figure()
    for i in xrange(nbrCellsV1):
        #idx = np.argsort(meanV1[i])
        mean = meanV1[i]#,idx]
        var = varV1[i]#,idx]
        idD = np.where(mean >=5.0)
        idU = np.where(mean >=7.0)
        if(len(idU[0])>0 and len(idD)>0 ):
            slopesV1[i], offsetsV1[i]= np.polyfit(mean[idD[0][0]:idU[0][0]],var[idD[0][0]:idU[0][0]],1)
        else:
            slopesV1[i], offsetsV1[i]= np.polyfit(mean,var,1)
        plt.plot(mean,var,'bo')
        plt.xlabel('mean spike count')
        plt.ylabel('variance spike count')
    plt.plot(np.linspace(5,8,10),np.mean(slopesV1)*np.linspace(5,8,10)+np.mean(offsetsV1),'k-+',lw=2.0)
    plt.xlim(0,15)
    plt.ylim(0,15)
    plt.savefig('./Output/MeanToVar/V1_ALL.png')
    print(np.mean(slopesV1))

    x = np.linspace(0,15)
    l1 = np.mean(slopesV1) * x
    l2 = np.mean(slopesLGN) * x
    plt.figure()
    plt.plot(x,l1,'b--',label='V1')
    plt.plot(x,l2,'r-',label='LGN')
    plt.legend()
    plt.savefig('./Output/MeanToVar/Slopes.png')
#------------------------------------------------------------------------------
def analyseSpikeCount(spcMatrix,tmRF):
    nbrCells,nbrPatches,nbrSamples = np.shape(spcMatrix)

    diffCounts = np.zeros((nbrCells,nbrPatches,nbrSamples-1))

    for i in xrange(nbrCells):
        for j in xrange(nbrPatches):
            for k in xrange(nbrSamples-1):
                diffCounts[i,j,k] = np.abs(spcMatrix[i,j,k+1] - spcMatrix[i,j,k])

    meanDiff = (np.mean(diffCounts,axis=2))
    print(np.mean(meanDiff))

    for i in xrange(nbrCells):
        tmRF[i,i] = np.nan

    corr   = np.ones((nbrCells,nbrPatches))
    corrShuf = np.zeros((nbrCells,nbrPatches))
    for i in xrange(nbrCells):
        for j in xrange(nbrPatches):
            if np.sum(spcMatrix[i,j,:]) > 0.0:
                corr[i,j] = np.corrcoef(spcMatrix[i,j,:],spcMatrix[i,j,:])[0,1]
                arr = np.copy(spcMatrix[i,j,:])
                np.random.shuffle(arr)
                corrShuf[i,j] = np.corrcoef(spcMatrix[i,j,:],arr)[0,1]

    plt.figure()
    plt.hist(np.mean(corr,axis=1))
    plt.savefig('./Output/MeanToVar/Correl_Self_MeanPatch.png')

    plt.figure()
    plt.hist(np.mean(corrShuf,axis=1))
    plt.savefig('./Output/MeanToVar/Correl_SelfShuffle_MeanPatch.png') 

    corrN= np.zeros((nbrCells,nbrPatches))
    corrNShuf=np.zeros((nbrCells,nbrPatches))
    for n1 in xrange(nbrCells):        
        ix = np.where(tmRF[n1] == np.nanmax(tmRF[n1]))
        n2 = ix[0]
        for j in xrange(nbrPatches):
            if np.sum(spcMatrix[n1,j,:]) > 0.0 and np.sum(spcMatrix[n2,j,:]) > 0.0:
                corrN[i,j] = np.corrcoef(spcMatrix[n1,j,:],spcMatrix[n2,j,:])[0,1]
                arr = np.copy(spcMatrix[n2,j,:])
                np.random.shuffle(arr)
                corrNShuf[i,j] = np.corrcoef(spcMatrix[n1,j,:],arr)[0,1]

    plt.figure()
    plt.hist(np.mean(corrN,axis=1))
    plt.savefig('./Output/MeanToVar/Correl_Equal_MeanPatch.png')

    plt.figure()
    plt.hist(np.mean(corrNShuf,axis=1))
    plt.savefig('./Output/MeanToVar/Correl_EqualShuffle_MeanPatch.png') 
#------------------------------------------------------------------------------
def startAnalise():

    if not os.path.exists('./Output/MeanToVar'):
        os.mkdir('./Output/MeanToVar')

    spcMatrix = np.load('./work/meanTOvar_frExc.npy')
    spkLGN = np.load('./work/meanTOvar_frLGN.npy')
    tmRF = np.load('./work/TemplateMatch.npy')
    

    nbrCellsLGN,nbrPatches,nbrSamples = np.shape(spkLGN)
    nbrCells = np.shape(spcMatrix)[0]

    #analyseSpikeCount(spcMatrix,tmRF)

    meanLGN = np.mean(spkLGN,axis=2)
    varLGN = np.var(spkLGN,axis=2)
    stdLGN = np.std(spkLGN,axis=2)
    
    plt.figure()
    plt.hist(np.mean(stdLGN,axis=1))
    plt.savefig('./Output/MeanToVar/LGN_STD.png')

    meanV1 = np.mean(spcMatrix,axis=2)
    varV1 = np.var(spcMatrix,axis=2)
    stdV1 = np.std(spcMatrix,axis=2)

    plt.figure()
    plt.hist(np.mean(stdV1,axis=1))
    plt.savefig('./Output/MeanToVar/V1_STD.png')

    plotDataPoints(meanV1,varV1,meanLGN,varLGN)


#------------------------------------------------------------------------------

if __name__=="__main__":
    startAnalise()

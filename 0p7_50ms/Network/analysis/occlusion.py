import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os.path
import mutualInformation as mi
import sys

#----------------------------------------------------------
def calcCorrelation(origFr,occFr):
    print(np.shape(origFr))
    nbrPatches,occLvL,nbrCells = np.shape(occFr)
    correlMperPatch = np.zeros((nbrPatches,occLvL))
    correlMperCell = np.zeros((nbrCells,occLvL))

    for i in range(nbrCells):
        for j in range(occLvL):
            if ((np.sum(origFr[:,j,i]) > 0) and (np.sum(occFr[:,j,i]) > 0)):
                #print(occFr[i,j])
                #print('----')
                #print(origFr[i,j])
                #print('-----------------------')
                correlMperCell[i,j]=np.corrcoef(origFr[:,j,i],occFr[:,j,i])[0,1]

    for i in range(nbrPatches):
        for j in range(occLvL):
            if ((np.sum(origFr[i,j]) > 0) and (np.sum(occFr[i,j]) > 0)):
                #print(occFr[i,j])
                #print('----')
                #print(origFr[i,j])
                #print('-----------------------')
                correlMperPatch[i,j]=np.corrcoef(origFr[i,j],occFr[i,j])[0,1]
    return(correlMperCell,correlMperPatch)
#----------------------------------------------------------
def plotStats(valueInp,valueM,matter,name):
    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 15)
    plt.rc('ytick',labelsize = 15)
    meanOfInp = np.mean(valueInp,axis=0)

    meanOverCells = np.mean(valueM,axis=0)
    stdA = np.std(valueM,axis=0)
    minA = np.min(valueM,axis=0)
    maxA = np.max(valueM,axis=0)

    plt.figure()
    plt.plot(meanOfInp,'b--o',label='Input',lw=2)
    plt.errorbar(range(0,len(meanOverCells)),meanOverCells,yerr=stdA,fmt='g--+',label='Recon',lw=2)
    plt.legend()
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel(matter,fontsize=17,weight = 'bold')
    plt.xlabel('replaced pixels [%]',fontsize=17,weight = 'bold')
    plt.savefig('./Output/Dissolve/'+name+'/'+matter+'_STD.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(minA,'b--+')
    plt.plot(maxA,'g--+')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel(matter)
    plt.xlabel('percent of replaced pixels',fontsize=20,weight = 'bold')
    plt.savefig('./Output/Dissolve/'+name+'/'+matter+'_MinMax.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(stdA,'b-o')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.savefig('./Output/Dissolve/'+name+'/'+matter+'_STDA.png')

    diff =np.zeros(len(stdA))
    for i in range(1,len(stdA)):
        diff[i] = np.abs( meanOverCells[i] - meanOverCells[i-1])
    plt.figure()
    plt.plot(diff,'-+')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.savefig('./Output/Dissolve/'+name+'/'+matter+'_DeltaMean.png')
    
    plt.close('all')
#----------------------------------------------------------
def startAnalyseImage():
    
    #----- analysis from dissolve with a image patch -----#
    inptOcc = np.load('./work/dissolveImage_Input.npy')
    frExc = np.load('./work/dissolveImage_frExc.npy')
    frLGN = np.load('./work/dissolveImage_frLGN.npy')
    weights = np.loadtxt('./Input_network/V1weight.txt')
    print(np.shape(frExc))
    frExc = np.mean(frExc,axis=4)
    frLGN = np.mean(frLGN,axis=4)

    n_patches,d_steps,patchsize = np.shape(inptOcc)[0:3]

    print('Start analysis of dissolving with an other image patch')
    
    correlEx_pC,correlEx_pP =  calcCorrelation(frExc[:,1,:,:],frExc[:,0,:,:])


    correlLGN_pC,correlLGN_pP = calcCorrelation(frLGN[:,1,:,:],frLGN[:,0,:,:])

    f,ax = plt.subplots(2)
    ax[0].plot(np.mean(correlEx_pC,axis=0),label='model')    
    ax[0].plot(np.mean(correlLGN_pC,axis=0),label='LGN')
    ax[0].set_ylabel('per Cell')
    ax[0].legend()
    ax[1].plot(np.mean(correlEx_pP,axis=0),label='model')    
    ax[1].plot(np.mean(correlLGN_pP,axis=0),label='LGN')
    ax[1].set_ylabel('per Patch')
    ax[1].legend()
    plt.savefig('Output/Dissolve/Image/divTest.png')

    plotStats(correlLGN_pC,correlEx_pC,'correlation_perCell','Image')
    plotStats(correlLGN_pP,correlEx_pP,'correlation_perPatch','Image')

#----------------------------------------------------------
def startDissolveAnalyse(select):

    if not os.path.exists('./Output/Dissolve'):
        os.mkdir('./Output/Dissolve')
    if not os.path.exists('./Output/Dissolve/Image'):
        os.mkdir('./Output/Dissolve/Image')
    
    startAnalyseImage()

#----------------------------------------------------------    
if __name__=="__main__":
    data = (sys.argv[1:])
    select = 0
    if len(data) > 0:
        select = float(data[0])
    startDissolveAnalyse(select)

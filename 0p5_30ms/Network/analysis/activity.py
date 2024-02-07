import numpy as np
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import sparseness as sps
import os
# main function - startIREAnalysis() :
# different functions to analyse the results from the calculation of the image reconstruction error
# and to compare with the original input image.

# Include: Image Reconstruction and IR Error via NMSE after Spartling(?!)
#          Frequency analysis of reconstructed image and comparision with original Image
#          Statistic analysis like correlation coefficient
#          Sparseness of network 
           
# Inputs find in ./Input_IRE/ and ./work/ . 
# See for further information comments at the functions.
# Note! Not every function has a comment! Read the function names.

# R. Larisch, Technische Universitaet Chemnitz
# 2015-06-23

#---------------- plotting functions ----------------#
def plotPercentsNeuronsPerInput(spks0,spks20,spks40,numberOfNeurons):
    nbrOfPatches = float(np.size(spks0))
    #print(nbrOfPatches)
    percInp = spks0/numberOfNeurons * 100 #percents of Neurons they fired per Inputpatch



    percInpitRange = np.zeros(10+2)
    nbrInpt =  np.size(np.where(percInp==0))
    percInpitRange[0] =nbrInpt/nbrOfPatches * 100
    for i in range(9):
        nbrInpt = np.size(np.where((percInp >=(1.0+(10.0*i))) & (percInp <(10.0+(10.0*i)))))#get numbers of Inputs where between 80 to 90 percents of Neurons are fired
        percInpitRange[i+1] = nbrInpt/nbrOfPatches * 100
    
    nbrInpt = np.size(np.where((percInp >=91) & (percInp <100 ) ) )
    percInpitRange[10] =nbrInpt/nbrOfPatches * 100
    nbrInpt = np.size(np.where(percInp ==100))
    percInpitRange[11] =nbrInpt/nbrOfPatches * 100

    #percentLabel = ('0','1-10','11-20','21-30','31-40',
    #                '41-50','51-60','61-70',
    #                '71-80','81-90','91-99','100')

    percentLabel = ('0','1 - 10','','21 - 30','',
                    '41 - 50','','61 - 70',
                    '','81 - 90','','100')


    plt.figure(figsize=(16,10))
    plt.bar(range(len(percInpitRange)),percInpitRange,align='center',label=percentLabel,width=0.75)
    plt.xticks(np.arange(len(percInpitRange)),percentLabel)
    plt.ylabel('% of Input',fontsize=30,fontweight='bold')
    plt.xlabel('% of Neurons',fontsize=30,fontweight='bold')
    plt.ylim(ymin=0.0,ymax=35.0)
    plt.savefig('./Output/Activ/percentHist0.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotFirerates(fr,desc):

    fig = plt.figure()
    plt.hist(np.mean(fr,axis=0),15)
    plt.xlabel('mean firing rate',fontsize=18)
    plt.ylabel('# Inputs',fontsize=18)
    fig.savefig('./Output/Activ/Firerates/meanFRhistPatches_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    fig = plt.figure()
    plt.hist(np.mean(fr,axis=1),15)
    plt.xlabel('mean firing rate',fontsize=18)
    plt.ylabel('# Of Neurons',fontsize=18)
    fig.savefig('./Output/Activ/Firerates/meanFRhistNeurons_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
    
    
    plt.figure()
    plt.plot(fr[0,:])
    plt.xlabel('Patch Index')
    plt.ylabel('firing rate [Hz]')
    plt.savefig('./Output/Activ/Firerates/FR_Single_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.xlabel('Patch Index')
    plt.ylabel('max firing rate [Hz]')
    plt.plot(np.max(fr,axis=0))
    plt.savefig('./Output/Activ/Firerates/maxFr_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    
    fr = fr *125.0/1000.0 # switch from fr in Hz to counted Spikes
    frPerPatch =np.mean(fr,axis=0) #np.sum(fr,axis=0)
    #zeroIDX = np.where(frPerPatch > 0)
    #frPerPatch = frPerPatch[zeroIDX]
    idxSort = np.argsort(frPerPatch)
    plt.figure()
    plt.plot(frPerPatch[idxSort])
    plt.savefig('./Output/Activ/Firerates/SpikesPerInput_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
#--------------------------- calculating functions ---------------------------#
def calcSpk_actMaps(numberOfNeurons,nbrOfPatchesIRE,frEx):
    #spks, number of neurons they fired per Input
    #actMap, shows, which neuron fired on which Input
    spks0 = np.zeros(nbrOfPatchesIRE) 
    actMap0 = np.zeros((numberOfNeurons,nbrOfPatchesIRE))
    spks20 = np.zeros(nbrOfPatchesIRE)
    actMap20 = np.zeros((numberOfNeurons,nbrOfPatchesIRE))
    spks40 = np.zeros(nbrOfPatchesIRE)
    actMap40 = np.zeros((numberOfNeurons,nbrOfPatchesIRE))
    
    for i in range(numberOfNeurons):
        indexFR0 = np.where(frEx[i] >0.0)
        spks0[indexFR0] = spks0[indexFR0] +1.0
        actMap0[i,indexFR0] = 1.0      
        indexFR20 = np.where(frEx[i] >20.0)
        spks20[indexFR20] = spks20[indexFR20] +1.0
        actMap20[i,indexFR20] = 1.0      
        indexFR40 = np.where(frEx[i] >40.0)
        spks40[indexFR40] = spks40[indexFR40] +1.0
        actMap40[i,indexFR40] = 1.0      
    return(spks0,spks20,spks40,actMap0,actMap20,actMap40)
#------------------------------------------------------------------------------
def calcCorrExcInh(frActiv,frAInhib):
    nbrExc = np.shape(frActiv)[0]
    nbrInh = np.shape(frAInhib)[0]
    corrMatrix = np.zeros((nbrExc,nbrInh))
    for i in range(nbrExc):
        excFr = frActiv[i,:]
        for j in range(nbrInh):
            inhFr = frAInhib[j,:]
            corrMatrix[i,j] =  np.corrcoef(excFr,inhFr)[0,1]
    return(corrMatrix)
#------------------------------------------------------------------------------
def plotCorr(corrExcInh):
    nbrOfExc,nbrOfInhib = np.shape(corrExcInh)
    
    plt.figure()
    plt.imshow(corrExcInh.T)
    plt.xlabel('index excitatory neuron')
    plt.ylabel('index inhibitory neuron')
    plt.savefig('Output/Activ/CorrelationExc_Inh.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def startAnalysis():
    print('Start analysis network activity')
    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 28)
    plt.rc('ytick',labelsize = 28)

    if not os.path.exists('Output/Activ/Firerates/'):
        os.mkdir('Output/Activ/Firerates/')

    #----load necessary datas----# 
    frActiv = np.load('work/Active_fr.npy')
    frAInhib = np.load('work/Active_frInhib.npy')
    frALGN = np.load('work/Active_frLGN.npy')

    corrExcInh = calcCorrExcInh(frActiv,frAInhib)
    plotCorr(corrExcInh)
    numberOfNeuronsActiv,nbrOfPatchesActiv=np.shape(frActiv)
    
    print('Calculate Spike Activity')
    spks0,spks20,spks40,actMap0,actMap20,actMap40 = calcSpk_actMaps(numberOfNeuronsActiv,nbrOfPatchesActiv,frActiv)
    
    print('Plot Datas')
    print('Percent Activation')
    plotPercentsNeuronsPerInput(spks0,spks20,spks40,numberOfNeuronsActiv)
    print('Plot firing rates')    
    plotFirerates(frActiv,'Exc')
    plotFirerates(frAInhib,'Inhib')
    plotFirerates(frALGN,'LGN')
    print('finish with Analysis the network Activity')
#-----------------------------------------------------
if __name__=="__main__":
    startAnalysis()

import numpy as np
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import os.path
import os
# calculatet sparseness of three different Ways (see Spartling,2012)


#Sparseness based on Kurtosis(see Field,1994) 
#values > 0 for sparse distributions
def calculateKurtosisSparseness(frEx):
    nbrOfNeurons,nbrOfPatches = np.shape(frEx)
    normFrEx = frEx/np.max(frEx)

    sparsenessOverCells = []
    sparsenessOverInput = []
    
    for i in range(nbrOfNeurons):
        frNeuron = normFrEx[i,:]
        meanFr = np.mean(frNeuron) # mean Firerate
        stdFr = np.std(frNeuron) # standart deviation of Firerates
        if(stdFr !=0.0):
            k = np.sum((frNeuron-meanFr)**4/(stdFr**4))
            sparsenessOverCells.append(1/np.float(nbrOfPatches)*(k)-3)
    
    for i in range(nbrOfPatches):
        frPatch = normFrEx[:,i]
        meanFr = np.mean(frPatch) # mean Firerate
        stdFr = np.std(frPatch) # standart deviation of Firerates
        if(stdFr != 0.0): #standart deviation can be zero, if no one spiked on these Input
            k = np.sum((frPatch-meanFr)**4/(stdFr**4))
            sparsenessOverInput.append(1/np.float(nbrOfNeurons)*(k)-3)

    return(sparsenessOverCells,sparsenessOverInput)


# Sparseness based on Rolls-Tovee measure (Rolls and Tovee,1995)
# by Vinje and Gallant(Vinje and Gallant,2000)
# 0< value <1 , higher than sparser distributions
def calculateVinjeGallantSparseness(frEx):
    nbrOfNeurons,nbrOfPatches = np.shape(frEx)
    normFrEx = frEx/np.max(frEx)

    sparsenessOverCells = []
    sparsenessOverInput = []
    
    for i in range(nbrOfNeurons):
        frNeuron = normFrEx[i,:]
        s1 = np.sum(frNeuron/np.float(nbrOfPatches))**2
        s2 = np.sum(frNeuron**2)/np.float(nbrOfPatches)
        if s2 !=0.0:
            d = 1 - (1/np.float(nbrOfPatches))
            sparsenessOverCells.append((1-(s1/s2))/d)
    
    for i in range(nbrOfPatches):
        frPatch = normFrEx[:,i]
        s1 = np.sum(frPatch/np.float(nbrOfNeurons))**2
        s2 = np.sum(frPatch**2)/np.float(nbrOfNeurons)
        if s2 !=0.0:
            d = 1 - (1/np.float(nbrOfNeurons))
            sparsenessOverInput.append((1-(s1/s2))/d)
    return(sparsenessOverCells,sparsenessOverInput)


# Sparseness based on L1 and L2 Norm (see Hoyer,2004)
# 0< value <1 , higher than sparser distributions 
def calculateHoyerSparseness(frEx):
    nbrOfNeurons,nbrOfPatches = np.shape(frEx)
    normFrEx = frEx/np.max(frEx)

    sparsenessOverCells = []
    sparsenessOverInput = []
    
    for i in range(nbrOfNeurons):
        frNeuron = normFrEx[i,:]
        d1 = np.sqrt(np.sum(frNeuron**2))
        if d1 !=0.0:
            d2 = np.sqrt(nbrOfPatches) - (np.sum(frNeuron)/d1)
            sparsenessOverCells.append(d2/(np.sqrt(nbrOfPatches)-1.0))

    for i in range(nbrOfPatches):
        frPatch = normFrEx[:,i]
        d1 = np.sqrt(np.sum(frPatch**2))
        if d1 !=0.0:
            d2 = np.sqrt(nbrOfNeurons) - (np.sum(frPatch)/d1)
            sparsenessOverInput.append(d2/(np.sqrt(nbrOfNeurons)-1.0))
    return(sparsenessOverCells,sparsenessOverInput)

def calculateAndPlotKurtosis(frEx):
    print('Sparseness over Kurtosis')
    sparsOverCellKurt,sparsOverInputKurt = calculateKurtosisSparseness(frEx)
    plt.figure()
    plt.hist(sparsOverCellKurt,15)
    #plt.xlim(xmin=0.2,xmax=0.5)
    plt.title('mean: '+str(np.round(np.mean(sparsOverCellKurt),4)))
    plt.xlabel('Sparseness over Kurtosis',fontsize=13)
    plt.ylabel('# of Neurons',fontsize=13)
    plt.savefig('./Output/Activ/Sparseness/sparsKurtosis_histCells.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(sparsOverInputKurt,15)
    #plt.xlim(xmin=0.0,xmax=1.0)
    plt.title('mean: '+str(np.round(np.mean(sparsOverInputKurt),4)))
    plt.xlabel('Sparseness over Kurtosis',fontsize=13)
    plt.ylabel('# of Inputs',fontsize=13)
    plt.savefig('./Output/Activ/Sparseness/sparsKurtosis_histInput.png',bbox_inches='tight', pad_inches = 0.1)

def calculateAndPlotVinjeGallant(frEx):
    print('Sparseness at Vinje and Gallant')
    sparsOverCellVG,sparsOverInputVG = calculateVinjeGallantSparseness(frEx)
    plt.figure()
    plt.hist(sparsOverCellVG,15)
    plt.title('mean: '+str(np.round(np.mean(sparsOverCellVG),4)))
    plt.xlabel('Lifetime Sparseness',fontsize=18)
    plt.ylabel('# of Neurons',fontsize=18)
    plt.savefig('./Output/Activ/Sparseness/sparsVinjeGallant_histCells.png',bbox_inches='tight', pad_inches = 0.1)

    stepSize = 0.9/40.0
    plt.figure()
    plt.hist(sparsOverCellVG,bins = np.arange(0.35,0.9+stepSize,stepSize))
    #plt.xlim(xmin=0.35,xmax=0.9)
    #plt.ylim(ymin=0,ymax=140)
    plt.xlabel('Lifetime Sparseness',fontsize=18)
    plt.ylabel('# of Neurons',fontsize=18)
    plt.savefig('./Output/Activ/Sparseness/sparsVinjeGallant_histCells_bounds.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(sparsOverInputVG,15)
    plt.xlim(xmin=0.0,xmax=1.0)
    plt.title('mean: '+str(np.round(np.mean(sparsOverInputVG),4)))
    plt.xlabel('Population Sparseness',fontsize=18)
    plt.ylabel('# of Inputs',fontsize=18)
    plt.savefig('./Output/Activ/Sparseness/sparsVinjeGallant_histInput.png',bbox_inches='tight', pad_inches = 0.1)

    np.save('./work/V&G_LifeTime',sparsOverCellVG)
    np.save('./work/V&G_Population',sparsOverInputVG)

def calculateAndPlotHoyer(frEx):
    print('Sparseness after Hoyer')
    sparsOverCellHoyer,sparsOverInputHoyer = calculateHoyerSparseness(frEx)
    plt.figure()
    plt.hist(sparsOverCellHoyer,15)
    plt.title('mean: '+str(np.round(np.mean(sparsOverCellHoyer),4)))
    plt.xlabel('Lifetime Sparseness',fontsize=23,fontweight='bold')
    plt.ylabel('# of Neurons',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Sparseness/sparsHoyer_histCells.png',bbox_inches='tight', pad_inches = 0.1)

    stepSize = 0.6/40.0
    plt.figure()
    plt.hist(sparsOverCellHoyer,bins = np.arange(0.2,0.6+stepSize,stepSize))
    plt.xlim(xmin=0.2,xmax=0.55)
    plt.ylim(ymin=0.0,ymax=120)
    plt.xlabel('Lifetime Sparseness',fontsize=23,fontweight='bold')
    plt.ylabel('# of Neurons',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Sparseness/sparsHoyer_histCells_bounds.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(sparsOverInputHoyer,15)
    #plt.xlim(xmin=0.0,xmax=1.0)
    #plt.ylim(ymin=0.0,ymax=60000)
    plt.title('mean: '+str(np.round(np.mean(sparsOverInputHoyer),4)))
    plt.xlabel('Population Sparseness',fontsize=23,fontweight='bold')
    plt.ylabel('# of Inputs',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Sparseness/sparsHoyer_histInput.png',bbox_inches='tight', pad_inches = 0.1)

    np.save('./work/Hoyer_LifeTime',sparsOverCellHoyer)
    np.save('./work/Hoyer_Population',sparsOverInputHoyer)

def calculateAndPlotSparseness():
    print('Start to calculate and Plot the Sparseness')
    if not os.path.exists('Output/Activ/Sparseness/'):
        os.mkdir('Output/Activ/Sparseness/')
    frEx = np.load('work/Active_fr.npy')
    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 18)
    plt.rc('ytick',labelsize = 18)

    #calculateAndPlotKurtosis(frEx)
    calculateAndPlotVinjeGallant(frEx)
    calculateAndPlotHoyer(frEx)

    print('Finish with Sparseness!')

if __name__=="__main__":
    calculateAndPlotSparseness()

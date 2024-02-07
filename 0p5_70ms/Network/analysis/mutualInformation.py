import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import sys
import os

def shanEntropyBit(c):
    cNormalized = c/np.float(np.sum(c))
    cNormalized = cNormalized[np.nonzero(cNormalized)]
    H = -np.sum(cNormalized* np.log2(cNormalized))  
    return(H)


# new shannon entropy, inspired by matlab
def shanEntropyNat(c):
    cNormalized = c/np.float(np.sum(c))
    cNormalized = cNormalized[np.nonzero(cNormalized)]
    H =- np.sum(cNormalized * np.log(cNormalized))
    return(H)
#------------------------------------------------------------------------------
def calc_MI_Hist(X,Y,bins,norm=False):
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    c_XY = np.histogram2d(X,Y,bins)[0]

    H_X = shanEntropyNat(c_X)
    H_Y = shanEntropyNat(c_Y)
    H_XY = shanEntropyNat(c_XY)

    
    if (H_X == 0.0 and H_Y == 0.0):
        return(0.0)
    if (norm):
        MI = (H_X + H_Y - H_XY)/(H_X + H_Y)
        return(MI)
    else:
        MI = (H_X + H_Y - H_XY)
        return(MI)
#------------------------------------------------------------------------------
def reconstructInputPatches(ffWeights,fRperPatch):
    nbrOfNeurons,nbrOfInputs = np.shape(fRperPatch)
    patchsize = int(np.sqrt(nbrOfNeurons))
    inputPatches = np.zeros((patchsize,patchsize,nbrOfInputs))
    for patchNbr in range(nbrOfInputs):
        neuronActivity = np.sum([fRperPatch[n,patchNbr] * ffWeights[n,:] for n in range(nbrOfNeurons)],axis=0)
        inpPatch = np.reshape(neuronActivity,(patchsize,patchsize,2))
        inputPatches[:,:,patchNbr] = inpPatch[:,:,0] - inpPatch[:,:,1]
        if (patchNbr%5000 ==0):
            sys.stdout.write('.')
            sys.stdout.flush()
    np.save('./work/reconstr_InptPatches.npy',inputPatches)
    return(inputPatches)
#------------------------------------------------------------------------------
def calculateMI(inputPatch,recPatch,size,ifNorm):
    miPatch = np.zeros((size,size))
    for w in range(size):
        for h in range(size):
            inputPixel = inputPatch[w,h,:]
            recPixel = recPatch[w,h,:]
            miPatch[w,h] = calc_MI_Hist(inputPixel,recPixel,10,ifNorm) 
    return(miPatch)
#------------------------------------------------------------------------------
def calculateMIPatch(inputPatch,recPatch,size,ifNorm):
    nbrPatches = np.shape(inputPatch)[2]
    mi = np.zeros(nbrPatches)
    for i in range(nbrPatches):
        inputArray = np.reshape(inputPatch[:,:,i],size*size)
        recArray = np.reshape(recPatch[:,:,i],size*size)
        mi[i] = calc_MI_Hist(inputArray,recArray,10,ifNorm) 
    return(mi)

#------------------------------------------------------------------------------
def plotmIImage(miPatch):
    plt.figure()
    plt.imshow(miPatch,cmap=plt.get_cmap('gray'),interpolation='none')
    plt.colorbar()
    plt.savefig('./Output/Activ/Stats/miPatch.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotmIHist(miPatch,mi):
    w,h = np.shape(miPatch)
    miArray = np.reshape(miPatch,w*h)
    plt.figure()
    plt.hist(miArray,13)
    #plt.xlim(xmin=0.0,xmax=0.3)
    plt.xlabel('mutual Information',fontsize=23,fontweight='bold')
    plt.ylabel('# of Pixels',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Stats/MIover200kNatSc_Hist.png',bbox_inches='tight', pad_inches = 0.1)
    np.save('./work/MIover200kNatSc',miPatch)

    plt.figure()
    plt.hist(mi)
    plt.savefig('./Output/Activ/Stats/MI_PerPatch_Hist.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def calcTemplateMatch(X,Y):
    tm = 0
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if (normX != 0 and normY !=0):
        tm = (np.dot(X,Y) / (normX*normY))
    return(tm)
#------------------------------------------------------------------------------
def calculateTM(inputPatch,recPatch):
    w,h,nbrPatches = np.shape(inputPatch)
    tmPerPatch = np.zeros(nbrPatches)
    tmPerPixel = np.zeros((w,h))
    for i in range(nbrPatches):
        inputA = np.reshape(inputPatch[:,:,i],(w*h))
        recA = np.reshape(recPatch[:,:,i],(w*h))
        tmPerPatch[i] = calcTemplateMatch(inputA,recA)
    for x in range(w):
        for y in range(h):
            tmPerPixel[x,y] = calcTemplateMatch(inputPatch[x,y,:],recPatch[x,y,:])
    return(tmPerPatch,tmPerPixel)
#------------------------------------------------------------------------------
def plotTM(tmPerPatch,tmPerPixel):
    w,h = np.shape(tmPerPixel)

    plt.figure()
    plt.imshow(tmPerPixel,interpolation='none',cmap=plt.get_cmap('gray'))
    plt.savefig('./Output/Activ/Stats/TMPerPixel_IMG.png',bbox_inches='tight', pad_inches = 0.1)

    tmArray = np.reshape(tmPerPixel,w*h)
    plt.figure()
    plt.hist(tmArray,13)
    plt.xlabel('template match',fontsize=23,fontweight='bold')
    plt.ylabel('# of Pixels',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Stats/TMPerPixel_Hist.png',bbox_inches='tight', pad_inches = 0.1)
    
    plt.figure()
    plt.hist(tmPerPatch,13)
    plt.xlabel('template match',fontsize=23,fontweight='bold')
    plt.ylabel('# of Patches',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Stats/TMPerPatch_Hist.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def calculateCorrel(inputPatch,recPatch):
    w,h,nbrPatches = np.shape(inputPatch)
    corrPerPatch = np.zeros(nbrPatches)
    corrPerPixel = np.zeros((w,h))
    for i in range(nbrPatches):
        inputA = np.reshape(inputPatch[:,:,i],w*h)
        recA = np.reshape(recPatch[:,:,i],w*h)
        if np.mean(recA) != 0:
            corrPerPatch[i] = np.corrcoef(inputA,recA)[0,1]
        else: 
            corrPerPatch[i] = 0.0
    for x in range(w):
        for y in range(h):
              corrPerPixel[x,y] = np.corrcoef(inputPatch[x,y,:],recPatch[x,y,:])[0,1]
    return(corrPerPatch,corrPerPixel)
#------------------------------------------------------------------------------
def plotCorr(corrPerPatch,corrPerPixel):
    w,h = np.shape(corrPerPixel)

    plt.figure()
    plt.imshow(corrPerPixel,interpolation='none',cmap=plt.get_cmap('gray'))
    plt.savefig('./Output/Activ/Stats/CorrPerPixel_IMG.png',bbox_inches='tight', pad_inches = 0.1)

    corrArray = np.reshape(corrPerPixel,w*h)
    plt.figure()
    plt.hist(corrArray,13)
    plt.xlabel('correlation',fontsize=23,fontweight='bold')
    plt.ylabel('# of Pixels',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Stats/CorrPerPixel_Hist.png',bbox_inches='tight', pad_inches = 0.1)
    
    plt.figure()
    plt.hist(corrPerPatch,13)
    plt.xlabel('correlation',fontsize=23,fontweight='bold')
    plt.ylabel('# of Patches',fontsize=23,fontweight='bold')
    plt.savefig('./Output/Activ/Stats/CorrPerPatch_Hist.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def calc_STA(inputPatch,fRperPatch):
    w,h,d,nbrPatches = np.shape(inputPatch)
    nbrCells = np.shape(fRperPatch)[0]
    sta = np.zeros((nbrCells,w,h,d))
    for i in range(nbrPatches):
        activCells = np.asarray(np.where(fRperPatch[:,i] > 0.0))
        for c in (activCells[0]):
            sta[c] += inputPatch[:,:,:,i]*fRperPatch[c,i]
    for i in range(nbrCells):
        nbrFiredPatches = np.where(fRperPatch[i,:] > 0.0)
        sta[i] =  sta[i] / len(nbrFiredPatches[0])
    return(sta)
#------------------------------------------------------------------------------
def startMutualInformation():
    print('Start to calculate the mutual information')

    if not os.path.exists('./Output/Activ/Stats'):
        os.mkdir('./Output/Activ/Stats')

    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 13)
    plt.rc('ytick',labelsize = 13)
    inputPatch = np.load('./work/Active_InptPatches.npy')
    #inputPatch = np.load('./work/IRE_singlePatches.npy')
    h,w,d,nbr = (np.shape((inputPatch)))
    #inputPatch = np.reshape(inputPatch, (h,w,d,nbr)) 
    fRperPatch = np.load('./work/Active_fr.npy')
    sta = calc_STA(inputPatch,fRperPatch)# as a mean activity per pixel per neuron
    sta = sta[:,:,:,0] - sta[:,:,:,1]
    #fRperPatch = np.load('./work/IRE_fr.npy')
    ffW = np.loadtxt('./Input_network/V1weight.txt')
    if os.path.exists('./work/reconstr_InptPatches.npy') !=  True:
        print('reconstruct single patches')
        recPatch = reconstructInputPatches(ffW,fRperPatch) # only onetime to calculate the single input Patches
    else:
        print('load reconstructed patches')
        recPatch = np.load('./work/reconstr_InptPatches.npy')
    recPatch = recPatch/np.max(np.abs(recPatch))
    inputPatch = inputPatch[:,:,0,:] - inputPatch[:,:,1,:]
    inputPatch = inputPatch/np.max(np.abs(inputPatch))
    #print(np.shape(recPatch))
    #print(np.shape(inputPatch))
    miPatch = calculateMI(inputPatch,recPatch,w,False)# MI per pixel !
    mi = calculateMIPatch(inputPatch,recPatch,w,False)

    meanFR = np.mean(fRperPatch,axis=1)
    print(np.shape(meanFR))
    print(np.shape(miPatch))
    sumFR = np.sum(fRperPatch,axis=1)
    sumMI = np.sum(miPatch)
    meanMI = np.mean(miPatch)
    print(np.shape(meanMI))
    meanMIPerSpk = meanMI / meanFR
    sumMIPerSpk = sumMI / sumFR

    meanMIperPxl = np.zeros(np.shape(sta)[0])
    for i in range(np.shape(sta)[0]):
        meanMIperPxl[i] = np.sum(miPatch)/np.sum(sta[i])

    tmPerPatch,tmPerPixel = calculateTM(inputPatch,recPatch)
    corrPerPatch,corrPerPixel = calculateCorrel(inputPatch,recPatch)
    print(np.min(corrPerPatch))
    print(np.mean(corrPerPixel))
    #---- plots ----#
    plt.figure()
    plt.title('mean: '+str(np.mean(meanMIPerSpk)))
    plt.hist(meanMIPerSpk)
    plt.xlabel('mutual Information '+r'$ \frac{\overline{mI_i}}{\overline{FR}}  $')
    #plt.xlim(4.6,11.5)
    #plt.ylim(0.0,35)
    plt.savefig('./Output/Activ/Stats/MeanMIperSpike.png',bbox_inches='tight', pad_inches = 0.1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('mean: '+str(np.mean(sumMIPerSpk)))
    plt.hist(sumMIPerSpk)
    plt.xlabel('mutual Information '+r'$ \frac{\sum{mI_i}}{\sum{FR}}  $')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #plt.xlim(4.6,11.5)
    #plt.ylim(0.0,35)
    plt.savefig('./Output/Activ/Stats/SumMIperSpike.png',bbox_inches='tight', pad_inches = 0.1)
    

    plt.figure()
    plt.hist(meanMIperPxl)
    plt.savefig('./Output/Activ/Stats/MeanMIperPixel.png',bbox_inches='tight', pad_inches = 0.1)

    plotmIImage(miPatch)
    plotmIHist(miPatch,mi)
    plotTM(tmPerPatch,tmPerPixel)
    plotCorr(corrPerPatch,corrPerPixel)
    print("finish with mutual Information")
#-----------------------------------------------------
if __name__=="__main__":
    startMutualInformation()
    

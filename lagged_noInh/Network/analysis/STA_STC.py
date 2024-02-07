import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

# R. Larisch, Technische Universitaet Chemnitz
# 2015-06-23
# ueberarbeiten, STA und STC dynamisch fuer jede moegliche Neuronenpopulation!

#------------------------------------------------------------------------------
def setSubplotDimension(a):
    x = 0
    y = 0
    if ( (a % 1) == 0.0):
        x = a
        y =a
    elif((a % 1) < 0.5):
        x = round(a) +1
        y = round(a)
    else :
        x = round(a)
        y = round(a)
    return (x,y)   
#------------------------------------------------------------------------------
def calcSTA(Input,frEx,frInh):
    # calculate based on the simulation results the Spike - Triggered - Average of exitatory and inhibitory Neurons
    # Input -> matrix with all input patches from the simulation, shape = (number of input patches, size of one patch )
    # frEx -> fire rate of the exitatory neurons, for every input patch, shape=(number of Neurons, number of input patches)
    # frInh -> fire rate of the inhibitory neurons, for every input patch, shape=(number of Neurons, number of input patches)
    print('calculate STA')
    Input = Input/np.max(Input)
    numberOfNeurons = np.shape(frEx)[0]
    patchsize = np.shape(Input)[1]
    STAV1 = np.zeros((numberOfNeurons,patchsize,patchsize,2))
    STAIN = np.zeros((numberOfNeurons/4,patchsize,patchsize,2)) 
    for i in range(numberOfNeurons):
        spInp = np.squeeze(Input[np.nonzero(frEx[i,:]),:,:,:])
        spk = np.squeeze(frEx[i,np.nonzero(frEx[i,:])])
        for j in range(np.shape(spInp)[0]):
            STAV1[i,:,:,:] = STAV1[i,:,:,:] + (spInp[j] * spk[j])
        STAV1[i,:,:,:] = STAV1[i,:,:,:] / np.sum(spk)
        if i < (numberOfNeurons/4):
            spInp = Input[np.nonzero(frInh[i,:])]
            spk = np.squeeze(frInh[i,np.nonzero(frInh[i,:])])
            for j in range(np.shape(spInp)[0]):
                STAIN[i,:,:,:] = STAIN[i,:,:,:] + (spInp[j] * spk[j])
            STAIN[i,:,:,:] = STAIN[i,:,:,:]/np.sum(spk)
    return(STAV1,STAIN)
#------------------------------------------------------------------------------
def calcSTA2(inpt,frExc,frInh):
    print('calculate STA')
    inpt = inpt/np.max(inpt)
    nbrOfNeurons = np.shape(frExc)[0]
    print('Mean activity:',np.mean(frExc))
    patchsize = np.shape(inpt)[1]
    STAV1 = np.zeros((nbrOfNeurons,2,patchsize,patchsize))
    STAIN = np.zeros((int(nbrOfNeurons/4),2,patchsize,patchsize)) 
    for i in range(nbrOfNeurons):
        spk = np.squeeze(frExc[i,np.nonzero(frExc[i,:])])
        actInp = inpt[np.nonzero(frExc[i,:])]
        sta= actInp.T*spk
        sta= np.sum(sta,axis=3)
        sta /= np.sum(spk)
        STAV1[i] = sta
        if i < (int(nbrOfNeurons/4)):
            spk = np.squeeze(frInh[i,np.nonzero(frInh[i,:])])   
            actInp = inpt[np.nonzero(frInh[i,:])]
            sta= actInp.T*spk
            sta= np.sum(sta,axis=3)
            sta /= np.sum(spk)
            STAIN[i] = sta
            
    return(STAV1,STAIN)
#-----------------------------------------------------------------------
def calcSTC(Input,frEx,frInh):
    # calculate based on the simulation results the Spike - Triggered - Covariance of exitatory and inhibitory Neurons
    # Input -> matrix with all input patches from the simulation, shape = (number of input patches, size of one patch )
    # frEx -> fire rate of the exitatory neurons, for every input patch, shape=(number of Neurons, number of input patches)
    # frInh -> fire rate of the inhibitory neurons, for every input patch, shape=(number of Neurons, number of input patches)
    # after Sharpee et al. (2013) (doi:10.1146/annurev-neuro-062012-170253)
    print('calculate STC')
    numberOfNeurons = np.shape(frEx)[0]
    patchsize = np.shape(Input)[1]
    STCV1 = np.zeros((numberOfNeurons,patchsize*patchsize*2,patchsize*patchsize*2))  
    eigVals = np.zeros((numberOfNeurons,patchsize*patchsize*2))
    eigVects=np.zeros((numberOfNeurons,patchsize*patchsize*2,patchsize*patchsize*2))
    sumInp = np.sum(Input,axis=0)/np.shape(Input)[0]
    sumInp = np.reshape(sumInp ,patchsize*patchsize*2)
    sumInp = np.matrix([sumInp])
    covInp = np.dot(sumInp.T,sumInp)
    for i in range(numberOfNeurons):
        spInp = np.squeeze(Input[np.nonzero(frEx[i,:]),:,:,:])
        spk = np.squeeze(frEx[i,np.nonzero(frEx[i,:])])   
        for j in range(np.shape(spInp)[0]):
            spInp[j,:,:,:] = spInp[j,:,:,:]#* spk[j]
        sumSpkInp = np.sum(spInp,axis=0)/np.shape(Input)[0]
        sumSpkInp = np.reshape(sumSpkInp,patchsize*patchsize*2)
        sumSpkInp = np.matrix([sumSpkInp])
        covSpkInp = np.dot(sumSpkInp.T,sumSpkInp)# /np.sum(spk)

        STCV1[i,:,:] = covInp - covSpkInp

        eigVal,eigVect= np.linalg.eig(STCV1[i,:,:])
        eigVals[i] = np.squeeze(eigVal)
        eigVects[i] = np.squeeze(eigVect)

        sys.stdout.write('.')
        sys.stdout.flush()
    print('')
    #print(np.shape(eigVects))
    return (eigVals,eigVects)
#-----------------------------------------------------------------------------
def plotSTA(STAV1,STAIN):
    print('plot STA')
    fig = plt.figure(figsize=(8,8))
    x,y = setSubplotDimension(np.sqrt(np.shape(STAV1)[0]))
    for i in range(np.shape(STAV1)[0]):
        field = (STAV1[i,0,:,:] - STAV1[i,1,:,:])
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(field.T,cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STA_STC/STAEX_.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    nbrCells = 64
    fig = plt.figure(figsize=(8,8))
    x,y = setSubplotDimension(np.sqrt(nbrCells))
    for i in range(nbrCells):
        field = (STAV1[i,0,:,:] - STAV1[i,1,:,:])
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(field.T,cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STA_STC/STAEX_'+str(nbrCells)+'.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    nbrCells = 49
    fig = plt.figure(figsize=(8,8))
    x,y = setSubplotDimension(np.sqrt(nbrCells))
    for i in range(nbrCells):
        field = (STAV1[i,0,:,:] - STAV1[i,1,:,:])
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(field.T,cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STA_STC/STAEX_'+str(nbrCells)+'.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)


    fig = plt.figure(figsize=(8,8))
    x,y = setSubplotDimension(np.sqrt(np.shape(STAIN)[0]))
    for i in range(np.shape(STAIN)[0]):
        field = (STAIN[i,0,:,:] - STAIN[i,1,:,:])
        plt.subplot(x,y,i+1)
        im = plt.imshow(field.T,cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
        mp.pyplot.axis('off')
        #mp.pyplot.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STA_STC/STAIN_.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)
#-----------------------------------------------------------------------------
def plotSTCeigVect2(eigVals,eigVects):
    print('plot eigVect of STC')

    fig =plt.figure()
    
    x,y = setSubplotDimension(np.sqrt(np.shape(eigVals)[0]))
    print('plotEigVal1')
    for i in range(np.shape(eigVals)[0]):
        vects = np.squeeze(eigVects[i,:,np.where((eigVals[i,:])>1)])
        img = np.reshape(vects,(12,12,2))
        #print(img)
        #print('------------')
        mp.pyplot.subplot(x,y,i+1)
        mp.pyplot.axis('off')
        im = plt.imshow(img[:,:,0] - img[:,:,1],cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')
        plt.axis('equal')
    plt.axis('equal')
    fig.savefig('./Output/STA_STC/STC_1.png',bbox_inches='tight', pad_inches = 0.1)

    fig2 =plt.figure()
    print('plotEigVal2')
    for i in range(np.shape(eigVals)[0]):
        vects = np.squeeze(eigVects[i,:,np.where((eigVals[i,:])< -1)])
        #print(eigVects[i,:,np.where((eigVals[i,:])>1)] - eigVects[i,:,np.where((eigVals[i,:])< -1)])
        img = np.reshape(vects,(12,12,2))
        #print(img)
        #print('------------')
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(img[:,:,0] - img[:,:,1],cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')    
        plt.axis('equal')
    plt.axis('equal')
    fig2.savefig('./Output/STA_STC/STC_2.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------

def calculateSTAandSTC():
    print('Start to estimate STA and STC')
    Input = np.load('./work/STA_Input.npy')
    frEx  = np.load('./work/STA_frExc.npy')
    frInh = np.load('./work/STA_frInh.npy')
    STAV1,STAIN = calcSTA2(Input,frEx,frInh)
    
    #---plotSTA----#
    plotSTA(STAV1,STAIN)

    eigVals,eigVects = calcSTC(Input,frEx,frInh)
    #---plotSTC---#
    #try:
    plotSTCeigVect2(eigVals,eigVects)
    #except:
    #    print('Error in ploting eigenvectors')

    print("finish with STA and STC")
#------------------------------------------------------------------------------
if __name__=="__main__":
    calculateSTAandSTC()

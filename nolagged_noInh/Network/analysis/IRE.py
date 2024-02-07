import numpy as np
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import json
import scipy.io as sio
import mutualInformation as mI
# Description !!
#
#

def loadOrignalInput():
    imagesMat = sio.loadmat('Input_Data/input_IRE.mat')
    image = imagesMat['inputImage']
    return(image)

#---------------- plotting functions ----------------#
def plotInputImage(input_Image):
    fig = mp.pyplot.figure()
    mp.pyplot.imshow(input_Image,cmap=plt.get_cmap('gray'),aspect='auto',interpolation='nearest')
    fig.savefig('./Output/IRE/IRE_input.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------    
def plotIRE_Image(ire_Image):
    plt.figure()#figsize=(28,28),dpi=800)
    plt.imshow(ire_Image,cmap=plt.get_cmap('gray'),aspect='auto',interpolation='none')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig('./Output/IRE/IRE_image.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotHistONOFF(ireImage):
    w,h,d =np.shape(ireImage)
    stepSize = 1.0/20.0
    bins = 20#np.arange(0,1+stepSize,stepSize)
    plt.figure()
    plt.hist(np.reshape(ireImage[:,:,0],w*h),bins = bins,facecolor='b',label='On' )#,histtype='stepfilled')
    plt.hist(np.reshape(ireImage[:,:,1],w*h),bins = bins,facecolor='g',label='Off')#,histtype='stepfilled')
    plt.legend()
    plt.xlabel('pixel value',fontsize=23,fontweight='bold')
    plt.ylabel('# of Pixels',fontsize=23,fontweight='bold')
    plt.savefig('./Output/IRE/OnOffHist_stepSize.png',bbox_inches='tight', pad_inches = 0.1)

    stepSize = 1.0/20.0
    plt.figure()
    plt.hist(np.reshape(ireImage[:,:,0],w*h),bins = bins,facecolor='b',label='On' ,histtype='stepfilled')
    plt.hist(np.reshape(ireImage[:,:,1],w*h),bins = bins,facecolor='g',label='Off',histtype='stepfilled')
    plt.legend()
    plt.xlabel('pixel value',fontsize = 23,fontweight='bold')
    plt.ylabel('# of Pixels',fontsize = 23,fontweight='bold')
    plt.savefig('./Output/IRE/OnOffHist.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def plotHists(inputImage,ireImage):
    w,h =np.shape(ireImage)
    plt.figure()
    plt.hist( np.reshape(ireImage,w*h),20)
    plt.xlabel('pixel value',fontsize=23,fontweight='bold')
    plt.ylabel('# of Pixels',fontsize=23,fontweight='bold')
    #plt.xlim(xmin=-10,xmax=10)
    plt.ylim(ymin=0,ymax=120000)
    plt.savefig('./Output/IRE/IREHist.png',bbox_inches='tight', pad_inches = 0.1)

    w,h =np.shape(inputImage)
    plt.figure()
    plt.hist( np.reshape(inputImage,w*h),20)
    plt.xlabel('pixel value',fontsize=23,fontweight='bold')
    plt.ylabel('number',fontsize=23,fontweight='bold')
    plt.savefig('./Output/IRE/InputHist.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotFourier(ffInput,ffIRE):
    # Plot the 2D- Frequency spectrum from the Fourier - Analyses.
    # Shift therefore the zero-frequency component to the center of the spectrum.
    # Use natural logarithm to increase the high number of small values and 
    # to depress the few high values to improve the comparability and visibility of the plot

    inputShift = np.log(np.abs(np.fft.fftshift(ffInput)))**2
    plt.figure()
    plt.imshow(inputShift)
    cb = plt.colorbar()
    cb.set_label('Magnitude of specific wave')
    plt.xlabel('Frequency along x-Axis',fontsize=18)
    plt.ylabel('Frequency along y-Axis',fontsize=18)
    plt.savefig('./Output/IRE/FourierMagnInput.png',bbox_inches='tight', pad_inches = 0.1)
    h,w = np.shape(inputShift)
    plt.figure()
    plt.hist(np.reshape(inputShift,w*h),20)
    plt.savefig('./Output/IRE/FourierMagnHistInput.png',bbox_inches='tight', pad_inches = 0.1)

    ireShift=np.log(np.abs(np.fft.fftshift(ffIRE)))**2
    plt.figure()
    plt.imshow(ireShift)
    cb = plt.colorbar()
    cb.set_label('Magnitude of specific wave')
    plt.xlabel('Frequency along x-Axis',fontsize=23)
    plt.ylabel('Frequency along y-Axis',fontsize=23)
    plt.savefig('./Output/IRE/FourierMagnIRE.png',bbox_inches='tight', pad_inches = 0.1)
    h,w = np.shape(ireShift)
    plt.figure()
    plt.hist(np.reshape(ireShift,w*h),20)
    plt.savefig('./Output/IRE/FourierMagnHistIRE.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotDifPixelLGN(differences):
    plt.figure(figsize=(20,10))
    plt.hist(differences,20)
    plt.xlabel('Difference between LGN and input pixel')
    plt.ylabel('# of pixel')
    plt.savefig('./Output/IRE/DifferencPixelLGN.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
#--------------------------- calculating functions ---------------------------#
def calcIRE_Image(frEx,ffW,patchsize):
    # reconstruction of the input Image,  
    # based on the network weights and the cell activity
    pixelStep = 3
    post,pre = np.shape(ffW)
    numberOfNeurons,nbrOfPatches=np.shape(frEx)
    w = (int(np.sqrt(nbrOfPatches))*pixelStep) + patchsize # * patchsize
    h = w
    ire_Image = np.zeros((w,h,2))
    counter_M = np.ones((w,h,2))
    for x in range(0,w-patchsize,pixelStep):
        for y in range(0,h - patchsize,pixelStep):
            patchActivity=np.zeros((patchsize,patchsize,2))
            xPosMin=0 + (1*x)
            xPosMax=patchsize+ (1*x)
            yPosMin=0+ (1*y)
            yPosMax=patchsize+ (1*y)
            patchIndex = int((y/pixelStep) +(x/pixelStep)*((h-patchsize)/pixelStep))
            #print(patchIndex)
            for neuron in range(numberOfNeurons):
                weight = np.reshape(ffW[neuron,:],(patchsize,patchsize,2))
                patchActivity=patchActivity + (weight* frEx[neuron,patchIndex]) 
            ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] = ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] + (patchActivity/numberOfNeurons)
            counter_M[xPosMin:xPosMax,yPosMin:yPosMax,:]+= 1#*np.mean(frEx[:,patchIndex])/numberOfNeurons
    ire_Image = ire_Image/(counter_M)
    return(ire_Image)
#------------------------------------------------------------------------------
def calcIRE(input_Image,ire_Image):
    # via NMSE after Spartling(2012)
    # for [0,1] normalized Images

    #old used Error
    errorMatrix =  np.sum((input_Image - ire_Image)**2)/(np.sum(ire_Image**2))
    errorAbs =  np.sum((np.abs(input_Image) - np.abs(ire_Image))**2)/(np.sum(ire_Image**2))

    # not used error alternative
    #w,h = np.shape(ire_Image)
    #ireV = np.reshape(ire_Image, w*h)
    #iptV = np.reshape(input_Image,w*h)
    #errorMatrix =  (np.sum(iptV-ireV)**2)/np.sum((np.abs(iptV) - np.abs(ireV))**2)
    #errorAbs =  0.0
    return(errorMatrix,errorAbs)
#------------------------------------------------------------------------------
def calcStatisticProperties(inputImage,ireImage):
    #inputImage = inputImage[:,:,0] - inputImage[:,:,1]
    #ireImage = ireImage[:,:,0] - ireImage[:,:,1]
    w,h = np.shape(inputImage)
    ireImage = np.reshape(ireImage,w*h)
    inputImage = np.reshape(inputImage,w*h)
    #variance
    varInput = np.var(inputImage)
    varIRE = np.var(ireImage)
    #mean/expected value
    meanInput=np.mean(inputImage)
    meanIRE=np.mean(ireImage)
    #standard deviation
    stdInput = np.std(inputImage)
    stdIRE = np.std(ireImage)
    #calculate covariance
    covInput = np.mean((inputImage-meanInput)*(inputImage-meanInput) )
    covInputIre = np.mean((inputImage-meanInput)*(ireImage-meanIRE) )
    #calculate correlation coefficient
    corInput = covInput/(stdInput*stdInput )
    corInputIre= covInputIre/(stdInput*stdIRE)
    #save different statistic information in a dictionary and write in a .txt - file
    statIRE={'mean':meanIRE,'variance':varIRE,'standard deviation':stdIRE}
    statInput={'mean':meanInput,'variance':varInput,'standard deviation':stdInput}
    stats = {'Input':statInput,'IRE':statIRE}
    json.dump(stats,open('Output/IRE/Stats.txt','w'))
    return(corInput,corInputIre)
#------------------------------------------------------------------------------
def calcFourierMagnitude(inputImage,ireImage):
    #inputImage = inputImage[:,:,0] - inputImage[:,:,1]
    #ireImage = ireImage[:,:,0] - ireImage[:,:,1]
    ffIRE = np.fft.fft2(ireImage)
    ffInput = np.fft.fft2(inputImage)
    return(ffInput,ffIRE)
#------------------------------------------------------------------------------
def calcFourierFrequencys(inputImage,ireImage):
    #inputImage = inputImage[:,:,0] - inputImage[:,:,1]
    #ireImage = ireImage[:,:,0] - ireImage[:,:,1]
    ffIRE = np.fft.fft2(ireImage)
    freqIRE =np.fft.fftfreq(len(ireImage)**2)
    plt.figure()
    plt.hist(freqIRE,100)
    plt.savefig('./Output/IRE/Frequencys.png',bbox_inches='tight', pad_inches = 0.1)
    ffInput = np.fft.fft2(inputImage)
    #freqInput=np.fft.fftfreq()
    return(freqIRE)
#------------------------------------------------------------------------------
def calcTM(input_Image,ire_Image):
    w,h = np.shape(ire_Image)
    irV = np.reshape(ire_Image,w*h)
    iptV= np.reshape(input_Image,w*h)
    tm = 0
    normX = np.linalg.norm(irV)
    normY = np.linalg.norm(iptV)
    if (normX != 0 and normY !=0):
        tm = (np.dot(irV,iptV) / (normX*normY))
    return(tm)
#------------------------------------------------------------------------------
def calcIREoverRMS(input_Image,ire_Image):
    #error after King et al. (2013)
    resError          = input_Image - ire_Image
    thisRmsResErr     = np.sqrt(np.mean(resError**2)) #sqrt(mean(resError(:).^2));    
    error = thisRmsResErr
    return(error)
#------------------------------------------------------------------------------
def calcIREoverRMS2(input_Image,ire_Image):
    h,w=(np.shape(ire_Image))
    n = h*w
    error = np.sqrt(np.sum((input_Image-ire_Image)**2)*1/n )
    return(error)
#------------------------------------------------------------------------------
def calcLGNIRE(frLGN,patchsize):
# reconstruction of the input Image,  
    # based on the network weights and the cell activity
    pixelStep = 3
    nbrCells,nbrPatches=np.shape(frLGN)
    w = (int(np.sqrt(nbrPatches))*pixelStep) + patchsize # * patchsize
    h = w
    ire_Image = np.zeros((w,h,2))
    counter_M = np.ones((w,h,2))
    for x in range(0,w-patchsize,pixelStep):
        for y in range(0,h - patchsize,pixelStep):
            patchActivity=np.zeros((patchsize,patchsize,2))
            xPosMin=0 + (1*x)
            xPosMax=patchsize+ (1*x)
            yPosMin=0+ (1*y)
            yPosMax=patchsize+ (1*y)
            patchIndex = int((y/pixelStep) +(x/pixelStep)*((h-patchsize)/pixelStep))
            patchActivity = np.reshape(frLGN[:,patchIndex],(patchsize,patchsize,2))
            ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] = ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] + (patchActivity)
            counter_M[xPosMin:xPosMax,yPosMin:yPosMax,:]+= 1#*np.mean(frEx[:,patchIndex])/numberOfNeurons
    ire_Image = ire_Image/(counter_M)
    return(ire_Image)
#------------------------------------------------------------------------------
def calcDiffBetweenPixelLGN(frLGN,patchesIRE):
    nbrOfPatches,patchsize = np.shape(patchesIRE)[0:2]
    inputPixels = np.reshape(patchesIRE,(nbrOfPatches,patchsize*patchsize*2))
    differences = (frLGN/125.0 - inputPixels.T)
    differences = np.mean(differences,axis=1)
    return(differences)
#------------------------------------------------------------------------------
def calcStreamIRE(activeFR,activeInp,synWeights):
    nbrCells,nbrPatches = np.shape(activeFR)
    patchsize = np.shape(activeInp)[1]
    reconstrImages = np.zeros(np.shape(activeInp))
    for i in range(nbrPatches):
        for j in range(nbrCells):
            img = activeFR[j,i]*synWeights[j,:]
            img = np.reshape(img,(patchsize,patchsize,2))
            reconstrImages[:,:,:,i] += img
        reconstrImages[:,:,:,i] = reconstrImages[:,:,:,i]/nbrCells

    reconstrImages = reconstrImages[:,:,0,:] - reconstrImages[:,:,1,:]
    activeInp = activeInp[:,:,0,:] - activeInp[:,:,1,:]
    ire = 0.0
    for i in xrange(nbrPatches):
        if np.mean(reconstrImages[:,:,i] != 0.0):
            reconstrImages[:,:,i] = (reconstrImages[:,:,i]-np.mean(reconstrImages[:,:,i]))/np.std(reconstrImages[:,:,i])
        activeInp[:,:,i] = (activeInp[:,:,i]-np.mean(activeInp[:,:,i]))/np.std(activeInp[:,:,i])
        n = patchsize*patchsize#* nbrPatches
        ire += calcIREoverRMS2(activeInp[:,:,i],reconstrImages[:,:,i])
    print((ire/nbrPatches))
#------------------------------------------------------------------------------
def plotMIHist(mI,name):
    plt.figure()
    plt.hist(mI)
    plt.savefig('./Output/IRE/'+name+'.png')
#------------------------------------------------------------------------------
def startIREAnalysis():
    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 25)
    plt.rc('ytick',labelsize = 25)

    #----load necessary datas----# 
    frIREs = np.load('work/IRE_fr.npy')
    patchesIRE = np.load('work/IRE_singlePatches.npy')
    ffW = np.loadtxt('Input_network/V1weight.txt')
    frLGNs = np.load('work/IRE_LGNFR.npy')
    inputImages=np.load('work/IRE_Images.npy')#loadOrignalInput()

    #activeFR  = np.load('work/Active_fr.npy')
    #activeInp = np.load('work/Active_InptPatches.npy')
    #activeLGN = np.load('work/Active_frLGN.npy')
    #calcStreamIRE(activeFR,activeInp,ffW)

    patchsizeIRE = np.shape(patchesIRE)[1]
    
    ire = np.zeros(10)
    print('Calculate IRE Image and IRE')
    for i in range(10):
        frIRE = frIREs[:,i,:]
        frLGN = frLGNs[:,i,:]
        inputImage= inputImages[i,:,:,:]

        ireImage = calcIRE_Image(frIRE,ffW,patchsizeIRE)
        np.save('./Output/IRE/IREImage_'+str(i),ireImage)
        plotHistONOFF(ireImage)    

        differences = calcDiffBetweenPixelLGN(frLGN,patchesIRE)
        plotDifPixelLGN(differences)
        ireLGN = calcLGNIRE(frLGN,patchsizeIRE)

        inputImage=inputImage[:,:,0] - inputImage[:,:,1]
        ireImage = ireImage[:,:,0] - ireImage[:,:,1]
        ireLGN = ireLGN[:,:,0]- ireLGN[:,:,1]
        
        imageSize = np.shape(inputImage)[0]

        inputSTD = (inputImage-np.mean(inputImage))/np.std(inputImage)
        lgnSTD = (ireLGN-np.mean(ireLGN))/np.std(ireLGN)
        ireSTD = (ireImage-np.mean(ireImage))/np.std(ireImage)

        inputImageArr = np.reshape(inputImage,imageSize*imageSize)
        ireImageArr =np.reshape(ireImage,imageSize*imageSize)
        ireLGNArr = np.reshape(ireLGN,imageSize*imageSize)

        inputSTDArr = np.reshape(inputSTD,imageSize*imageSize)
        lgnSTDArr = np.reshape(lgnSTD,imageSize*imageSize)
        ireSTDArr = np.reshape(ireSTD,imageSize*imageSize)


        mIInptLGN = mI.calc_MI_Hist(inputImageArr,ireLGNArr,9,False)
        mIInptV1 = mI.calc_MI_Hist(inputImageArr,ireImageArr,9,False)
        mILGNV1 = mI.calc_MI_Hist(ireLGNArr,ireImageArr,9,False)

        mIInptLGNSTD = mI.calc_MI_Hist(inputSTDArr,lgnSTDArr,9,False)
        mIInptV1STD = mI.calc_MI_Hist(inputSTDArr,ireSTDArr,9,False)
        mILGNV1STD = mI.calc_MI_Hist(lgnSTDArr,ireSTDArr,9,False)



        rmsSTDNormLGN = calcIREoverRMS2((inputImage-np.mean(inputImage))/np.std(inputImage),(ireLGN-np.mean(ireLGN))/np.std(ireLGN))
        tmSTDNormLGN = calcTM((inputImage-np.mean(inputImage))/np.std(inputImage),(ireLGN-np.mean(ireLGN))/np.std(ireLGN))

        rmsSTDNormLGNV1 = calcIREoverRMS2((ireLGN-np.mean(ireLGN))/np.std(ireLGN),(ireImage-np.mean(ireImage))/np.std(ireImage))
        tmSTDNormLGNV1 = calcTM((ireLGN-np.mean(ireLGN))/np.std(ireLGN),(ireImage-np.mean(ireImage))/np.std(ireImage))

        rmsSTDNorm = calcIREoverRMS2((inputImage-np.mean(inputImage))/np.std(inputImage),(ireImage-np.mean(ireImage))/np.std(ireImage))
        tmSTDNorm=calcTM((inputImage-np.mean(inputImage))/np.std(inputImage),(ireImage-np.mean(ireImage))/np.std(ireImage))


        ireMatrix,ireAbs = calcIRE(inputImage,ireImage)    
        ffInput,ffIRE = calcFourierMagnitude(inputImage,ireImage)
        freqIRE = calcFourierFrequencys(inputImage,ireImage)



        stat = {'IRE_Input_LGN_STDNorm':rmsSTDNormLGN,'TM_Input_LGN_STDNorm':tmSTDNormLGN,
                'IRE_Input_V1_STDNorm':rmsSTDNorm,'TM_Input_V1_STDNorm':tmSTDNorm,
                'IRE_LGN_V1_STDNorm':rmsSTDNormLGNV1,'TM_LGN_V1_STDNorm':tmSTDNormLGNV1,
                'MI_Input_LGN':mIInptLGN,'MI_Input_V1':mIInptV1,'MI_LGN_V1':mILGNV1,
                'MI_Input_LGNSTD':mIInptLGNSTD,'MI_Input_V1STD':mIInptV1STD,'MI_LGN_V1STD':mILGNV1STD}
        json.dump(stat,open('./Output/IRE/IRE_'+str(i)+'.txt','w'))
        ire[i] = rmsSTDNorm
    #print('Plot Image and IRE')
    #plotIRE_Image(ireImage)
    #plotHists(inputImage,ireImage)
    #plotFourier(ffInput,ffIRE)

    plt.figure()
    plt.plot(ire,'o')
    plt.title('MeanIRE= '+str(np.round(np.mean(ire),6) ))
    plt.ylabel('IRE')
    plt.xlabel('Image index')
    plt.ylim(ymin=0.5,ymax=1.2)
    plt.savefig('./Output/IRE/IRE_all.png')
    print('Finish with IRE')
#-----------------------------------------------------
if __name__=="__main__":
    startIREAnalysis()

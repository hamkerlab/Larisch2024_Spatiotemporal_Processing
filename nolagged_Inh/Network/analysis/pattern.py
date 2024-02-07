import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os.path
import mutualInformation as mi
import sys

def calcTemplateMatch(X,Y):
    tm = 0
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if (normX != 0 and normY !=0):
        tm = (np.dot(X,Y) / (normX*normY))
    return(tm)
#------------------------------------------------------------------------------
def calcTMBetweenImages(data1,data2):
    occLvL,sizeX,sizeY,depth = np.shape(data1)
    data1 = data1[:,:,:,0] - data1[:,:,:,1]
    data2 = data2[:,:,:,0] - data2[:,:,:,1]
    tm = np.zeros(occLvL)
    for i in range(occLvL):
        array1 = np.reshape(data1[i], sizeX*sizeY)
        array2 = np.reshape(data2[i], sizeX*sizeY)
        tm[i] = calcTemplateMatch(array1,array2)
    return(tm)
#------------------------------------------------------------------------------
def calcTMBetweenPixels(data1,data2):
    nbrPatches,sizeX,sizeY,depth = np.shape(data1)
    data1 = data1[:,:,:,0] - data1[:,:,:,1]
    data2 = data2[:,:,:,0] - data2[:,:,:,1]
    tm = np.zeros((sizeX,sizeY))
    for x in range(sizeX):
        for y in range(sizeY):
            tm[x,y] = calcTemplateMatch(data1[:,x,y],data2[:,x,y])
    return(tm)
#------------------------------------------------------------------------------
def reconstrPatch(frExc,weightsV1,patchsize,task):
    nbrOfPatches,depth,occLvL,nbrOfNeurons = np.shape(frExc)
    print(patchsize)
    reconstrIMG = np.zeros((nbrOfPatches,2,occLvL,patchsize,patchsize,2))
    weights = np.zeros((nbrOfNeurons,patchsize,patchsize,2))
    for i in range(nbrOfNeurons):
        weights[i] = np.reshape(weightsV1[i],(patchsize,patchsize,2))
    for i in range(nbrOfPatches):    
        for k in range(occLvL):
            for j in range(nbrOfNeurons):
               reconstrIMG[i,0,k] += frExc[i,0,k,j] * weights[j]
               reconstrIMG[i,1,k] += frExc[i,1,k,j] * weights[j] 
            reconstrIMG[i,:,k] = reconstrIMG[i,:,k]/nbrOfNeurons
    np.save('./work/Pattern_'+task+'_reconstrIMG',reconstrIMG)
    return(reconstrIMG)
#------------------------------------------------------------------------------
def plotTM_IMG(meanOrig,meanReco,name):
    plt.figure()
    plt.plot(meanOrig,'b-o',label = 'input image')
    plt.plot(meanReco,'r-+',label = 'reconstructed image')
    plt.xlabel('level of occlusion')
    plt.xlim(0,len(meanReco)-1)
    plt.xticks(np.linspace(0,len(meanReco)-1,6),np.linspace(0,100,6))
    plt.ylabel('template Match')
    plt.legend()
    plt.savefig('./Output/Pattern/'+name+'/Reconstruct.png',bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
#------------------------------------------------------------------------------
def plotMeanFROfOcclusion(frExc,name,matter):
    meanFr = np.mean(frExc,axis=2) # mean over cells
    meanFr = np.mean(meanFr,axis=0)#mean over patches
    plt.figure()
    plt.plot(meanFr,'bo')
    plt.xlabel('level of occlusion')
    plt.xlim(0,len(meanFr)-1)
    plt.xticks(np.linspace(0,len(meanFr)-1,6),np.linspace(0,100,6))
    plt.ylabel('mean fire rate')
    plt.savefig('./Output/Pattern/'+name+'/MeanFR_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
#------------------------------------------------------------------------------
def plotScatterTMImages(reconstTM,name):
    nbrPatches,lvls = np.shape(reconstTM)
    plt.figure()
    for i in range(nbrPatches):
        plt.plot(reconstTM[i,:],'o')
    plt.xlabel('level of occlusion')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel('TM of rec. images')
    plt.savefig('./Output/Pattern/'+name+'/ScatterTM_reconst.png',bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
#------------------------------------------------------------------------------
def plotImages(inptOcc,recImg):   
    nbrPatches,dept,occLVL = np.shape(inptOcc)[0:3]
    nbr = np.random.randint(0,nbrPatches,3)
    for n in nbr:
    
        imgO_1 = inptOcc[n,1,0,:,:,:]
        imgO_2 = inptOcc[n,1,5,:,:,:]
        imgO_3 = inptOcc[n,1,19,:,:,:]

        imgRI_1 = recImg[n,0,0,:,:,:]
        imgRI_2 = recImg[n,0,5,:,:,:]
        imgRI_3 = recImg[n,0,19,:,:,:]


        imgR_1 = recImg[n,1,0,:,:,:]
        imgR_2 = recImg[n,1,5,:,:,:]
        imgR_3 = recImg[n,1,19,:,:,:]

        plt.figure()
        plt.imshow(imgO_1[:,:,0] - imgO_1[:,:,1],cmap=plt.get_cmap('gray'),interpolation='none')
        plt.savefig('./Output/Pattern/Images/Input_'+str(n)+'_Img_1Ocll.png',bbox_inches='tight', pad_inches = 0.1)

        plt.figure()
        plt.imshow(imgO_2[:,:,0] - imgO_2[:,:,1],cmap=plt.get_cmap('gray'),interpolation='none')
        plt.savefig('./Output/Pattern/Images/Input_'+str(n)+'_Img_2Ocll.png',bbox_inches='tight', pad_inches = 0.1)

        plt.figure()
        plt.imshow(imgO_3[:,:,0] - imgO_3[:,:,1],cmap=plt.get_cmap('gray'),interpolation='none')
        plt.savefig('./Output/Pattern/Images/Input_'+str(n)+'_Img_3Ocll.png',bbox_inches='tight', pad_inches = 0.1)


        plt.figure()
        plt.imshow(imgR_1[:,:,0] - imgO_1[:,:,1],cmap=plt.get_cmap('gray'),interpolation='none')
        plt.savefig('./Output/Pattern/Images/Rec_'+str(n)+'_Img_1Ocll.png',bbox_inches='tight', pad_inches = 0.1)

        plt.figure()
        plt.imshow(imgR_1[:,:,0] - imgO_2[:,:,1],cmap=plt.get_cmap('gray'),interpolation='none')
        plt.savefig('./Output/Pattern/Images/Rec_'+str(n)+'_Img_2Ocll.png',bbox_inches='tight', pad_inches = 0.1)

        plt.figure()
        plt.imshow(imgR_1[:,:,0] - imgO_3[:,:,1],cmap=plt.get_cmap('gray'),interpolation='none')
        plt.savefig('./Output/Pattern/Images/Rec_'+str(n)+'_Img_3Ocll.png',bbox_inches='tight', pad_inches = 0.1)
        
        plt.close('all')
#------------------------------------------------------------------------------    
def calcCorrelation(origFr,occFr):
    print(np.shape(occFr))
    nbrPatches,occLvL,nbrCells = np.shape(occFr)
    correlM_pP = np.zeros((nbrPatches,occLvL))
    correlM_pC = np.zeros((nbrCells,occLvL))
    for i in range(nbrPatches):
        for j in range(occLvL):
            if ( (np.sum(occFr[i,j]) > 0.0) and (np.sum(origFr[i,0]) > 0.0)):
                correlM_pP[i,j]=np.corrcoef(origFr[i,0],occFr[i,j])[0,1]

    for i in range(nbrCells):
        for j in range(occLvL):
            if ( (np.sum(occFr[:,j,i]) > 0.0) and (np.sum(origFr[:,0,i]) > 0.0)):
                correlM_pC[i,j]=np.corrcoef(origFr[:,0,i],occFr[:,j,i])[0,1]
    return(correlM_pP,correlM_pC)
#------------------------------------------------------------------------------    
def calcCorrelation2(origFr,occFr):
    print(np.shape(occFr))
    nbrPatches,occLvL,nbrCells = np.shape(occFr)
    correlM_pP = np.zeros((nbrPatches,occLvL))
    correlM_pC = np.zeros((nbrCells,occLvL))
    for i in range(nbrPatches):
        for j in range(occLvL):
            if ( (np.sum(occFr[i,j]) > 0.0) and (np.sum(origFr[i,j]) > 0.0)):
                correlM_pP[i,j]=np.corrcoef(origFr[i,j],occFr[i,j])[0,1]

    for i in range(nbrCells):
        for j in range(occLvL):
            if ( (np.sum(occFr[:,j,i]) > 0.0) and (np.sum(origFr[:,j,i]) > 0.0)):
                correlM_pC[i,j]=np.corrcoef(origFr[:,j,i],occFr[:,j,i])[0,1]
    return(correlM_pP,correlM_pP)
#------------------------------------------------------------------------------
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
    plt.savefig('./Output/Pattern/'+name+'/'+matter+'_STD.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(minA,'b--+')
    plt.plot(maxA,'g--+')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel(matter)
    plt.xlabel('percent of replaced pixels',fontsize=20,weight = 'bold')
    plt.savefig('./Output/Pattern/'+name+'/'+matter+'_MinMax.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(stdA,'b-o')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.savefig('./Output/Pattern/'+name+'/'+matter+'_STDA.png')

    diff =np.zeros(len(stdA))
    for i in range(1,len(stdA)):
        diff[i] = np.abs( meanOverCells[i] - meanOverCells[i-1])
    plt.figure()
    plt.plot(diff,'-+')
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.savefig('./Output/Pattern/'+name+'/'+matter+'_DeltaMean.png')
    
    plt.close('all')
#------------------------------------------------------------------------------
def calcMutualInformation(origFr,occFr):
    nbrPatches,occLvL,nbrCells = np.shape(occFr)
    mutInfM = np.zeros((nbrCells,occLvL))
    mutInfMnorm = np.zeros((nbrCells,occLvL))
    binSize = 10
    for i in range(nbrCells):
        for j in range(occLvL):
            if np.sum(occFr[:,j,i]) == 0.0:
                mutInfM[i,j] = 0.0
                mutInfMnorm[i,j] = 0.0
            else:
                mutInfM[i,j]=mi.calc_MI_Hist(origFr[:,j,i],occFr[:,j,i],bins=binSize,norm=False)
                mutInfMnorm[i,j]=mi.calc_MI_Hist(origFr[:,j,i],occFr[:,j,i],bins=binSize,norm=True)
    return(mutInfM,mutInfMnorm) 
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
def calcNMSE(input_Image,ire_Image):
    # via NMSE after Spartling(2012)
    # for [0,1] normalized Images

    errorMatrix =  np.sum((input_Image - ire_Image)**2)/(np.sum(ire_Image**2))
    errorAbs =  np.sum((np.abs(input_Image) - np.abs(ire_Image))**2)/(np.sum(ire_Image**2))
    return(errorMatrix)
#------------------------------------------------------------------------------
def calcAndPlotReconDifference(recImg,name):
    nbrOfPatches,dept,occLvLs,patchsize = np.shape(recImg)[0:4]
    recDiff = np.zeros((nbrOfPatches,occLvLs-1))
    for i in range(nbrOfPatches):
        for j in range(0,occLvLs-1):
            if (np.sum(recImg[i,1,j+1]) == 0.0 ) and (np.sum(recImg[i,1,j+1]) == 0.0 ):
                recDiff[i,j] = 0.0
            else:
                recDiff[i,j] = calcNMSE(recImg[i,1,j],recImg[i,1,j+1]) 
    plt.figure()
    plt.plot(np.mean(recDiff,axis=0))
    plt.savefig('./Output/Pattern/'+name+'/DifferenceBetweenReconstImages_NMS.png')

    plt.figure()
    plt.hist(np.reshape(recDiff,nbrOfPatches*(occLvLs-1)))
    plt.savefig('./Output/Pattern/'+name+'/DifferenceBetweenReconstImages_NMSHist.png')

    recDiff = np.zeros((nbrOfPatches,(occLvLs-1)))
    for i in range(nbrOfPatches):
        for j in range(0,occLvLs-1):
            data1 = np.reshape(recImg[i,1,j],patchsize*patchsize*2)
            data2 = np.reshape(recImg[i,1,j+1],patchsize*patchsize*2)
            recDiff[i,j] = np.mean(data1-data2) 

    plt.figure()
    plt.plot(np.mean(recDiff,axis=0))
    plt.savefig('./Output/Pattern/'+name+'/DifferenceBetweenReconstImages_MeanError.png')

    plt.figure()
    plt.hist(np.reshape(recDiff,nbrOfPatches*(occLvLs-1)))
    plt.savefig('./Output/Pattern/'+name+'/DifferenceBetweenReconstImages_MeanHist.png')

    recTM = np.zeros((nbrOfPatches,occLvLs-1))
    for i in range(nbrOfPatches):
        for j in range(0,occLvLs-1):
            data1 = np.reshape(recImg[i,1,j],patchsize*patchsize*2)
            data2 = np.reshape(recImg[i,1,j+1],patchsize*patchsize*2)
            recDiff[i,j] = calcTemplateMatch(data1,data2) 
    plt.figure()
    plt.plot(np.mean(recDiff,axis=0))
    plt.savefig('./Output/Pattern/'+name+'/DifferenceBetweenReconstImages_TM.png')

    plt.figure() 
    plt.hist(np.reshape(recDiff,nbrOfPatches*(occLvLs-1)))
    plt.savefig('./Output/Pattern/'+name+'/DifferenceBetweenReconstImages_TMHist.png')
    
    plt.close('all')
#------------------------------------------------------------------------------
def estimateD_prime(frExc,inpt):
    n_patches,n_occl,n_cells,n_repeat = np.shape(frExc)
    w,h,d =np.shape(inpt)[2:]
    d_prime = np.zeros((n_patches,n_occl))
    for p in range(n_patches):
        for i in range(1,n_occl):
            resp_proj1 = np.zeros(n_repeat)
            resp_proj2 = np.zeros(n_repeat)

            #mean over the acticity of the choosen patches
            mu1 = np.mean(frExc[p,0],axis=1)
            mu2 = np.mean(frExc[p,i],axis=1)      

            diffvec = mu1 - mu2

            for i_repeat in range(n_repeat):
                resp_proj1[i_repeat] = np.dot(frExc[p,0,:,i_repeat],diffvec)
                resp_proj2[i_repeat] = np.dot(frExc[p,i,:,i_repeat],diffvec)

            mu_proj1 = np.mean(resp_proj1)
            var_proj1 = np.var(resp_proj1)   

            mu_proj2 = np.mean(resp_proj2)
            var_proj2 = np.var(resp_proj2)

            d_prime[p,i] = np.abs(mu_proj1- mu_proj2) / np.sqrt( 0.5*(var_proj1 + var_proj2) ) 
    return(d_prime)       
#------------------------------------------------------------------------------
def startAnalysisOccl():

    #----- analysis from pixel occlusion -----#
    inptOcc = np.load('./work/patternOccl_Input.npy')
    frExc = np.load('./work/patternOccl_frExc.npy')
    frLGN = np.load('./work/patternOccl_frLGN.npy')
    weights = np.loadtxt('./Input_network/V1weight.txt')

    d_p = estimateD_prime(frExc,inptOcc)
    np.save('./work/patternSwitch_dPrime',d_p)

    mean_dp=np.mean(d_p,axis=0)
    std_dp = np.std(d_p,axis=0,ddof=1)

    #----print the average d'prime per occl. level over the different patches--#
    plt.figure()
    plt.errorbar(range(0,len(mean_dp)),mean_dp,yerr=std_dp)
    plt.ylabel("d'",fontsize=13)
    plt.ylim(ymin=0.0,ymax=25)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.xlabel('replaced pixels [%]',fontsize=13)
    plt.savefig('./Output/Pattern/Switch/d_Prime.png',dpi=300,bbox_inches='tight')

    
    frExc = np.mean(frExc,axis=3)
    frLGN = np.mean(frLGN,axis=3)

    nbrPatches,lvlsOfOcc,nbrCells = np.shape(frExc)

    # switch the order of activity depending on the occlusion
    frExcBack = np.zeros((nbrPatches,lvlsOfOcc,nbrCells))
    for i in range(nbrCells):       
        for j in range(nbrPatches):
            d =0
            for c in range(lvlsOfOcc-1,-1,-1):
                values = frExc[j,c,i]
                frExcBack[j,d,i] = values
                d+=1

    # get actvity on 100% occluded/blended and calculate correlation with that

    frExcBlend = np.ones((nbrPatches,lvlsOfOcc,nbrCells))
    for i in range(nbrCells):
        values = frExc[:,lvlsOfOcc-1,i]
        for j in range(nbrPatches):
            frExcBlend[j,:,i] = frExcBlend[j,:,i]* values[j]

    print('Start analysis from pattern occlusion task')

    correlM_pP,correlM_pC =  calcCorrelation(frExc,frExc)
    correlInp_pP,correlInp_pC = calcCorrelation(frLGN,frLGN)
    correlMBack_pP,correlMBack_pC = calcCorrelation(frExc,frExcBack)
    plotStats(correlInp_pP,correlM_pP,'correlation_pP','Switch')
    plotStats(correlInp_pC,correlM_pC,'correlation_pC','Switch')
    plotStats(correlM_pP,correlMBack_pP,'correlationBack_pP','Switch')
    plotStats(correlM_pC,correlMBack_pC,'correlationBack_pC','Switch')

    correlMBlend_pP,correlMBlend_pC = calcCorrelation2(frExcBlend,frExcBack)

    meanOfInp_pP = np.mean(correlInp_pP,axis=0)
    meanOverCells_pP = np.mean(correlM_pP,axis=0)
    meanOverBack_pP = np.mean(correlMBack_pP,axis=0)
    meanOverBlend_pP = np.mean(correlMBlend_pP,axis=0)
    stdA = np.std(correlM_pP,axis=0,ddof=1)
    minA = np.min(correlM_pP,axis=0)
    maxA = np.max(correlM_pP,axis=0)
    stdABack = np.std(correlMBack_pP,axis=0,ddof=1)
    stdABlend = np.std(correlMBlend_pP,axis=0,ddof=1)

    plt.figure()
    plt.errorbar(range(0,len(meanOverCells_pP)),meanOverCells_pP,yerr=stdA,fmt='g--+',label='Recon',lw=2)
    plt.errorbar(range(0,len(meanOverBack_pP)),meanOverBack_pP,yerr=stdABack,fmt='r--+',label='backward',lw=2)
    plt.legend()
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel('correlation',fontsize=17,weight = 'bold')
    plt.xlabel('replaced pixels [%]',fontsize=17,weight = 'bold')
    plt.savefig('./Output/Pattern/'+'Switch'+'/'+'correlation'+'_STD_BackwardOcclusion_PerPatch.png',bbox_inches='tight', pad_inches = 0.1)

    meanOfInp_pC = np.mean(correlInp_pC,axis=0)
    meanOverCells_pC = np.mean(correlM_pC,axis=0)
    meanOverBack_pC = np.mean(correlMBack_pC,axis=0)
    meanOverBlend_pC = np.mean(correlMBlend_pC,axis=0)
    stdA = np.std(correlM_pC,axis=0,ddof=1)
    minA = np.min(correlM_pC,axis=0)
    maxA = np.max(correlM_pC,axis=0)
    stdABack = np.std(correlMBack_pC,axis=0,ddof=1)
    stdABlend = np.std(correlMBlend_pC,axis=0,ddof=1)

    plt.figure()
    plt.errorbar(range(0,len(meanOverCells_pC)),meanOverCells_pC,yerr=stdA,fmt='g--+',label='Recon',lw=2)
    plt.errorbar(range(0,len(meanOverBack_pC)),meanOverBack_pC,yerr=stdABack,fmt='r--+',label='backward',lw=2)
    plt.legend()
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel('correlation',fontsize=17,weight = 'bold')
    plt.xlabel('replaced pixels [%]',fontsize=17,weight = 'bold')
    plt.savefig('./Output/Pattern/'+'Switch'+'/'+'correlation'+'_STD_BackwardOcclusion_PerCell.png',bbox_inches='tight', pad_inches = 0.1)




    return -1
    #--------------------------------------------------------------------------
    mutInfM,mutInfMnorm = calcMutualInformation(frExc[:,0,:,:],frExc[:,1,:,:])
    inpMI,inpMINorm = calcMutualInformation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(inpMI,mutInfM,'mutualInformation','Switch')
    plotStats(inpMINorm,mutInfMnorm,'mutualInformationNorm','Switch')

    #----calculate the reconstructed patch---#
    patchsize = np.shape(inptOcc)[3]
    recImg = reconstrPatch(frExc,weights,patchsize,'Switch')
    ire = np.zeros((nbrPatches,lvlsOfOcc))
    for j in range(nbrPatches):
        for i in range(lvlsOfOcc):
            inpIMG =inptOcc[j,0,i,:,:,0] - inptOcc[j,0,i,:,:,1]
            inpIMG = (inpIMG - np.mean(inpIMG))/np.std(inpIMG)
            recImage = recImg[j,1,i,:,:,0] - recImg[j,1,i,:,:,1]        
            recImage = (recImage-np.mean(inpIMG))/np.std(recImage)
            ire[j,i] = calcIREoverRMS2(inpIMG,recImage)

    meanIRE = np.mean(ire,axis=0)
    stdIRE = np.std(ire,axis=0,ddof=1)

    x = np.linspace(0,lvlsOfOcc,lvlsOfOcc)
    plt.figure()
    plt.errorbar(x,meanIRE,yerr = stdIRE)
    plt.savefig('./Output/Pattern/'+'Switch'+'/'+'IRE.jpg',bbox_inches='tight', pad_inches = 0.1)

    calcAndPlotReconDifference(recImg,'Switch')
    nbrOfPatches,depth,occLvLs,patchsize = np.shape(inptOcc)[0:4]

    inptLGN = np.zeros(np.shape(inptOcc))
    for i in range(nbrOfPatches):
        for j in range(depth):
            for k in range(occLvLs):
                inptLGN[i,j,k] =  np.reshape(frLGN[i,j,k],(patchsize,patchsize,2))
    
    inputTM = np.zeros((nbrOfPatches,occLvLs))

    for j in range(nbrOfPatches):
        origImg = inptLGN[j,0]  #inptOcc[j,0]
        occlImg = inptLGN[j,1]  #inptOcc[j,1]
        inputTM[j] = calcTMBetweenImages(origImg,occlImg)
    meanTM = np.mean(inputTM,axis=0)
    
    reconstTM = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        origImg = inptLGN[j,1] #recImg[j,0]
        occlImg = recImg[j,1]
        reconstTM[j] = calcTMBetweenImages(origImg,occlImg)
    plotScatterTMImages(reconstTM,'Switch')
    plotTM_IMG(meanTM,np.mean(reconstTM,axis=0),'Switch')

    plotMeanFROfOcclusion(frExc[:,0,:,:],'Switch','Orig')
    plotMeanFROfOcclusion(frExc[:,1,:,:],'Switch','Occl')

#------------------------------------------------------------------------------
def startAnalysisNoise():
    #----- analysis from pixel occlusion with noise -----#
    print('Start analysis from pattern noisy task')
    inptOcc = np.load('./work/patternNoise_Input.npy')
    frExc = np.load('./work/patternNoise_frExc.npy')
    frLGN = np.load('./work/patternNoise_frLGN.npy')
    weights = np.loadtxt('./Input_network/V1weight.txt')
    nbrOfPatches,depth,occLvLs,patchSize = np.shape(inptOcc)[0:4]

    d_p = estimateD_prime(frExc,inptOcc)
    np.save('./work/patternNoise_dPrime',d_p)
    #----print the average d'prime per occl. level over the different patches--#
    plt.figure()
    plt.plot(np.mean(d_p,axis=0))
    plt.ylabel("d'",fontsize=13)
    plt.ylim(ymin=0.0,ymax=25)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.xlabel('replaced pixels [%]',fontsize=13)
    plt.savefig('./Output/Pattern/Noise/d_Prime.jpg',dpi=300,bbox_inches='tight')

    frExc = np.mean(frExc,axis=4) #mean over all repetations of one input
    frLGN = np.mean(frLGN,axis=4)

    nbrPatches,depth,lvlsOfOcc,nbrCells = np.shape(frExc)
    # switch the order of activity depending on the occlusion
    frExcBack = np.zeros((nbrPatches,lvlsOfOcc,nbrCells))
    for i in range(nbrCells):       
        for j in range(nbrPatches):
            d =0
            for c in range(lvlsOfOcc-1,-1,-1):
                values = frExc[j,1,c,i]
                frExcBack[j,d,i] = values
                d+=1

    # get actvity on 100% occluded/blended and calculate correlation with that
    frExcBlend = np.ones((nbrPatches,lvlsOfOcc,nbrCells))
    for i in range(nbrCells):
        values = frExc[:,1,lvlsOfOcc-1,i]
        for j in range(nbrPatches):
            frExcBlend[j,:,i] = frExcBlend[j,:,i]* values[j]

    correlM =  calcCorrelation(frExc[:,0,:,:],frExc[:,1,:,:])
    correlMBack = calcCorrelation(frExc[:,0,:,:],frExcBack)
    correlMBlend = calcCorrelation(frExcBlend,frExcBack)#frExc[:,1,:,:])
    correlInp = calcCorrelation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(correlInp,correlM,'correlation','Noise')

    meanOfInp = np.mean(correlInp,axis=0)
    meanOverCells = np.mean(correlM,axis=0)
    meanOverBack = np.mean(correlMBack,axis=0)
    meanOverBlend = np.mean(correlMBlend,axis=0)
    stdA = np.std(correlM,axis=0,ddof=1)
    minA = np.min(correlM,axis=0)
    maxA = np.max(correlM,axis=0)
    stdABack = np.std(correlMBack,axis=0,ddof=1)
    stdABlend = np.std(correlMBlend,axis=0,ddof=1)

    plt.figure()
    plt.errorbar(range(0,len(meanOverCells)),meanOverCells,yerr=stdA,fmt='g--+',label='Recon',lw=2)
    plt.errorbar(range(0,len(meanOverBack)),meanOverBack,yerr=stdABack,fmt='r--+',label='backward',lw=2)
    plt.legend()
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel('correlation',fontsize=17,weight = 'bold')
    plt.xlabel('replaced pixels [%]',fontsize=17,weight = 'bold')
    plt.savefig('./Output/Pattern/'+'Noise'+'/'+'correlation'+'_STD_BackwardOcclusion.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    #plt.errorbar(range(0,len(meanOverCells)),meanOverCells,yerr=stdA,fmt='g--+',label='Recon',lw=2)
    plt.errorbar(range(0,len(meanOverBlend)),meanOverBlend,yerr=stdABlend,fmt='g--+',label='last',lw=2)
    plt.legend()
    plt.xlim(0,20)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel('correlation',fontsize=17,weight = 'bold')
    plt.xlabel('replaced pixels [%]',fontsize=17,weight = 'bold')
    plt.savefig('./Output/Pattern/'+'Noise'+'/'+'correlation'+'_STD_CorrtoBlend.png',bbox_inches='tight', pad_inches = 0.1)


    mutInfM,mutInfMnorm = calcMutualInformation(frExc[:,0,:,:],frExc[:,1,:,:])
    inpMI,inpMINorm = calcMutualInformation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(inpMI,mutInfM,'mutualInformation','Noise')
    plotStats(inpMINorm,mutInfMnorm,'mutualInformationNorm','Noise')


    #----calculate the reconstructed patch---#
    patchsize = np.shape(inptOcc)[3]
    recImg = reconstrPatch(frExc,weights,patchsize,'Noise')
    ire = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        for i in range(occLvLs):
            inpIMG =inptOcc[j,0,i,:,:,0] - inptOcc[j,0,i,:,:,1]
            inpIMG = (inpIMG - np.mean(inpIMG))/np.std(inpIMG)
            recImage = recImg[j,1,i,:,:,0] - recImg[j,1,i,:,:,1]        
            recImage = (recImage-np.mean(inpIMG))/np.std(recImage)
            ire[j,i] = calcIREoverRMS2(inpIMG,recImage)

    meanIRE = np.mean(ire,axis=0)
    stdIRE = np.std(ire,axis=0,ddof=1)

    x = np.linspace(0,occLvLs,occLvLs)
    plt.figure()
    plt.errorbar(x,meanIRE,yerr = stdIRE)
    plt.savefig('./Output/Pattern/'+'Noise'+'/'+'IRE.jpg',bbox_inches='tight', pad_inches = 0.1)
    
    inputTM = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        origImg = inptOcc[j,0]
        occlImg = inptOcc[j,1]
        inputTM[j] = calcTMBetweenImages(origImg,occlImg)
    meanTM = np.mean(inputTM,axis=0)
    
    inputTMPixel = np.zeros((occLvLs,patchSize,patchSize))
    for j in range(occLvLs):
        origImg = inptOcc[:,0,j]
        occlImg = inptOcc[:,1,j]
        inputTMPixel[j,:,:] = calcTMBetweenPixels(origImg,occlImg)
    
    reconstTMPixel = np.zeros((occLvLs,patchSize,patchSize))
    for j in range(occLvLs):
        origImg = inptOcc[:,0,j]#recImg[:,0,j]
        occlImg = recImg[:,1,j]
        reconstTMPixel[j,:,:] = calcTMBetweenPixels(origImg,occlImg)

    mean = np.mean(inputTMPixel,axis=1)
    meanRec=np.mean(reconstTMPixel,axis=1)
    plt.figure()
    plt.plot(np.mean(mean,axis=1))
    plt.plot(np.mean(meanRec,axis=1))
    plt.savefig('Output/Pattern/Test.png')

    reconstTM = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        origImg = recImg[j,0]
        occlImg = recImg[j,1]
        reconstTM[j] = calcTMBetweenImages(origImg,occlImg)
    plotScatterTMImages(reconstTM,'Noise')
    plotTM_IMG(meanTM,np.mean(reconstTM,axis=0),'Noise')

    plotMeanFROfOcclusion(frExc[:,0,:,:],'Noise','Orig')
    plotMeanFROfOcclusion(frExc[:,1,:,:],'Noise','Occl')
    plotImages(inptOcc,recImg)

#------------------------------------------------------------------------------
def startAnalysisOppo():

    #----- analysis from pixel occlusion -----#
    inptOcc = np.load('./work/patternOcclTM_Input.npy')
    frExc = np.load('./work/patternOcclTM_frExc.npy')
    frLGN = np.load('./work/patternOcclTM_frLGN.npy')
    weights = np.loadtxt('./Input_network/V1weight.txt')

    nbrPatches,depth,lvlsOfOcc,nbrCells = np.shape(frExc)

    frExcEnd = np.ones((nbrPatches,lvlsOfOcc,nbrCells))
    for i in range(nbrCells):
        values = frExc[:,1,lvlsOfOcc-1,i]
        for j in range(nbrPatches):
            frExcEnd[j,:,i] = frExcEnd[j,:,i]* values[j]
    
    print('Start analysis from pattern opposition task')

    correlM =  calcCorrelation(frExc[:,0,:,:],frExc[:,1,:,:])
    correlMEnd=calcCorrelation(frExcEnd,frExc[:,0,:,:])
    correlInp = calcCorrelation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(correlInp,correlM,'correlation','Oppo')

    meanOfInp = np.mean(correlInp,axis=0)
    meanOverCells = np.mean(correlM,axis=0)
    meanOverEnd = np.mean(correlMEnd,axis=0)
    stdA = np.std(correlM,axis=0)
    minA = np.min(correlM,axis=0)
    maxA = np.max(correlM,axis=0)

    plt.figure()
    plt.errorbar(range(0,len(meanOverCells)),meanOverCells,yerr=stdA,fmt='g--+',label='Recon',lw=2)
    plt.errorbar(range(0,len(meanOverEnd)),meanOverEnd,yerr=stdA,fmt='r--+',label='switch',lw=2)
    plt.legend()
    plt.xlim(0,20)
    plt.ylim(-0.5,1.2)
    plt.xticks(np.linspace(0,20,6),np.linspace(0,100,6))
    plt.ylabel('correlation',fontsize=17,weight = 'bold')
    plt.xlabel('replaced pixels [%]',fontsize=17,weight = 'bold')
    plt.savefig('./Output/Pattern/'+'Oppo'+'/'+'correlation'+'_STD_both.png',bbox_inches='tight', pad_inches = 0.1)

    mutInfM,mutInfMnorm = calcMutualInformation(frExc[:,0,:,:],frExc[:,1,:,:])
    inpMI,inpMINorm = calcMutualInformation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(inpMI,mutInfM,'mutualInformation','Oppo')
    plotStats(inpMINorm,mutInfMnorm,'mutualInformationNorm','Oppo')

    #----calculate the reconstructed patch---#
    patchsize = np.shape(inptOcc)[3]
    recImg = reconstrPatch(frExc,weights,patchsize,'Oppo')
    ire = np.zeros((nbrPatches,lvlsOfOcc))
    for j in range(nbrPatches):
        for i in range(lvlsOfOcc):
            inpIMG =inptOcc[j,0,i,:,:,0] - inptOcc[j,0,i,:,:,1]
            inpIMG = (inpIMG - np.mean(inpIMG))/np.std(inpIMG)
            recImage = recImg[j,1,i,:,:,0] - recImg[j,1,i,:,:,1]        
            recImage = (recImage-np.mean(inpIMG))/np.std(recImage)
            ire[j,i] = calcIREoverRMS2(inpIMG,recImage)

    meanIRE = np.mean(ire,axis=0)
    stdIRE = np.std(ire,axis=0,ddof=1)

    x = np.linspace(0,lvlsOfOcc,lvlsOfOcc)
    plt.figure()
    plt.errorbar(x,meanIRE,yerr = stdIRE)
    plt.savefig('./Output/Pattern/'+'Oppo'+'/'+'IRE.jpg',bbox_inches='tight', pad_inches = 0.1)
    calcAndPlotReconDifference(recImg,'Oppo')
    nbrOfPatches,depth,occLvLs,patchsize = np.shape(inptOcc)[0:4]

    inptLGN = np.zeros(np.shape(inptOcc))
    for i in range(nbrOfPatches):
        for j in range(depth):
            for k in range(occLvLs):
                inptLGN[i,j,k] =  np.reshape(frLGN[i,j,k],(patchsize,patchsize,2))
    
    inputTM = np.zeros((nbrOfPatches,occLvLs))

    for j in range(nbrOfPatches):
        origImg = inptLGN[j,0]  #inptOcc[j,0]
        occlImg = inptLGN[j,1]  #inptOcc[j,1]
        inputTM[j] = calcTMBetweenImages(origImg,occlImg)
    meanTM = np.mean(inputTM,axis=0)
    

    reconstTM = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        origImg = recImg[j,0]
        occlImg = recImg[j,1]
        reconstTM[j] = calcTMBetweenImages(origImg,occlImg)
    plotScatterTMImages(reconstTM,'Oppo')
    plotTM_IMG(meanTM,np.mean(reconstTM,axis=0),'Oppo')

    plotMeanFROfOcclusion(frExc[:,0,:,:],'Oppo','Orig')
    plotMeanFROfOcclusion(frExc[:,1,:,:],'Oppo','Occl')
    plotImages(inptOcc,recImg)
#------------------------------------------------------------------------------
def startAnalysisZero():

    #----- analysis from pixel occlusion -----#
    inptOcc = np.load('./work/patternZero_Input.npy')
    frExc = np.load('./work/patternZero_frExc.npy')
    frLGN = np.load('./work/patternZero_frLGN.npy')
    weights = np.loadtxt('./Input_network/V1weight.txt')

    nbrOfPatches,depth,occLvLs,patchsize = np.shape(inptOcc)[0:4]

    print('Start analysis where pixels are set to zero')
    
    #----calculate the reconstructed patch---#
    patchsize = np.shape(inptOcc)[3]
    recImg = reconstrPatch(frExc,weights,patchsize,'Zero')
    ire = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        for i in range(occLvLs):
            inpIMG =inptOcc[j,0,i,:,:,0] - inptOcc[j,0,i,:,:,1]
            inpIMG = (inpIMG - np.mean(inpIMG))/np.std(inpIMG)
            recImage = recImg[j,1,i,:,:,0] - recImg[j,1,i,:,:,1]        
            recImage = (recImage-np.mean(inpIMG))/np.std(recImage)
            ire[j,i] = calcIREoverRMS2(inpIMG,recImage)

    meanIRE = np.mean(ire,axis=0)
    stdIRE = np.std(ire,axis=0,ddof=1)

    x = np.linspace(0,occLvLs,occLvLs)
    plt.figure()
    plt.errorbar(x,meanIRE,yerr = stdIRE)
    plt.savefig('./Output/Pattern/'+'Zero'+'/'+'IRE.jpg',bbox_inches='tight', pad_inches = 0.1)
    plotImages(inptOcc,recImg)


    correlM =  calcCorrelation(frExc[:,0,:,:],frExc[:,1,:,:])

    correlInp = calcCorrelation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(correlInp,correlM,'correlation','Zero')

    mutInfM,mutInfMnorm = calcMutualInformation(frExc[:,0,:,:],frExc[:,1,:,:])
    inpMI,inpMINorm = calcMutualInformation(frLGN[:,0,:,:],frLGN[:,1,:,:])
    plotStats(inpMI,mutInfM,'mutualInformation','Zero')
    plotStats(inpMINorm,mutInfMnorm,'mutualInformationNorm','Zero')

    calcAndPlotReconDifference(recImg,'Zero')
  
    inptLGN = np.zeros(np.shape(inptOcc))
    for i in range(nbrOfPatches):
        for j in range(depth):
            for k in range(occLvLs):
                inptLGN[i,j,k] =  np.reshape(frLGN[i,j,k],(patchsize,patchsize,2))
    
    inputTM = np.zeros((nbrOfPatches,occLvLs))

    for j in range(nbrOfPatches):
        origImg = inptLGN[j,0]  #inptOcc[j,0]
        occlImg = inptLGN[j,1]  #inptOcc[j,1]
        inputTM[j] = calcTMBetweenImages(origImg,occlImg)
    meanTM = np.mean(inputTM,axis=0)
    

    reconstTM = np.zeros((nbrOfPatches,occLvLs))
    for j in range(nbrOfPatches):
        origImg = recImg[j,0]
        occlImg = recImg[j,1]
        reconstTM[j] = calcTMBetweenImages(origImg,occlImg)
    plotScatterTMImages(reconstTM,'Oppo')
    plotTM_IMG(meanTM,np.mean(reconstTM,axis=0),'Zero')

    plotMeanFROfOcclusion(frExc[:,0,:,:],'Zero','Orig')
    plotMeanFROfOcclusion(frExc[:,1,:,:],'Zero','Occl')
    
#------------------------------------------------------------------------------
def startPatternCompletion(select):

    if not os.path.exists('./Output/Pattern'):
        os.mkdir('./Output/Pattern')
    if not os.path.exists('./Output/Pattern/Images'):
        os.mkdir('./Output/Pattern/Images')
    if not os.path.exists('./Output/Pattern/Switch'):
        os.mkdir('./Output/Pattern/Switch')
    if not os.path.exists('./Output/Pattern/Noise'):
        os.mkdir('./Output/Pattern/Noise')
    if not os.path.exists('./Output/Pattern/Oppo'):
        os.mkdir('./Output/Pattern/Oppo')
    if not os.path.exists('./Output/Pattern/Zero'):
        os.mkdir('./Output/Pattern/Zero')

    if select == 0:
        startAnalysisOccl()
    if select == 1:
        startAnalysisOppo()
    if select == 2:
        startAnalysisNoise()
    if select ==3:
        startAnalysisZero()
#----------------------------------------------------------    
if __name__=="__main__":
    data = (sys.argv[1:])
    select = 0
    if len(data) > 0:
        select = float(data[0])
    startPatternCompletion(select)

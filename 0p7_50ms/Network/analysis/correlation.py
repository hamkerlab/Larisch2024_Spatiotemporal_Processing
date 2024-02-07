import matplotlib as mp
mp.use('Agg')
from matplotlib.colors import LogNorm
import plots as plt
import numpy as np
import spiking
import json
import os
#------------------------------------------------------------------------------
def calcSpikeCorrelation(spikeMatrix,nbrOfNeurons):
    spkCorr = np.zeros((nbrOfNeurons,nbrOfNeurons))
    for i in range(nbrOfNeurons):
        spkPost = spikeMatrix[i,:]
        for j in range(nbrOfNeurons):
            spkPre = spikeMatrix[j,:]
            spkCorr[i,j] = np.corrcoef(spkPost,spkPre)[0,1]
        #spkCorr[i,i] = np.nan
        #print('Correllation to Neuron: %i' %(i))
    return(spkCorr)
#------------------------------------------------------------------------------
# calculate the noise correlation over subtracting the mean response of 
# each neuron to the stimulus across trails (see Stringer et al. 2016)
def calcNoiseCorrelation(cellActivity):
    nbrCells,nbrPatches,nbrTrials = np.shape(cellActivity)
    print(np.shape(cellActivity))
    c = np.zeros((nbrPatches,nbrCells,nbrCells))
    for patch in xrange(nbrPatches):        
        for i in xrange(nbrCells):
            meanCellPost = np.mean(cellActivity[i,patch])
            for j in xrange(nbrCells):    
                meanCellPre = np.mean(cellActivity[j,patch])
                c[patch,i,j]=np.sum((cellActivity[i,patch] - meanCellPost)*(cellActivity[j,patch] - meanCellPre))/nbrTrials
    noiseCorr = np.mean(c,axis=0)
    return(noiseCorr)
#------------------------------------------------------------------------------
def plotCorrelation(corrMatrix,path,desc):
    nbrOfPost,nbrOfPre = np.shape(corrMatrix)
    for i in range(nbrOfPost):
        corrMatrix[i,i] = np.nan
    corrArray = np.reshape(corrMatrix, nbrOfPost*nbrOfPre)
    mp.pyplot.figure()
    mp.pyplot.hist(corrArray[~np.isnan(corrArray)],15)
    #mp.pyplot.xlim(-0.5,0.5)
    #mp.pyplot.ylim(0,9000)
    mp.pyplot.savefig('./Output/ReceptiveFields/Correlation/'+path+'/'+desc+'_Hist.png',bbox_inches='tight', pad_inches = 0.1)
    mp.pyplot.close('all')
#------------------------------------------------------------------------------
def shanEntropy(c):
    cNormalized = c/np.float(np.sum(c))
    cNormalized = cNormalized[np.nonzero(cNormalized)]
    H = -np.sum(cNormalized* np.log2(cNormalized))  
    return(H)
#------------------------------------------------------------------------------
def calc_MI_Hist(X,Y,bins):
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    c_XY = np.histogram2d(X,Y,bins)[0]

    H_X = shanEntropy(c_X)
    H_Y = shanEntropy(c_Y)
    H_XY = shanEntropy(c_XY)

    MI = H_X + H_Y - H_XY
    return(MI)
#------------------------------------------------------------------------------
def calculateMISpk(spikeMatrix):
    nbrOfNeurons,nbrOfPatches = np.shape(spikeMatrix)
    miMatrix = np.zeros((nbrOfNeurons,nbrOfNeurons))
    for i in range(nbrOfNeurons):
        spkPost = spikeMatrix[i,:]
        for j in range(nbrOfNeurons):
            spkPre = spikeMatrix[j,:]
            miMatrix[i,j] = calc_MI_Hist(spkPost,spkPre,10) 
    return(miMatrix)
#------------------------------------------------------------------------------
def plotSpkCorrOrient(corr,orientations,desc):
    nbrOfNeurons = np.shape(corr)[0]
    img = np.zeros((nbrOfNeurons,nbrOfNeurons))
    idxOrt = np.argsort(orientations)
    for i in range(nbrOfNeurons):
        for j in range(nbrOfNeurons):
            img[i,j] = corr[idxOrt[i],idxOrt[j]]


    nullfmt = mp.ticker.NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_IMG   = [left,bottom,width,height]
    rect_histX = [left,bottom_h,width,0.2]
    rect_histY = [left_h,bottom,0.2,height]

    # start with a rectangular Figure

    mp.pyplot.figure(1, figsize=(8, 8))

    axIMG = mp.pyplot.axes(rect_IMG)
    axHistX = mp.pyplot.axes(rect_histX)
    axHistY = mp.pyplot.axes(rect_histY)

    # no labels
    axHistX.xaxis.set_major_formatter(nullfmt)
    axHistY.xaxis.set_major_formatter(nullfmt)
    axHistY.yaxis.set_major_formatter(nullfmt)

    # plot with imshow
    axIMG.imshow(img,cmap=mp.pyplot.get_cmap('gray'),interpolation='none')
    axIMG.set_xlabel('neuron index, ascending over orientation in degree')
    axIMG.set_ylabel('neuron index, ascending over orientation in degree')

    # normal plots
    axHistX.plot(orientations[idxOrt],'o')
    axHistX.set_ylabel('Orientation')
    axHistY.plot(orientations[np.argsort(orientations*-1)],np.linspace(0,len(orientations),len(orientations)),'o')

    # set limits by hand
    axHistX.set_xlim((0,144))
    axHistX.set_ylim((0,np.pi))

    axHistY.set_xlim((0,np.pi))
    axHistY.set_ylim((0,144))

    mp.pyplot.savefig('./Output/ReceptiveFields/Corr_Orientation_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
    mp.pyplot.close('all')
#------------------------------------------------------------------------------
def plotMIHist(matrixMI,path):
    nbrPost,nbrPre = np.shape(matrixMI)
    for i in range(nbrPost):
        matrixMI[i,i] = np.nan
    arrayMI = np.reshape(matrixMI,(nbrPost*nbrPre))
    mp.pyplot.figure()
    mp.pyplot.hist(arrayMI[~np.isnan(arrayMI)],20)
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/MISpk_Hist.png',bbox_inches='tight', pad_inches = 0.1)
    mp.pyplot.close('all')
#------------------------------------------------------------------------------
def calcRMSCorr(corrMatrix,identic = True):
    nbrOfPost,nbrOfPre = np.shape(corrMatrix)
    if identic: # if correlation calculated with himself
        for i in range(nbrOfPost):
            corrMatrix[i,i] = np.nan
    rmsCorr = np.sqrt(np.mean(corrMatrix[~np.isnan(corrMatrix)]**2))
    return(rmsCorr)
#------------------------------------------------------------------------------
def calcOrientCorr(activity,orientations):
    nbrCells,nbrOrientations,nbrSamples = np.shape(activity)
    prefOr = np.zeros(nbrCells)
    corrMatrix = np.zeros((nbrCells,nbrCells))
    for i in xrange(nbrCells):
        prefOr[i] = np.where(np.mean(activity[i],axis=1) == np.max(np.mean(activity[i],axis=1)))[0]
    for i in xrange(nbrCells):
        cell1 = activity[i,prefOr[i],:]
        for j in xrange(nbrCells):
            cell2 = activity[j,prefOr[j],:]
            corrMatrix[i,j] = np.corrcoef(cell1,cell2)[0,1]
    return(corrMatrix)
#------------------------------------------------------------------------------
def calcCorrelaltions(frExc):
    nbrCells,nbrPatches,nbrTrails = np.shape(frExc)
    totalCorr = np.zeros((nbrCells,nbrCells))
    signalCorr= np.zeros((nbrCells,nbrCells))
    noiseCorr = np.zeros((nbrCells,nbrCells))
    #calculate the total correlation, use every response in the correct order
    for i in xrange(nbrCells):
        spkPost = frExc[i]
        spkPost = np.reshape(spkPost,nbrPatches*nbrTrails)
        for j in xrange(nbrCells):
            spkPre = frExc[j]
            spkPre = np.reshape(spkPre,nbrPatches*nbrTrails)
            totalCorr[i,j] = np.corrcoef(spkPost,spkPre)[0,1]
    repeats = 3
    for k in xrange(repeats):
        #shuffle the single trails from one patch id
        shuffledSpk = np.zeros((nbrCells,nbrPatches,nbrTrails))
        for i in xrange(nbrCells):
            spkCell= np.copy(frExc[i])
            for j in xrange(nbrPatches):
                patchSpk = spkCell[j]
                np.random.shuffle(patchSpk)
                shuffledSpk[i,j] = patchSpk
        #calculate the signal correlation through shuffeling the trails of one patch
        for i in xrange(nbrCells):
            spkPost = shuffledSpk[i]
            spkPost = np.reshape(spkPost,nbrPatches*nbrTrails)
            for j in xrange(nbrCells):
                spkPre = shuffledSpk[j]
                spkPre = np.reshape(spkPre,nbrPatches*nbrTrails)
                signalCorr[i,j] += np.corrcoef(spkPost,spkPre)[0,1]
    signalCorr /=repeats
    noiseCorr = totalCorr - signalCorr
    return(totalCorr,signalCorr,noiseCorr) 
#------------------------------------------------------------------------------
def plotCorreNeurons(corrRF,frEx,path):
    idxmin = np.where(corrRF == np.min(corrRF))
    idxmax = np.where(corrRF == np.max(corrRF))
    frMin1 = frEx[idxmin[0][0]]
    frMin2 = frEx[idxmin[0][1]]
    sortIdx = np.argsort(frMin1)
    mp.pyplot.figure()
    mp.pyplot.scatter(frMin1[sortIdx],frMin2[sortIdx])
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/CorrelN.png',bbox_inches='tight', pad_inches = 0.1)
    mp.pyplot.close('all')
#------------------------------------------------------------------------------
def plotRFtoSpkCorr(tmRF,spkCorr,path,manner):
    nbrCells = np.shape(tmRF)[0]

    mp.pyplot.figure()
    for i in xrange(nbrCells):
        mp.pyplot.scatter(tmRF[i,:],spkCorr[i,:])
    mp.pyplot.xlabel('RF similarity')
    mp.pyplot.ylabel('spike count correlation')
    mp.pyplot.xlim(-1.0,1.0)
    #mp.pyplot.ylim(-0.2,1.0)
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_ALL.png',bbox_inches='tight')
    
    sortRF = np.zeros((nbrCells,nbrCells))
    sortSP = np.zeros((nbrCells,nbrCells))

    for i in xrange(nbrCells):
        idx = np.argsort(tmRF[i,:])
        sortRF[i,:] = tmRF[i,idx]
        sortSP[i,:] = spkCorr[i,idx]

    mp.pyplot.figure()
    mp.pyplot.scatter(np.mean(sortRF,axis=0),np.mean(sortSP,axis=0))#,'o')
    mp.pyplot.xlabel('RF similarity')
    mp.pyplot.ylabel('spike count correlation')
    mp.pyplot.xlim(-1.0,1.0)
    #mp.pyplot.ylim(-0.2,1.0)
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_MEAN.png',bbox_inches='tight')
    

    xmin = np.min(sortRF)
    xmax = np.max(sortRF)
    ymin = np.min(sortSP)
    ymax = np.max(sortSP)

    maxTM = xmax
    minTM = xmin

    sortRFA = np.reshape(sortRF,nbrCells*nbrCells)

    sizeA = 10
    tmRange = np.linspace(minTM,maxTM,sizeA)    
    tmPoints = np.zeros(sizeA-1)
    meanCorr = np.zeros(sizeA-1)
    elePerPoint = np.zeros(sizeA-1)
    corrPoints = []
    for i in xrange(sizeA-1):
        idx = (np.where((sortRF>=tmRange[i]) & (sortRF<tmRange[i+1])))       
        correlP = sortSP[idx[0],idx[1]]
        corrPoints.append(correlP)
        meanCorr[i] = np.mean(correlP)
        elePerPoint[i] = len(correlP)
        tmPoints[i] = np.mean([tmRange[i],tmRange[i+1]])

    mp.pyplot.figure()
    #mp.pyplot.plot(tmPoints[0:sizeA-2],meanCorr[0:sizeA-2],'-b')
    mp.pyplot.scatter(tmPoints[0:sizeA-2],meanCorr[0:sizeA-2],c=elePerPoint[0:sizeA-2],lw=0,alpha = 1.0,s=30.0)#,s=elePerPoint[0:sizeA-2]
    mp.pyplot.colorbar()
    mp.pyplot.xlim(-1.0,1.0)
    mp.pyplot.ylim(-1.0,1.0)
    mp.pyplot.xlabel('RF similarity')
    mp.pyplot.ylabel('spike count correlation')    
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_MEANCorrel_Scatter.jpg',dpi=300,bbox_inches='tight')

    mp.pyplot.figure()
    #mp.pyplot.plot(tmPoints,meanCorr,'-b')
    mp.pyplot.boxplot(corrPoints[0:sizeA-2],sym='')    
    #mp.pyplot.xlim(-1.0,1.0)
    #mp.pyplot.ylim(-1.0,1.0)
    mp.pyplot.xlabel('RF similarity')
    #mp.pyplot.xticks(np.linspace(0,sizeA-1,5),np.round(np.linspace(np.min(tmPoints),np.max(tmPoints),5),2))
    mp.pyplot.xticks(np.linspace(0,sizeA-1,6),np.round(np.linspace(-1,1,6),2))
    mp.pyplot.ylabel('spike count correlation')    
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_MEANCorrel_Box.jpg',dpi=300,bbox_inches='tight')
    

    binSize = 75#50
    heatAll = np.zeros((binSize,binSize))
    extent = []
    xedg = np.zeros(binSize)
    yedg = np.zeros(binSize)

    for i in xrange(nbrCells):
        #heat,xedges,yedges = np.histogram2d(sortRF[i,~np.isnan(sortRF[i,:])],sortSP[i,~np.isnan(sortSP[i,:])],bins = binSize,range=([xmin,xmax],[ymin,ymax]))
        heat,xedges,yedges = np.histogram2d(sortRF[i,0:nbrCells-1],sortSP[i,0:nbrCells-1],bins = binSize,range=([xmin,xmax],[ymin,ymax]))        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xedg = xedges
        yedg = yedges
        heatAll +=heat

    mp.pyplot.rc('xtick',labelsize = 20)
    mp.pyplot.rc('ytick',labelsize = 20)

    my_cmap = mp.pyplot.get_cmap('jet')
    my_cmap.set_under('w')

    mp.pyplot.figure()
    mp.pyplot.imshow(heatAll.T,extent=extent,origin='lower',interpolation='none',aspect='auto',cmap=my_cmap,vmin=0.001)#,norm=LogNorm())
    cbar = mp.pyplot.colorbar()
    cbar.set_label('# of neuron pairs',fontsize=22)#,weight='bold')
    mp.pyplot.xlabel('RF similarity',fontsize=22)
    mp.pyplot.ylabel('spike count correlation',fontsize=22)
    mp.pyplot.xlim(-1.0,1.0)
    #mp.pyplot.ylim(-0.2,1.0)
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_HIST.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    

    mp.pyplot.figure()
    mp.pyplot.hist(np.mean(heatAll.T,axis=1))
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_1DHist.png',bbox_inches='tight')

    idx = np.where(heatAll > 0.0)
    x = xedg[idx[0]]
    y = yedg[idx[1]]
    marker = heatAll[idx[0],idx[1]]


    mp.pyplot.figure()
    mp.pyplot.scatter(x,y, s= marker,c=marker,lw=0, alpha = 0.7)
    mp.pyplot.colorbar()
    #mp.pyplot.title('mean FR = '+str(meanFR))
    mp.pyplot.xlabel('RF similarity')
    mp.pyplot.ylabel('spike count correlation')
    mp.pyplot.xlim(-1.0,1.0)
    #mp.pyplot.ylim(-0.2,1.0)
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Correl_'+manner+'_SCATTER.png',bbox_inches='tight')
    mp.pyplot.close('all')
#------------------------------------------------------------------------------
def plotBar(totalCorr,signalCorr,noiseCorr,path,manner):
    fig = mp.pyplot.figure()
    fig,ax =  mp.pyplot.subplots()
    bar_width = 0.1
    r1 = mp.pyplot.bar(0.0 ,totalCorr,bar_width/2,
                  color = 'r',
                  label = str(np.round(totalCorr,4)),
                  linewidth=2.0)
    r2 = mp.pyplot.bar(bar_width,noiseCorr,bar_width/2,
                  color = 'g',
                  label = str(np.round(noiseCorr,4)),
                  linewidth = 2.0)
    r3 = mp.pyplot.bar(bar_width*2,signalCorr,bar_width/2,
                  color = 'b',
                  label =str(np.round(signalCorr,4)),
                  linewidth=2.0)
    #plt.plot(np.linspace(-0.1,0.4,3),np.zeros(3),'k--',linewidth=3.0)
    #plt.xlim(-0.05,0.3)
    #ax.get_xaxis().set_ticks(['total','noise','signal'])
    ax.set_xticks(np.arange(0,3)*bar_width + (bar_width/4))
    ax.set_xticklabels(('total','noise','signal'))
    #plt.ylim(-0.13,0.13)
    mp.pyplot.ylabel('correlation',fontsize=33)#,fontweight='bold')
    #plt.ylim(-4.0,4.0)
    mp.pyplot.legend()
    mp.pyplot.savefig('./Output/Activ/Correlation/'+path+'/Corr_Bar_'+manner+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def barPlot(totalCorr,signalCorr,noiseCorr,path,matter):
    nbrCells = np.shape(totalCorr)[0]
    for i in xrange(nbrCells):
        totalCorr[i,i] = np.nan
        signalCorr[i,i] = np.nan
        noiseCorr[i,i] = np.nan

    plotBar(np.nanmean(totalCorr),np.nanmean(signalCorr),np.nanmean(noiseCorr),path,'mean_'+matter)
    meanIdx = np.where(totalCorr == np.nanmax(totalCorr))
    #plotBar(totalCorr[meanIdx[0]],signalCorr[meanIdx[0]],noiseCorr[meanIdx[0]],'max_'+matter)
    #plotBar(np.nanmin(totalCorr),np.nanmin(signalCorr),np.nanmin(noiseCorr),'min_'+matter)
#------------------------------------------------------------------------------
def analyseCorrelation():

    if not os.path.exists('Output/Activ/Correlation/NoiseCorr'):
        os.mkdir('Output/Activ/Correlation/NoiseCorr')

    path ='NoiseCorr'
    tmRF = np.load('./work/TemplateMatch.npy')

##############################################################################
# Calculate total, signal and noise correlation like in recordings           #
# signal corrleation over shuffeling the repetitions per patch-id            #
# noise correlation = total correlation - signal correlation                 #
# for further information see Dadarlat and Stryker 2017                      #
##############################################################################


#####################################################
#       Correlation over tuning curves              #
#####################################################
#    contrLVL = 3

#    frExc = np.load('./work/TuningCurves_Exc.npy')
#    totalCorr_S,signalCorr_S,noiseCorr_S = calcCorrelaltions(frExc[:,:,:,contrLVL])

#    barPlot(np.copy(totalCorr_S),np.copy(signalCorr_S),np.copy(noiseCorr_S),path,'gabor')
#    plotRFtoSpkCorr(np.copy(tmRF),np.copy(totalCorr_S),path,'Gabor_TOTAL')
#    plotRFtoSpkCorr(np.copy(tmRF),np.copy(signalCorr_S),path,'Gabor_SIGNAL')
#    plotRFtoSpkCorr(np.copy(tmRF),np.copy(noiseCorr_S),path,'Gabor_NOISE')
    

#    frExc = np.load('./work/TuningCurves_sinus_Exc.npy')
#    totalCorr_S,signalCorr_S,noiseCorr_S = calcCorrelaltions(frExc[:,:,contrLVL,:])

#    barPlot(np.copy(totalCorr_S),np.copy(signalCorr_S),np.copy(noiseCorr_S),path,'sinus')
#    plotRFtoSpkCorr(np.copy(tmRF),np.copy(totalCorr_S),path,'Sinus_TOTAL')
#    plotRFtoSpkCorr(np.copy(tmRF),np.copy(signalCorr_S),path,'Sinus_SIGNAL')
#    plotRFtoSpkCorr(np.copy(tmRF),np.copy(noiseCorr_S),path,'Sinus_NOISE')

#####################################################
#       Correlation over natural scenes             #
#####################################################
    frExc = np.load('./work/fr_Repeats.npy')
    totalCorr,signalCorr,noiseCorr = calcCorrelaltions(frExc)
    plotRFtoSpkCorr(np.copy(tmRF),np.copy(totalCorr),path,'TMtoSPK_TOTAL')
    plotRFtoSpkCorr(np.copy(tmRF),np.copy(signalCorr),path,'TMtoSPK_SIGNAL')
    plotRFtoSpkCorr(np.copy(tmRF),np.copy(noiseCorr),path,'TMtoSPK_NOISE')

    barPlot(totalCorr,signalCorr,noiseCorr,path,'natural')



    corrRF = np.load('./work/correlationRF_V1.npy')    
    frExc = np.load('./work/frSingleSpkes.npy')
    #sinTC = np.load('./work/TuningCurves_sinus_Exc.npy')
    #orTC = np.load('./work/TuningCuver_sinus_orientation.npy')

    frExc = frExc.item()
    nbrOfNeurons = len(frExc)
    nbrOfPatches = 10000
    nbrOfTimeSteps = nbrOfPatches * 125
    spikeMatrix = np.zeros((nbrOfNeurons ,nbrOfTimeSteps))
    for i in xrange(nbrOfNeurons ):
        array = frExc[i]
        spikeMatrix[i,array] = 1
    print('calculate the correlation')

    spkCount = spiking.calcSpikeCount(spikeMatrix,nbrOfPatches)
    frEx = spiking.calcFR(spikeMatrix,nbrOfPatches)
    plotCorreNeurons(corrRF,spkCount,path)

    #spkCorr = calcSpikeCorrelation(spikeMatrix,nbrOfNeurons)
    #plotCorrelation(spkCorr,'Spk_10k_RMS')
    #rmsCorrSpk = calcRMSCorr(spkCorr)
    
    meanFR = np.mean(frEx)

    cntCorr = calcSpikeCorrelation(spkCount,nbrOfNeurons)
    np.save('./work/corrleationSpkCnt_V1',cntCorr)
    plotRFtoSpkCorr(np.copy(tmRF),np.copy(cntCorr),path,'TMtoSPK')
    plotCorrelation(np.copy(cntCorr),path,'Spk_Counts_10k')

    #corrTC = calcOrientCorr(sinTC,orTC)

    #plotRFtoSpkCorr(corrTC,cntCorr,meanFR,path,'TCtoSPK')
    
    rmsCorrSPKCount = calcRMSCorr(np.copy(cntCorr))
 
   # frCorr = calcSpikeCorrelation(frEx,path)
   # plotCorrelation(frCorr,path,'FR_10k')
   # rmsCorrFR = calcRMSCorr(frCorr)

    #stat={'RMSCorr_Spikes':rmsCorrSpk, 'RMSCorr_SpikeCounts':rmsCorrSPKCount}
    #json.dump(stat,open('./Output/ReceptiveFields/RMS_Corr.txt','w'))
    #print('plot orientation to correlation')

    gabParameters = np.load('./work/parameter.npy')
    orientations = gabParameters[:,1]
    plotSpkCorrOrient(cntCorr,orientations,path,'FR_GAB')

    tcParameters = np.load('./work/TuningCurve_ParamsExc.npy')
    orientations = tcParameters[:,1]
    plotSpkCorrOrient(cntCorr,orientations,path,'FR_TC')

    #print('start with MI')
    #matrixMI = calculateMISpk(spikeMatrix)
    #np.save('./work/MISpk.npy',matrixMI)
    #plotMIHist(matrixMI)
#------------------------------------------------------------------------------
def analysCorrelation2():
    if not os.path.exists('Output/Activ/Correlation/NoiseCorr_Stringer'):
        os.mkdir('Output/Activ/Correlation/NoiseCorr_Stringer')

    print('Calculate noise correlation after Stringer et al. 2016')

    path ='NoiseCorr_Stringer'
    tmRF = np.load('./work/TemplateMatch.npy')
##############################################################################
# Calculate total, signal and noise correlation for model simulations        #
# for further information see Stringer et al. 2016                           #
##############################################################################    
#####################################################
#       Correlation over tuning curves              #
#####################################################
    contrLVL = 3

    print('Start with tuning curves over idealized gabor')

    frExc = np.load('./work/TuningCurves_Exc.npy')
    noiseCorr_S =calcNoiseCorrelation(frExc[:,:,:,contrLVL])
    np.save('./work/noiseCorrel_Stringer_Gabor',noiseCorr_S)

    #barPlot(np.copy(totalCorr_S),np.copy(signalCorr_S),np.copy(noiseCorr_S),path,'gabor')
    #plotRFtoSpkCorr(np.copy(tmRF),np.copy(totalCorr_S),path,'Gabor_TOTAL')
    #plotRFtoSpkCorr(np.copy(tmRF),np.copy(signalCorr_S),path,'Gabor_SIGNAL')
    plotRFtoSpkCorr(np.copy(tmRF),np.copy(noiseCorr_S),path,'Gabor_NOISE')
    
    print('Start with tuning curves over sinus gratings')

    frExc = np.load('./work/TuningCurves_sinus_Exc.npy')
    noiseCorr_S = calcNoiseCorrelation(frExc[:,:,contrLVL,:])
    np.save('./work/noiseCorrel_Stringer_Sinus',noiseCorr_S)

    #barPlot(np.copy(totalCorr_S),np.copy(signalCorr_S),np.copy(noiseCorr_S),path,'sinus')
    #plotRFtoSpkCorr(np.copy(tmRF),np.copy(totalCorr_S),path,'Sinus_TOTAL')
    #plotRFtoSpkCorr(np.copy(tmRF),np.copy(signalCorr_S),path,'Sinus_SIGNAL')
    plotRFtoSpkCorr(np.copy(tmRF),np.copy(noiseCorr_S),path,'Sinus_NOISE')

#####################################################
#       Correlation over natural scenes             #
#####################################################

    print('Start with natural scenes')
    frExc = np.load('./work/fr_Repeats.npy')
    nbrCells = np.shape(frExc)[0]
    #noiseCorr =calcNoiseCorrelation(frExc)
    noiseCorr = np.load('./work/noiseCorrel_Stringer.npy')
    print(np.shape(noiseCorr))    
    np.save('./work/noiseCorrel_Stringer',noiseCorr)

    plotRFtoSpkCorr(np.copy(tmRF),np.copy(noiseCorr),path,'TMtoSPK_NOISE')

    for i in xrange(nbrCells):
        noiseCorr[i,i] = np.nan
    print(np.nanmean(noiseCorr))

#------------------------------------------------------------------------------   
if __name__=="__main__":
    if not os.path.exists('Output/Activ/Correlation/'):
        os.mkdir('Output/Activ/Correlation/')
    analyseCorrelation()
    #analysCorrelation2()

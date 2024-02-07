import numpy as np
import matplotlib as mp
mp.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
interpolation='nearest'#'bilinear'#'nearest'
import Gabor as gabor
from matplotlib.colors import LinearSegmentedColormap
# main function - createFields(File)
# function to Plot out of the feed forward weights 
# the receptive filds of neuron population, by shape the weights
# back in the input space

# weights matrices in './Input/', as .txt file
# Input parameter 'File' is the name of .txt file as string

# on/off weights can be plot separate
# additionaly plot a histogram over arrangement of weights
# and a mean receptive fild, over all neurons

# R. Larisch, Technische Universitaet Chemnitz
# 2015-06-23

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
#-------------------------------------------------------------------
def reshapeFields(w):
    nbrPost,nbrPre = np.shape(w)
    fieldSize = int(np.sqrt(nbrPre/2))
    print(fieldSize)
    fields = np.zeros((nbrPost,fieldSize,fieldSize,2))
    for i in range(nbrPost):
        field = w[i,:]
        fields[i] = np.reshape(field,(fieldSize,fieldSize,2))
    return(fields)
#-------------------------------------------------------------------
def getV1RFtoIN(fields,weightsV1toIN):
    nbrOfInhib,nbrOfExc =np.shape(weightsV1toIN)
    wMax = np.max(fields[:,:,:,0] - fields[:,:,:,1])
    wMin = np.min(fields[:,:,:,0] - fields[:,:,:,1])
    patchsize = np.shape(fields)[1]    
    nbrOfNeurons = int(nbrOfExc/100.0*20) #ca. 10% of excitatory neurons
    allocatedNeurons = np.zeros((nbrOfInhib,nbrOfNeurons))
    for i in range(10):
        fig = plt.figure()
        sortIndex = np.argsort(weightsV1toIN[i,:]*-1.0) #get Index sorted by weightsize
        meanField = np.zeros((patchsize,patchsize,2))
        for j in range(nbrOfNeurons):
            field = fields[sortIndex[j],:,:,:]
            meanField = meanField + field*weightsV1toIN[i,sortIndex[j]]
            field = field[:,:,0] - field[:,:,1]
            plt.subplot(1,nbrOfNeurons+2,j+1)
            im = plt.imshow(field,cmap=plt.get_cmap('gray'),interpolation=interpolation,vmin=wMin,vmax=wMax)
            plt.axis('off')
        meanField = meanField/np.float(nbrOfNeurons)
        meanField = meanField[:,:,0] - meanField[:,:,1]
        plt.subplot(1,nbrOfNeurons+2,nbrOfNeurons+2)
        im = plt.imshow(meanField,cmap=plt.get_cmap('gray'),interpolation=interpolation)#,vmin=wMin,vmax=wMax)
        plt.axis('off')
        plt.title('meanWeight')
        #print(weightsV1toIN[i,sortIndex])
        #print('------------------------------')
        plt.savefig('./Output/ReceptiveFields/V1toINFields_'+str(i)+'.png',bbox_inches='tight', pad_inches = 0.1)
        plt.figure()
        plt.plot(weightsV1toIN[i,sortIndex[0:nbrOfNeurons]],'-o')
        plt.savefig('./Output/ReceptiveFields/V1toINWeights_'+str(i)+'.png',bbox_inches='tight', pad_inches = 0.1)
#-------------------------------------------------------------------
def getV1RFtoINMean(fields,weightsV1toIN):
    nbrOfInhib,nbrOfExc =np.shape(weightsV1toIN)
    wMax = np.max(fields[:,:,:,0] - fields[:,:,:,1])
    wMin = np.min(fields[:,:,:,0] - fields[:,:,:,1])
    patchsize = np.shape(fields)[1]    
    nbrOfNeurons = int(nbrOfExc/100.0*10) #ca. 10% of excitatory neurons
    allocatedNeurons = np.zeros((nbrOfInhib,nbrOfNeurons))

    correlationList = np.array([])
    correlationDynList = []
    corrV1 = np.load('./work/correlationRF_V1.npy')
    for i in range(nbrOfExc):
        corrV1[i,i] = np.nan

    fig = plt.figure()
    for i in range(nbrOfInhib):
        sortIndex = np.argsort(weightsV1toIN[i,:]*-1.0) #get Index sorted by weightsize
        meanField = np.zeros((patchsize,patchsize,2))
        for j in range(nbrOfNeurons):
            field = fields[sortIndex[j],:,:,:]
            meanField = meanField + field*weightsV1toIN[i,sortIndex[j]]
        meanField = meanField/np.float(nbrOfNeurons)
        meanField = meanField[:,:,0] - meanField[:,:,1]
        plt.subplot(np.sqrt(nbrOfInhib),np.sqrt(nbrOfInhib),i+1)
        im = plt.imshow(meanField,cmap=plt.get_cmap('gray'),interpolation=interpolation)#,vmin=wMin,vmax=wMax)
        plt.axis('off')
        #print(weightsV1toIN[i,sortIndex])
        #print('------------------------------')
        plt.savefig('./Output/ReceptiveFields/V1toINFields_mean.png',bbox_inches='tight', pad_inches = 0.1)
        # identify the indicis of the pre-neurons, which have a strong connection to the post-neuron (higher than half of maximum)
        bestIndex = np.where(weightsV1toIN[i] >= (np.max(weightsV1toIN[i])*1.0/2.0))  
        # look, how is the correlation between all pre-neurons with a strong connection to the post-neuron
        # problem! sometimes, only one pre-neuron have a strong connection
        corr = (corrV1[bestIndex[0][0],bestIndex])
        correlationList = np.concatenate((correlationList,corr[~np.isnan(corr)]))
        correlationDynList.append(corr[~np.isnan(corr)])
    plt.figure()
    plt.hist(correlationList,15)
    plt.xlabel('correlation')
    plt.ylabel('number of synapses')
    plt.savefig('./Output/ReceptiveFields/V1toINFields_SynapsCorr.png',bbox_inches='tight', pad_inches = 0.1)

    weights = np.ones_like(correlationList)/float(len(correlationList))
    plt.figure()
    plt.hist(correlationList,bins=15,weights=weights)
    plt.xlabel('correlation')
    plt.ylabel('number of synapses')
    plt.savefig('./Output/ReceptiveFields/V1toINFields_SynapsCorrNORMED.png',bbox_inches='tight', pad_inches = 0.1)


    binSize = 15
    correl = np.zeros((nbrOfInhib,binSize+1))
    nbr = np.zeros((nbrOfInhib,binSize))
    for i in range(nbrOfInhib):
        nbr[i],correl[i] = np.histogram(correlationDynList[i],bins=binSize,range=(-0.6,0.6))

    ind = np.arange(binSize)   
    x = np.linspace(0,binSize,5)     
    plt.figure()
    plt.bar(ind,np.mean(nbr,axis=0))
    plt.xticks(x,np.linspace(correl[0,0],correl[0,binSize],5))
    plt.xlabel('correlation')
    plt.ylabel('average number of synapses')
    plt.savefig('./Output/ReceptiveFields/SynapsCorr.png',bbox_inches='tight', pad_inches = 0.1)

#-------------------------------------------------------------------
def calcGaborInhibOverV1Gabor(parameter,weightsV1toIN):
    nbrOfInhibN,nbrOfExcN = np.shape(weightsV1toIN)
    nbrOfConnections = 1#nbrOfExcN/100*1# 10 % of connections are included
    parameterInhib = np.zeros((nbrOfInhibN,np.shape(parameter)[1]))
    for i in range(nbrOfInhibN):
        sortIndex = np.argsort(weightsV1toIN[i,:]*-1.0)
        paramI = parameter[sortIndex[0:nbrOfConnections]]
        for j in range(nbrOfConnections):
            parameterInhib[i,:] += paramI[j]#*1.0/(j+1)
        parameterInhib[i,:] = parameterInhib[i,:]/ nbrOfConnections
    ix = np.argsort(weightsV1toIN[0,:]*-1.0)
    np.save('./work/parameter_Inhib',parameterInhib)
#-------------------------------------------------------------------
def plotONOFF(fields,manner):
    wMax = np.max(fields[:,:,:,0] - fields[:,:,:,1])
    wMin = np.min(fields[:,:,:,0] - fields[:,:,:,1])#0.0
    fig = plt.figure()
    x,y = setSubplotDimension(np.sqrt(np.shape(fields)[0]))
    for i in range(np.shape(fields)[0]):
        field = fields[i,:,:,0] - fields[i,:,:,1]
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(field,cmap=plt.get_cmap('gray'),aspect='auto',interpolation=interpolation,vmin=wMin,vmax=wMax)
        plt.axis('equal')
    fig.savefig('./Output/ReceptiveFields/ONOFFRF_'+manner+'.png',bbox_inches='tight', pad_inches = 0.1)
    print('On - Off finish')
#-------------------------------------------------------------------------
def plotMeanRF(fields):
    nbrOfNeurons=np.shape(fields)[0]
    rfMean = np.sum(fields,axis=0)/float(nbrOfNeurons)
    plt.figure()
    plt.imshow(rfMean[:,:,0] - rfMean[:,:,1],cmap=mp.cm.Greys_r,aspect='auto',interpolation=interpolation)
    plt.savefig('Output/ReceptiveFields/meanRF.png',bbox_inches='tight', pad_inches = 0.1)
#-------------------------------------------------------------------
def plotWeightHist(weights,path):
    nbrOfPostC,nbrOfPreC = np.shape(weights)
    plt.figure()
    plt.hist(np.reshape(weights,nbrOfPostC*nbrOfPreC),25)
    plt.xlabel('Weights')
    plt.savefig('Output/ReceptiveFields/HistWeights'+path+'.png',bbox_inches='tight', pad_inches = 0.1)
#-------------------------------------------------------------------
def plotBiDirectWeights(name,wTo,wFrom):
    nbr_Post,nbr_Pre = (np.shape(wTo))
    w1 = wTo
    w2 = wFrom.T
    w1 = np.reshape(w1,(nbr_Post*nbr_Pre))
    w2 = np.reshape(w2,(nbr_Post*nbr_Pre))
    plt.figure()
    plt.scatter(w1,w2)
    plt.xlabel(name)
    plt.ylabel('back weights')
    plt.savefig('Output/ReceptiveFields/weightsBiDirect_'+name+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist2d(w1,w2, bins=20,norm=LogNorm())
    plt.colorbar()
    plt.xlabel(name)
    plt.ylabel('back weights')    
    plt.savefig('Output/ReceptiveFields/weightsBiDirect2DHist_'+name+'.png',bbox_inches='tight', pad_inches = 0.1)
    
    plt.close('all')
#-------------------------------------------------------------------
def calcTemplateMatch(X,Y):
    tm = 0
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if (normX != 0 and normY !=0):
        tm = (np.dot(X,Y) / (normX*normY))
    return(tm)
#-------------------------------------------------------------------
def calcTMperCell(fields):
    nbrOfCells,patchSizeX,patchSizeY,deep = np.shape(fields)
    tm = np.zeros((nbrOfCells,nbrOfCells))
    for i in range(nbrOfCells):
        rf1 = fields[i,:,:,0] - fields[i,:,:,1]
        rf1 = np.reshape(rf1, patchSizeX*patchSizeY)
        for j in range(nbrOfCells):
            rf2 = fields[j,:,:,0] - fields[j,:,:,1]
            rf2 = np.reshape(rf2, patchSizeX*patchSizeY)
            tm[i,j] = calcTemplateMatch(rf1,rf2)
    return(tm)
#-------------------------------------------------------------------
def plotTemplateMatch(tmC):
    data       = tmC[0,:]
    y,binEdges = np.histogram(data,bins=15)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    menStd     = np.sqrt(y) 
    width      = 0.075
    plt.figure()
    plt.bar(bincenters, y, width=width, color='r', yerr=menStd)
    plt.savefig('Output/ReceptiveFields/TMFirstCellHist.png',bbox_inches='tight', pad_inches = 0.1)

    
    nbrOfCells = np.shape(tmC)[0]
    for i in range(nbrOfCells):
        tmC[i,i]= np.nan
    data = np.reshape(tmC,nbrOfCells*nbrOfCells)
    y,binEdges = np.histogram(data[~np.isnan(data)],bins=15)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    menStd     = np.sqrt(y) 
    width      = 0.075
    plt.figure()
    plt.bar(bincenters, y, width=width, color='r', yerr=menStd)
    plt.xlim(-1,1)
    plt.ylim(0,6000)
    plt.savefig('Output/ReceptiveFields/TMAllCellsHist.png',bbox_inches='tight', pad_inches = 0.1)

    plt.close('all')
#-------------------------------------------------------------------
def plotTMtoRF(tmC,fields):
    neuronNBR = 121
    print('Test Neuron: '+str(neuronNBR))
    wMax = np.max(fields[:,:,:,0] - fields[:,:,:,1])
    wMin = np.min(fields[:,:,:,0] - fields[:,:,:,1])#0.0
    tmArray = tmC[neuronNBR,:]
    indxArray = (np.argsort(tmArray*-1))

    plt.figure()
    plt.imshow(fields[neuronNBR,:,:,0] - fields[neuronNBR,:,:,1],cmap=plt.get_cmap('gray'),aspect='auto',interpolation=interpolation,vmin=wMin,vmax=wMax)
    plt.savefig('./Output/ReceptiveFields/RF_Nr'+str(neuronNBR)+'.png',bbox_inches='tight', pad_inches = 0.1)

    fig = plt.figure(figsize=(12,12))
    x,y = setSubplotDimension(np.sqrt(144))
    for i in range(len(indxArray)):
        field = fields[indxArray[i],:,:,0] - fields[indxArray[i],:,:,1]
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(field,cmap=plt.get_cmap('gray'),aspect='auto',interpolation=interpolation,vmin=wMin,vmax=wMax)
        plt.title("%.2f"%tmArray[indxArray[i]])
        plt.axis('equal')
    fig.savefig('./Output/ReceptiveFields/RFToTM_Nr'+str(neuronNBR)+'.png',bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
#-------------------------------------------------------------------
def calcRFCorrelation(fields):
    wMax = np.max(fields[:,:,:,0] - fields[:,:,:,1])
    wMin = np.min(fields[:,:,:,0] - fields[:,:,:,1])
    nbrOfNeurons,h,w,d = np.shape(fields)
    correlations = np.zeros((nbrOfNeurons,nbrOfNeurons))
    for i in range(nbrOfNeurons):
        fieldPost = fields[i,:,:,:]
        fieldPost = fieldPost[:,:,0] - fieldPost[:,:,1]
        fieldPost = np.reshape(fieldPost,h*w)
        for j in range(nbrOfNeurons):
            fieldPre = fields[j,:,:,:]
            fieldPre = fieldPre[:,:,0] - fieldPre[:,:,1]
            fieldPre = np.reshape(fieldPre,h*w)
            correlations[i,j] = np.corrcoef(fieldPost,fieldPre)[0,1]
    np.save('./work/correlationRF_V1',correlations)
    return(correlations)
#-------------------------------------------------------------------
def plotOrientationToCorrelation(corr,orientations,desc):
    nbrOfNeurons = np.shape(corr)[0]
    img = np.zeros((nbrOfNeurons,nbrOfNeurons))
    idxOrt = np.argsort(orientations)
    plt.close('all')

    #---define own colormap---#
    colors = [(0, 0, 0), (0.5,0.5, 0.5), (1, 1, 1)] # black->red->white
    n_bin = 5  # Discretizes the interpolation into bins
    cmap_name = 'my_color'
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    for i in range(nbrOfNeurons):
        for j in range(nbrOfNeurons):
            img[i,j] = corr[idxOrt[i],idxOrt[j]]

    nullfmt = mp.ticker.NullFormatter()         # no labels

    mp.pyplot.rc('font',weight = 'bold')
    mp.pyplot.rc('xtick',labelsize = 15)
    mp.pyplot.rc('ytick',labelsize = 15)

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.03

    rect_IMG = [left,bottom,width,height]
    rect_histX = [left,bottom_h,width,0.1]
    rect_histY = [left_h,bottom,0.1,height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8, 8))

    axIMG = plt.axes(rect_IMG)
    axHistX = plt.axes(rect_histX)
    axHistY = plt.axes(rect_histY)

    # no labels
    axHistX.xaxis.set_major_formatter(nullfmt)
    axHistY.xaxis.set_major_formatter(nullfmt)
    axHistY.yaxis.set_major_formatter(nullfmt)

    # plot with imshow
    ax1 = axIMG.imshow(img,cmap=plt.get_cmap('gray'),interpolation='none')#plt.get_cmap('gray')
    cbaxes = fig.add_axes([-0.05, 0.1, 0.03, 0.8])
    cb = plt.colorbar(ax1,cax=cbaxes)
    #cb.set_label('correlation')
    cbaxes.yaxis.set_ticks_position('left')
    axIMG.set_xlabel('neuron index',fontsize=18,weight='bold')
    axIMG.set_ylabel('neuron index',fontsize=18,weight='bold') #, ascending over orientation in degree

    # normal plots
    axHistX.plot(orientations[idxOrt],'o')
    axHistX.set_ylabel('Orientation',fontsize=15,weight='bold')
    axHistX.set_yticks(np.linspace(0,np.around(np.max(orientations),1),3))
    axHistX.set_yticklabels((r'$0.0$',r'$\pi$',r'$2\pi$'),fontsize=14)


    axHistY.plot(orientations[np.argsort(orientations*-1)],np.linspace(0,len(orientations),len(orientations)),'o')

    # set limits by hand
    axHistX.set_xlim((0,nbrOfNeurons))
    axHistX.set_ylim((0,np.around(np.max(orientations)+0.05,1) ))

    axHistY.set_xlim((0,np.around(np.max(orientations)+0.05,1) ))
    axHistY.set_ylim((0,nbrOfNeurons))
    axHistY.set_xticks(np.linspace(0,np.around(np.max(orientations),1),3))
    axHistY.set_xticklabels((r'$0.0$',r'$\pi$',r'$2\pi$'),fontsize=14)

    plt.savefig('./Output/ReceptiveFields/Correlation_Orientation_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
#------------------------------------------------------------------------------
def plotCorrHist(corrMatrix):
    nbrOfPost,nbrOfPre = np.shape(corrMatrix)
    for i in range(nbrOfPost):
        corrMatrix[i,i] = np.nan
    corrArray = np.reshape(corrMatrix, nbrOfPost*nbrOfPre)
    plt.figure()
    plt.hist(corrArray[~np.isnan(corrArray)],15)
    plt.xlim(-1,1)
    plt.ylim(0,6000)
    plt.savefig('./Output/ReceptiveFields/CorrelationRF_Hist.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotRFSize(rfSize):
    binSize = 5
    plt.figure()
    plt.hist(rfSize[:,0],binSize)
    plt.xlim(0,1.0)
    plt.xlabel('size along x-axis [pixel/patchsize]')
    plt.savefig('./Output/ReceptiveFields/RFSizeX_Hist.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(rfSize[:,1],binSize)
    plt.xlim(0,1.0)
    plt.xlabel('size along y-axis [pixel/patchsize]')
    plt.savefig('./Output/ReceptiveFields/RFSizeY_Hist.png',bbox_inches='tight', pad_inches = 0.1)


    plt.figure()
    plt.hist2d(rfSize[:,0],rfSize[:,1],bins=binSize,norm=LogNorm(),cmap = plt.cm.jet)
    plt.xlabel('size along x-axis [pixel/patchsize]')
    plt.ylabel('size along y-axis [pixel/patchsize]')
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)
    plt.colorbar()
    plt.savefig('./Output/ReceptiveFields/RFSize2D.png',bbox_inches='tight', pad_inches = 0.1)

    plt.close('all')
#------------------------------------------------------------------------------
def estimateRFSize(rFields):
    nbrCells,patchsize = np.shape(rFields)[0:2]
    rFields = rFields[:,:,:,0] - rFields[:,:,:,1]
    sizeRF = np.zeros((nbrCells,2)) 
    for j in range(nbrCells):
        maxPixel = np.max(np.abs(rFields[j]))
        idx = np.where(np.abs(rFields[j]) > maxPixel/2.0 )
        v1 = np.array((idx[0][0],idx[1][0]))
        maxLx = np.zeros(patchsize)
        maxLy = np.zeros(patchsize)
        for i in range(1,len(idx[0])-1):
            v2 = np.array((idx[0][i+1],idx[1][i+1]))
            if (v1[0] == v2[0]):
                l = np.linalg.norm(np.abs(v1-v2))
                if l >  maxLx[v1[0]]:
                    maxLx[v1[0]] = l
            if (v1[0] != v2[0]):
                v1 = np.array((idx[0][i+1],idx[1][i+1]))

        idxy = np.argsort(idx[1])
        xy = idx[1][idxy]
        xx = idx[0][idxy]

        v1 = np.array((xx[0],xy[0]))
        for i in range(1,len(xy)-1):
            v2 = np.array((xx[i+1],xy[i+1]))
            if (v1[1] == v2[1]):
                l = np.linalg.norm(np.abs(v1-v2))
                if l >  maxLy[v1[1]]:
                    maxLy[v1[1]] = l
            if (v1[1] != v2[1]):
                v1 = np.array((xx[i+1],xy[i+1]))
        sizeRF[j] = (np.max(maxLx),np.max(maxLy))

    return(sizeRF/patchsize)
#------------------------------------------------------------------------------
def plotFeedForwardInhibition(weightsV1toIN,weightsInhib):
    nbrInhibCells,nbrLGN = np.shape(weightsInhib)
    nbrExcCells = np.shape(weightsV1toIN)[1]
    arrayFF = np.reshape(weightsInhib,nbrInhibCells*nbrLGN)
    arrayEI = np.reshape(weightsV1toIN,nbrInhibCells*nbrExcCells)
    plt.figure()
    plt.hist(arrayFF,bins=20,histtype='step',label='LGN')
    plt.hist(arrayEI,bins=20,histtype='step',label='E to I')
    plt.legend()
    plt.savefig('./Output/ReceptiveFields/InhibFeedForward.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def createReceptiveFields():
    print('Start Allocation of exitatory RF to inhibitory Neuron')

    weightsV1 = np.loadtxt('Input_network/V1weight.txt')
    weightsInhib=np.loadtxt('Input_network/InhibW.txt')
    weightsV1toIN = np.loadtxt('Input_network/V1toIN.txt')
    weightsINtoV1 = np.loadtxt('Input_network/INtoV1.txt')
    weightsLatIN = np.loadtxt('Input_network/INLat.txt')
    gabParameters = np.load('./work/parameter.npy')
    #orientTC = np.load('./work/TuningCuver_orientation.npy')
    #frExcTC = np.load('./work/TuningCurve_frEx.npy')
    #orientTC_sinus = np.load('./work/TuningCuver_sinus_orientation.npy')
    #frExcTC_sinus = np.load('./work/TuningCurves_sinus_Exc.npy')
    nbrOfExc = np.shape(weightsV1)[0]
    nbrOfInh = np.shape(weightsInhib)[0]
    orExc = np.zeros(nbrOfExc)
    orExc_sinus = np.zeros(nbrOfExc)
    orInh = np.zeros(nbrOfInh)

    """
    # get the orientation of the maximum response
    for i in range(nbrOfExc):
        maxIdxExc = np.where(frExcTC[i] == np.max(frExcTC[i]))
        orExc[i] = orientTC[maxIdxExc[0][0],maxIdxExc[1][0],maxIdxExc[2][0]]
    #    maxIdxExc = np.where(frExcTC_sinus[i] == np.max(frExcTC_sinus[i]))
    #    orExc_sinus[i] = orientTC_sinus[maxIdxExc[0][0]]
    for i in range(nbrOfInh):
        maxIdxInh = np.where(frInhTC[i] == np.max(frInhTC[i]))
        orInh[i] = orientTC[maxIdxInh[0][0],maxIdxInh[1][0],maxIdxInh[2][0]]
    """
    plotFeedForwardInhibition(weightsV1toIN,weightsInhib)
    
    fields = reshapeFields(weightsV1)
    tmC = calcTMperCell(fields)
    np.save('./work/TemplateMatch.npy',tmC) 
    rfSize = estimateRFSize(fields)
    plotRFSize(rfSize)
    fieldsInhib = reshapeFields(weightsInhib)
    corr = calcRFCorrelation(fields)
    orientationsGab = gabParameters[:,1]
    plotOrientationToCorrelation(corr,orientationsGab,'GAB')
    plotOrientationToCorrelation(corr,orExc,'TC')
    corr = np.load('./work/corrleationSpkCnt_V1.npy')
    plotOrientationToCorrelation(corr,orExc,'TC_SPkCnt')
    #plotOrientationToCorrelation(corr,orExc_sinus,'TC_sinus')

    plotCorrHist(corr)



    plotTemplateMatch(tmC)
    plotTMtoRF(tmC,fields)
    getV1RFtoIN(fields,weightsV1toIN)
    getV1RFtoINMean(fields,weightsV1toIN)

    #gaborV1 = np.load('./work/parameter.npy')
    #calcGaborInhibOverV1Gabor(gaborV1,weightsV1toIN)    

    print('Plot bidirectional weight diagrams')
    plotBiDirectWeights('V1toIN',weightsV1toIN,weightsINtoV1) 
    plotBiDirectWeights('INLat',weightsLatIN,weightsLatIN)

    plotMeanRF(fields)
    plotONOFF(fields,'V1')
#    plotV1WeightHist(fields)   

    plotONOFF(fieldsInhib,'Inhib')


    plotWeightHist(weightsV1,'ExcV1')
    plotWeightHist(weightsInhib,'ExcInhib')
    plotWeightHist(weightsV1toIN,'V1toIn')
    plotWeightHist(weightsINtoV1,'IntoV1')
    plotWeightHist(weightsLatIN,'LatIn')
    
    print('Plotting of Receptive Fields finished!')
        
#-----------------------------------------------------
if __name__=="__main__":
    createReceptiveFields()

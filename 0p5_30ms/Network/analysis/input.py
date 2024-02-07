import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def calcCorr(frExc,n_patches):
    correlM = np.zeros((n_patches,n_patches))
    for i in xrange(n_patches):
        act_1 = frExc[:,i]
        for j in xrange(n_patches):
            act_2 = frExc[:,j]
            correlM[i,j] = np.corrcoef(act_1,act_2)[0,1]
    return(correlM)
#------------------------------------------------------------------------------
def startAnalysis():
    frExcV1 = np.load('./work/fluctuation_frExc.npy')
    input_patches = np.load('./work/fluctuation_Input.npy')
    input_TM = np.load('./work/fluctuation_tm_input.npy')

    n_neurons,n_patches,n_trials = np.shape(frExcV1)
    print(np.shape(input_patches))
    print(np.shape(input_TM)) 
    meanFr = np.mean(frExcV1,axis=2)
    #corr_perPatch = calcCorr(meanFr,n_patches)
    #np.save('./work/corr_perPatch.npy',corr_perPatch)
    corr_perPatch = np.load('./work/corr_perPatch.npy')
    plt.figure()
    for i in xrange(n_patches):
        idx = np.argsort(input_TM[i,:])
        plt.scatter(input_TM[i],corr_perPatch[i])
    plt.savefig('./Output/Fluctuation/inputToCorr.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    sortTM = np.zeros((n_patches,n_patches))
    sortCo = np.zeros((n_patches,n_patches))

    for i in xrange(n_patches):
        idx = np.argsort(input_TM[i])
        sortTM[i,:] = input_TM[i,idx]
        sortCo[i,:] = corr_perPatch[i,idx]

    xmin =-0.5 #-1.0
    xmax =0.5 #1.0
    ymin=-0.5 #-1.0
    ymax =0.5# 1.0

    binSize = 125#75#50
    heatAll = np.zeros((binSize,binSize))
    extent = []
    xedg = np.zeros(binSize)
    yedg = np.zeros(binSize)

    for i in xrange(n_patches):
        #heat,xedges,yedges = np.histogram2d(sortRF[i,~np.isnan(sortRF[i,:])],sortSP[i,~np.isnan(sortSP[i,:])],bins = binSize,range=([xmin,xmax],[ymin,ymax]))
        heat,xedges,yedges = np.histogram2d(sortTM[i],sortCo[i],bins = binSize,range=([xmin,xmax],[ymin,ymax]))        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xedg = xedges
        yedg = yedges
        heatAll +=heat

    my_cmap = plt.get_cmap('jet')
    my_cmap.set_under('w')

    plt.figure()
    plt.imshow(heatAll.T,extent=extent,origin='lower',interpolation='none',aspect='auto',cmap=my_cmap,vmin=0.0)#,norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('# of neuron pairs',fontsize=22)#,weight='bold')
    plt.xlabel('Input similarity',fontsize=22)
    plt.ylabel('correlation',fontsize=22)
    plt.xlim(-.5,.5)
    plt.ylim(-.5,.5)
    plt.savefig('./Output/Fluctuation/inputToCorr_HIST_0p5.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    binSize=51
    tm_range = np.linspace(-0.2,0.2,binSize)
    tm_points = np.zeros(binSize-1)
    meanCorr = np.zeros(binSize-1)
    stdCorr = np.zeros(binSize-1)
    elePerPoint = np.zeros(binSize-1)
    corrPoints = []
    for i in xrange(binSize-1):
        idx = (np.where((sortTM>=tm_range[i]) & (sortTM<tm_range[i+1])))  
        correlP = sortCo[idx[0],idx[1]]
        corrPoints.append(correlP)
        meanCorr[i] = np.mean(correlP)
        stdCorr[i] = np.std(correlP,ddof=1)
        elePerPoint[i] = len(correlP)
        tm_points[i] = np.mean([tm_range[i],tm_range[i+1]])

    plt.figure()
    #mp.pyplot.plot(tm_points[0:binSize-2],meanCorr[0:binSize-2],'-b')
    plt.errorbar(tm_points[0:binSize-2],meanCorr[0:binSize-2],yerr=stdCorr[0:binSize-2])
    plt.scatter(tm_points[0:binSize-2],meanCorr[0:binSize-2],c=elePerPoint[0:binSize-2],lw=0,alpha = 1.0,s=30.0)#,s=elePerPoint[0:sizeA-2]
    plt.colorbar()
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.5,0.5)
    plt.xlabel('RF similarity')
    plt.ylabel('spike count correlation')    
    plt.savefig('./Output/Fluctuation/inputToCorr_MEAN_Scatter_0p5.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    plt.rc('xtick',labelsize = 20)
    plt.rc('ytick',labelsize = 20)

    startAnalysis()

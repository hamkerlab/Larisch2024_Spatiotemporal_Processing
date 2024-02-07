import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

def plotMeans(meanOT,meanOC,desc):
    plt.figure()
    plt.plot(meanOT)
    plt.xlabel('number Of Steps')
    plt.ylabel('mean current ('+desc+')')
    plt.savefig('./Output/Activ/meanCurrentOT_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(meanOT,25,log=True)
    plt.xlabel('mean current ('+desc+')')
    plt.ylabel('number of steps')
    plt.savefig('./Output/Activ/meanCurrentOT_Hist_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(meanOC,'o')
    plt.xlabel('neuron index')
    plt.ylabel('mean current ('+desc+')')
    plt.savefig('./Output/Activ/meanCurrentOC_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(meanOC,25)
    plt.ylabel('number of neurons')
    plt.xlabel('mean current ('+desc+')')
    plt.savefig('./Output/Activ/meanCurrentOC_Hist_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.close('all')
#------------------------------------------------------------------------------
def plotExcInh(exc_Curr,inh_Curr,desc):
    if 'OT' in desc:
        fig,(ax1,ax2) = plt.subplots(2,sharex=True)
        ax1.plot(exc_Curr)
        ax1.set_ylabel('excitatory current')
        ax2.plot(inh_Curr)
        ax2.set_ylabel('inhibitory current')
        ax2.set_xlabel('time step')
        plt.savefig('./Output/Activ/meanCurrentBoth_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

        plt.figure()
        plt.plot(exc_Curr,label='g_Exc')
        plt.plot(exc_Curr-inh_Curr,label='g_Exc - g_Inh')
        plt.plot(inh_Curr*-1,label='-g_Inh')
        plt.legend()
        plt.xlabel('input index')
        plt.ylabel('synapse currents [nA]')
        plt.savefig('./Output/Activ/BothCurrentsOT_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    if 'OC' in desc:
        fig,(ax1,ax2) = plt.subplots(2,sharex=True)
        ax1.plot(exc_Curr,'o')
        ax1.set_ylabel('excitatory current')
        ax2.plot(inh_Curr,'o')
        ax2.set_ylabel('inhibitory current')
        ax2.set_xlabel('neuron index')
        plt.savefig('./Output/Activ/meanCurrentBoth_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def calcCorrelation(exc_Curr,inh_Curr):
    nbrOfSteps,nbrOfCells = np.shape(exc_Curr)
    corrOT = np.zeros(nbrOfSteps)
    corrOC = np.zeros(nbrOfCells)

    #for i in range(nbrOfSteps):
    #    corrOT[i] = np.corrcoef(exc_Curr[i],inh_Curr[i])[0,1]

    for i in range(nbrOfCells):
        corrOC[i] = np.corrcoef(exc_Curr[:,i],inh_Curr[:,i])[0,1]
    return(corrOT,corrOC)
#------------------------------------------------------------------------------
def plotBars(gExc,gInh,manner):
    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 18)
    plt.rc('ytick',labelsize = 22)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar_width = 0.1
    r1 = plt.bar(0.0 ,gExc,bar_width/2,
                  color = 'r',
                  label = r'$\mathbf{\bar g_{Exc}} $',
                  linewidth=2.0)
    r2 = plt.bar(bar_width,gExc-gInh,bar_width/2,
                  color = 'g',
                  label = r'$\mathbf{ \bar g_{Exc} - \bar g_{Inh}} $',
                  linewidth = 2.0)
    r3 = plt.bar(bar_width*2,gInh*-1,bar_width/2,
                  color = 'b',
                  label = r'$\mathbf{- \bar g_{Inh}} $',
                  linewidth=2.0)
    plt.plot(np.linspace(-0.1,0.4,3),np.zeros(3),'k--',linewidth=3.0)
    plt.xlim(-0.05,0.3)
    ax.get_xaxis().set_ticks([])
    #plt.ylim(-0.13,0.13)
    plt.ylabel('Currents [nA]',fontsize=33,fontweight='bold')
    #plt.ylim(-4.0,4.0)
    #plt.legend()
    plt.savefig('./Output/Activ/Bars_'+manner+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotCorrelation(corrOT,corrOC,desc):
#    plt.figure()
#    plt.hist(corrOT)
#    plt.savefig('./Output/Activ/currentCorrOTHist_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

#    plt.figure()
#    plt.plot(corrOT)
#    plt.xlabel('time steps')
#    plt.savefig('./Output/Activ/currentCorr_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(corrOC)
    plt.xlabel('correlation')
    plt.ylabel('number of cells')
    plt.savefig('./Output/Activ/currentCorrOCHist_'+desc+'.png',bbox_inches='tight', pad_inches = 0.1)
    
#------------------------------------------------------------------------------
def plotCrossCorrelation(CrossCor,CrossSelf,CrossCorNorm,d,duration):
    print('start plotting cross correlation')
    #print(np.shape(CrossCor))

    meanCrossCor = np.mean(CrossCor,axis=1)
    meanCrossCorNorm = np.mean(CrossCorNorm,axis=1)
    meanCrossSelf = np.mean(CrossSelf,axis=1)

    nbrCells,length = np.shape(meanCrossCor)

    plt.figure()
    plt.plot(np.mean(meanCrossCor,axis=0),'-+')
    #plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.ylim(0.0,1.0)
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossMeans.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(np.mean(meanCrossCor,axis=0),'-+')
    plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(duration -d,duration +d,5),np.linspace(-d,d,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.ylim(0.0,1.0)
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossMeans_Zoom.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(np.mean(meanCrossCorNorm,axis=0),'-+')
    #plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    #plt.ylim(0.0,0.011)
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossMeans_norm.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(np.mean(meanCrossCorNorm,axis=0),'-+')
    plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(duration -d,duration +d,5),np.linspace(-d,d,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    #plt.ylim(0.0,0.011)
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossMeans_normZoom.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.plot(meanCrossSelf[0],'-+')
    plt.xlim(duration -d,duration +d)
    #plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossMean_Self.png',bbox_inches='tight', pad_inches = 0.1)

    error = (np.min(meanCrossCor,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossError_Cells.png',bbox_inches='tight', pad_inches = 0.1)

    error = (np.min(meanCrossCor,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(duration -d,duration +d,5),np.linspace(-d,d,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossError_Cells_zoom.png',bbox_inches='tight', pad_inches = 0.1)


    error = (np.min(meanCrossCorNorm,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossNormError_Cells.png',bbox_inches='tight', pad_inches = 0.1)

    error = (np.min(meanCrossCorNorm,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(duration -d,duration +d,5),np.linspace(-d,d,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossNormError_Cells_zoom.png',bbox_inches='tight', pad_inches = 0.1)

    #-------------------------------------------#

    meanCrossCor = np.mean(CrossCor,axis=0)
    meanCrossCorNorm = np.mean(CrossCorNorm,axis=0)
    meanCrossSelf = np.mean(CrossSelf,axis=0)

    patches = np.shape(meanCrossCor)[0]
    
    error = (np.min(meanCrossCor,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossError_Patches.png',bbox_inches='tight', pad_inches = 0.1)

    error = (np.min(meanCrossCor,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(duration -d,duration +d,5),np.linspace(-d,d,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossError_Patches_zoom.png',bbox_inches='tight', pad_inches = 0.1)
    
    error = (np.min(meanCrossCorNorm,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xticks(np.linspace(0,duration*2-1,5),np.linspace(-duration,duration,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossNormError_Patches.png',bbox_inches='tight', pad_inches = 0.1)

    error = (np.min(meanCrossCorNorm,axis=0),np.max(meanCrossCor,axis=0))
    x = np.arange(0,length)
    plt.figure()
    plt.errorbar(x=x,y=np.mean(meanCrossCor,axis=0),yerr=error,fmt='-+')
    plt.xlim(duration -d,duration +d)
    plt.xticks(np.linspace(duration -d,duration +d,5),np.linspace(-d,d,5))
    plt.axvline(x=duration,ymin=0.0,ymax=1.0,ls='dashed',color='k')
    plt.xlabel('lag [ms]')
    plt.savefig('./Output/Activ/currentCrossNormError_Patches_zoom.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def meanPerBin(data,timeBin):
    nbrSteps,nbrCells = np.shape(data)
    nbrPatches = nbrSteps/timeBin
    print(nbrPatches)
    m_data = np.zeros((nbrPatches,nbrCells))
    for i in range(nbrCells):
        t= np.split(data[:,i],nbrPatches)
        m_data[:,i] = np.mean(t,axis=1)
    return(m_data)
#------------------------------------------------------------------------------
def startAnalyseCurrents():
    exc_gExc = np.load('./work/currents_Exc_gExc.npy')
    exc_gInh = np.load('./work/currents_Exc_gInh.npy')
    inh_gExc = np.load('./work/currents_Inh_gExc.npy')
    inh_gInh = np.load('./work/currents_Inh_gInh.npy')
    
    timeBin = 125
    nbrSteps,nbrCells = np.shape(exc_gExc)

    gExc_t = meanPerBin(exc_gExc,timeBin)
    gInh_t = meanPerBin(exc_gInh,timeBin)

    #gExc_t = gExc_t[0:1000]
    #gInh_t = gInh_t[0:1000]
    n_P,n_C = np.shape(gExc_t)

    # 8000 patches of natural szenes, splitted in 8x1000 bins
    # calculate the emergent exc and inhibit. current per neuron is emergent on a specific patch

    binSize =15

    gExc_A = np.reshape(gExc_t,n_P*n_C)
    gInh_A = np.reshape(gInh_t,n_P*n_C)
    min_ex = 0.0
    max_ex = np.ceil(np.max(gExc_t))
    gExc_bins = np.linspace(min_ex,max_ex,binSize)
    gInh_v = []
    gInh_mean= []
    gInh_std = []
    gExc_v = np.zeros(binSize-1)
    n_ele = np.zeros(binSize-1)
    for i in range(binSize-1):
        idx = np.where((gExc_A > gExc_bins[i]) & (gExc_A <= gExc_bins[i+1]))
        n_ele[i]=np.shape(idx)[1]
        gInh_v.append(gInh_A[idx[0]])
        #gInh_mean.append(np.mean(gInh_A[idx[0]]))
        #gInh_std.append(np.std(gInh_A[idx[0]],ddof=1))

    plt.figure()
    for i in range(binSize-1):
        plt.scatter(gExc_bins[i],np.mean(gInh_v[i]),color='tomato')
    plt.xlabel('exc. current')
    plt.ylabel('inh. current')
    plt.savefig('./Output/Activ/currentScatter_Mean.png',bbox_inches='tight', pad_inches = 0.1)


    gExc_m = np.mean(gExc_t,axis=1)
    gInh_m = np.mean(gInh_t,axis=1)
    min_ex = 0.0
    max_ex = (np.max(gExc_m))
    gExc_bins = np.linspace(min_ex,max_ex,binSize)
    gInh_v = []
    gInh_mean= []
    gInh_std = []
    gExc_v = np.zeros(binSize-1)
    n_ele = np.zeros(binSize-1)
    for i in range(binSize-1):
        idx = np.where((gExc_m > gExc_bins[i]) & (gExc_m <= gExc_bins[i+1]))
        n_ele[i]=np.shape(idx)[1]
        gInh_v.append(gInh_m[idx[0]])
        gInh_mean.append(np.mean(gInh_m[idx[0]]))
        gInh_std.append(np.std(gInh_m[idx[0]]))


    plt.figure()
    for i in range(binSize-1):
        plt.scatter(gExc_bins[i],np.mean(gInh_v[i]),color='tomato')
    plt.xlabel('exc. current')
    plt.ylabel('inh. current')
    plt.savefig('./Output/Activ/currentScatter_MeanOverCells.png',bbox_inches='tight', pad_inches = 0.1)

    print(np.shape(gExc_m))

    binSizePatches = 4
    nbPatch = n_P/binSizePatches
    new_gExc = np.split(gExc_m,binSizePatches)
    new_gInh = np.split(gInh_m,binSizePatches)
    gInh_v = np.zeros((binSizePatches,binSize-1))
    gExc_v = np.zeros((binSizePatches,binSize-1))
    gInh_std = np.zeros((binSizePatches,binSize-1))
    n_ele = np.zeros((binSizePatches,binSize-1))
    for s in range(binSizePatches):
        m_gExc = new_gExc[s]
        m_gInh = new_gInh[s]
        for i in range(binSize-1):
            idx = np.where((m_gExc > gExc_bins[i]) & (m_gExc <= gExc_bins[i+1]))
            n_ele[s,i] = np.shape(idx)[1]
            gExc_v[s,i]= np.mean(m_gExc[idx])
            gInh_v[s,i]= np.mean(m_gInh[idx])
            #gInh_std[s,i] = np.std(m_gInh[idx])

    gExc_A = np.reshape(gExc_v,binSizePatches*(binSize-1))
    gInh_A = np.reshape(gInh_v,binSizePatches*(binSize-1))

    plt.figure()
    plt.plot(gExc_A,gInh_A,'o',color='seagreen')
    plt.savefig('./Output/Activ/currentScatter_BinPatches.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.errorbar(np.mean(gExc_v,axis=0),np.mean(gInh_v,axis=0),yerr=np.std(gInh_v,axis=0),color='seagreen')
    plt.savefig('./Output/Activ/currentScatter_BinPatchesErrorBars.png',bbox_inches='tight', pad_inches = 0.1)

    return -1

    steps = 1250
    duration = 125
    d = 40
    nbrStimulie = steps/duration

    meanCrossCor = np.zeros((nbrCells,nbrStimulie,duration*2 -1))
    meanCrossSelf = np.zeros((nbrCells,nbrStimulie,duration*2 -1))
    meanCrossCorNorm = np.zeros((nbrCells,nbrStimulie,duration*2 -1))
    for j in xrange(nbrCells):
        exc = exc_gExc[0:steps,j]#/np.max(exc_gExc[0:steps,0])
        inh = exc_gInh[0:steps,j]#/np.max(exc_gInh[0:steps,0])#

        splitExc = np.split(exc,nbrStimulie)
        splitInh = np.split(inh,nbrStimulie)
        norm = np.concatenate((range(1,duration+1),range(duration-1,0,-1)))
        for i in xrange(nbrStimulie):
            meanCrossCor[j,i,:] = np.correlate(splitExc[i],splitInh[i],mode='full')
            meanCrossCorNorm[j,i,:] = meanCrossCor[j,i]/norm

    plotCrossCorrelation(meanCrossCor,meanCrossSelf,meanCrossCorNorm,d,duration)

    meanOT_gExc_E = np.mean(exc_gExc,axis=1)
    meanOT_gInh_E = np.mean(exc_gInh,axis=1)
    meanOT_gExc_I = np.mean(inh_gExc,axis=1)
    meanOT_gInh_I = np.mean(inh_gInh,axis=1)

    meanOC_gExc_E = np.mean(exc_gExc,axis=0)
    meanOC_gInh_E = np.mean(exc_gInh,axis=0)
    meanOC_gExc_I = np.mean(inh_gExc,axis=0)
    meanOC_gInh_I = np.mean(inh_gInh,axis=0)
    
    meanGExc_E = np.mean(exc_gExc)
    meanGInh_E = np.mean(exc_gInh)
    meanGExc_I = np.mean(exc_gExc)
    meanGInh_I = np.mean(inh_gInh)

    stats ={'gExc_Sum_E':np.sum(exc_gExc),'gInh_Sum_E':np.sum(exc_gInh),
            'gExc_Sum_I':np.sum(inh_gExc),'gInh_Sum_I':np.sum(inh_gInh),
            'gExc_Mean_E':np.mean(exc_gExc),'gInh_Mean_E':np.mean(exc_gInh),
            'gExc_Mean_I':np.mean(inh_gExc),'gInh_Mean_I':np.mean(inh_gInh)} 
    json.dump(stats,open('./Output/Activ/Currents.txt','w'))

    plotMeans(meanOT_gExc_E,meanOC_gExc_E,'gExc_E')
    plotMeans(meanOT_gInh_E,meanOC_gInh_E,'gInh_E')
    plotMeans(meanOT_gExc_I,meanOC_gExc_I,'gExc_I')
    plotMeans(meanOT_gInh_I,meanOC_gInh_I,'gInh_I')

    plotBars(np.mean(exc_gExc),np.mean(exc_gInh),'meanExcL')
    plotBars(np.mean(inh_gExc),np.mean(inh_gInh),'meanInhL')

    plotExcInh(meanOT_gExc_E,meanOT_gInh_E,'OT_E')
    plotExcInh(meanOC_gExc_E,meanOC_gInh_E,'OC_E')

    plotExcInh(meanOT_gExc_I,meanOT_gInh_I,'OT_I')
    plotExcInh(meanOC_gExc_I,meanOC_gInh_I,'OC_I')

    corrOverTime,corrOverCells = calcCorrelation(exc_gExc,exc_gInh)
    plotCorrelation(corrOverTime,corrOverCells,'Exc')

    corrOverTime,corrOverCells = calcCorrelation(inh_gExc,inh_gInh)
    plotCorrelation(corrOverTime,corrOverCells,'Inh')
#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyseCurrents()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plotSingleCell(gExc,gInh,spikes):
    spikTimes = (np.ones(len(spikes)) * np.max(gInh)+ 0.25) *-1 
    plt.figure(figsize=(14,4))
    plt.plot(gExc,label = r'$\mathbf{g_{Exc}} $',linewidth=2.0)
    plt.plot(gInh*-1,label = r'$\mathbf{g_{Inh}} $',linewidth=2.0)
    plt.plot(gExc-gInh,label = r'$\mathbf{ g_{Exc} - g_{Inh}} $',linewidth=2.0)
    #plt.plot(spikes,spikTimes,'^',markerfacecolor='None',color='black',markeredgewidth=1.5)
    plt.legend(bbox_to_anchor=(1.0, 0.6),borderpad=0.1,labelspacing=0.1)
    plt.xlabel('Time [ms]',fontsize=14,fontweight='bold')
    plt.ylabel('Currents [nA]',fontsize=14,fontweight='bold')
    #plt.ylim(-1.5,1.0)
    plt.savefig('./Output/V1Layer/single.jpg',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotTotal(gExc,gInh,spikes):
    steps,nbrOfNeurons = np.shape(gExc)
    totalgExc = np.sum(gExc,axis=1)
    totalgInh = np.sum(gInh,axis=1)
    totalSpikes=[]
    for i in xrange(nbrOfNeurons):
        totalSpikes = totalSpikes + spikes[i]
    spikTimes = (np.ones(len(totalSpikes)) * np.max(totalgInh)+ 0.25) *-1
    plt.figure(figsize=(8,2.5))
    plt.plot(totalgExc,label = r'$\mathbf{g_{Exc}} $',linewidth=2.0)
    plt.plot(totalgInh*-1,label = r'$\mathbf{g_{Inh}} $',linewidth=2.0)
    plt.plot(totalgExc-totalgInh,label = r'$\mathbf{ g_{Exc} - g_{Inh}} $',linewidth=2.0)
    plt.plot(totalSpikes,spikTimes,'^',markerfacecolor='None',color='black',markeredgewidth=1.5)
    plt.legend(bbox_to_anchor=(1.0, 0.6),borderpad=0.1,labelspacing=0.1)
    plt.xlabel('Time [ms]',fontsize=14,fontweight='bold')
    plt.ylabel('Currents [nA]',fontsize=14,fontweight='bold')
    plt.savefig('./Output/V1Layer/total.jpg',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
def plotMean(gExc,gInh,spikes):
    steps,nbrOfNeurons = np.shape(gExc)
    meangExc = np.mean(gExc,axis=1)
    meangInh = np.mean(gInh,axis=1)
    meanSpikes=[]
    for i in xrange(nbrOfNeurons):
        meanSpikes = meanSpikes + spikes[i]
    spikTimes = np.ones(len(meanSpikes)) *-1.5 #* np.max(meangInh)+ 0.25) *-1
    plt.figure(figsize=(8,2.5))
    plt.plot(meangExc,label = r'$\mathbf{g_{Exc}} $',linewidth=2.0)
    plt.plot(meangInh*-1,label = r'$\mathbf{g_{Inh}} $',linewidth=2.0)
    plt.plot(meangExc-meangInh,label = r'$\mathbf{ g_{Exc} - g_{Inh}} $',linewidth=2.0)
    plt.plot(meanSpikes,spikTimes,'^',markerfacecolor='None',color='black',markeredgewidth=1.5)
    plt.legend(bbox_to_anchor=(1.0, 0.6),borderpad=0.1,labelspacing=0.1)
    plt.xlabel('Time [ms]',fontsize=14,fontweight='bold')
    plt.ylabel('Currents [nA]',fontsize=14,fontweight='bold')
    plt.ylim(-1.55,0.5)
    plt.savefig('./Output/V1Layer/mean2.jpg',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotMean2(gExc,gInh,spikes):
    steps,nbrOfNeurons = np.shape(gExc)
    meangExc = np.mean(gExc,axis=1)
    meangInh = np.mean(gInh,axis=1)
    meanSpikes=[]
    for i in xrange(nbrOfNeurons):
        meanSpikes = meanSpikes + spikes[i]
    spikTimes = np.ones(len(meanSpikes)) *-1.5 #* np.max(meangInh)+ 0.25) *-1


    fig = plt.figure(figsize=((13,4)))
    ax0 = plt.subplot2grid((5,3),(0,0),colspan=3,rowspan=3)
    ax1 = plt.subplot2grid((5,3),(3,0),colspan=3,rowspan=2)
    #f,ax =  plt.subplots(2,sharex = True, figsize=(8,2.5))
    
    ax0.plot(meangExc,label = r'$\mathbf{g_{Exc}} $',linewidth=3.5,color='r')
    ax0.plot(meangInh*-1,label = r'$\mathbf{g_{Inh}} $',linewidth=3.5,color='b')
    ax0.plot(meangExc-meangInh,label = r'$\mathbf{ g_{Exc} - g_{Inh}} $',linewidth=3.5,color='g')
    #ax0.plot(meanSpikes,spikTimes,'^',markerfacecolor='None',color='black',markeredgewidth=1.5)
    #ax0.legend(bbox_to_anchor=(1.0, 0.6),borderpad=0.1,labelspacing=0.1)
    ax0.set_ylabel('Currents [nA]',fontsize=24,fontweight='bold')
    #ax0.set_ylim(-1.35,0.5)
    ax0.set_xlim(0,len(meangExc))
    ax0.get_xaxis().set_ticks([])
    
    #ax1.plot(meanSpikes,spikTimes,'^',markerfacecolor='None',color='black',markeredgewidth=1.5)
    ax1.plot(meanSpikes,np.linspace(0,nbrOfNeurons,len(meanSpikes)),'.',color='black')
    ax1.set_xlabel('Time [ms]',fontsize=24,fontweight='bold')
    #ax1.get_yaxis().set_ticks([])
    ax1.set_ylabel(r'$ \mathbf{n_i}$',fontsize=30)
    ax1.get_yaxis().set_ticks(np.linspace(0,nbrOfNeurons,3))
    ax1.set_xlim(0,len(meangExc)) 
    plt.savefig('./Output/V1Layer/mean.jpg',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def startAnalyse():

    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 18)
    plt.rc('ytick',labelsize = 18)

    gExc = np.load('./work/DATA_gExcV1.npy')
    gInh = np.load('./work/DATA_gInhV1.npy')
    spikes = np.load('./work/DATA_frV1.npy')
    spikes = spikes.item()
    neuronIndex = 3
    steps,nbrOfNeurons = np.shape(gExc)
    plotSingleCell(gExc[:,neuronIndex],gInh[:,neuronIndex],spikes[neuronIndex])
    plotTotal(gExc,gInh,spikes)
    plotMean(gExc,gInh,spikes)
    plotMean2(gExc,gInh,spikes)
#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyse()

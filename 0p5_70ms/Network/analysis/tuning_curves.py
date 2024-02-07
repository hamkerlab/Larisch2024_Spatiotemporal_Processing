import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
    # function to calculate the orientation selectivity index (OSI)
    # see Ringach et al. (2002)
def calcOSI(circVar):
    return(1-circVar)
#------------------------------------------------------------------------------
    # function to calculate the circular Variance
    # see Ringach et al. (2002)
    # over the amount of the complex part
def calcCircularVarianceRingach(tuningCurves,minAngle,maxAngle):
    nbrOfCells,nbrOfSteps = np.shape(tuningCurves)
    orientations = np.linspace(minAngle,maxAngle,nbrOfSteps)
    circVar1 = np.zeros(nbrOfCells)
    circVar2 = np.zeros(nbrOfCells)
    tm = 2.0*np.pi/180.0
    for i in range(nbrOfCells):
        #circVar1[i] = np.sum(tuningCurves[i]*np.exp(2j*orientations))/np.sum(tuningCurves[i])
        x = tuningCurves[i]*np.cos(tm*orientations)
        y = tuningCurves[i]*np.sin(tm*orientations)
        circVar2[i] = np.sqrt(np.sum(x)**2 + np.sum(y)**2) / np.sum(tuningCurves[i])
    return(1 - circVar2)
#------------------------------------------------------------------------------
'''
float circular_variance_tuning_curve(r,theta,n,theta_max)
     float *r;         // response amplitude [n]
     float *theta;     // angle 0..theta_max (deg) [n]
     int n;            // number of data points
     float theta_max;  // e.g., 360.0 for dir, 180.0 for ori
{
  int i;
  float tot,x,y,cv,tm2pi;

  tm2pi = 2.0 * M_PI / theta_max;

  tot = x = y = 0.0;
  for(i=0;i<n;i++){
    tot += amp[i];
    x += amp[i] * cos(tm2pi * theta[i]);
    y += amp[i] * sin(tm2pi * theta[i]);
  }
  cv = 1.0 - sqrt(x*x + y*y) / tot;

  return cv;
}'''
#------------------------------------------------------------------------------
    # calc circular variance for sinus gratings after Ringach 2002!
    # over the mean respons per angle in the input
def calcCVSinus(tuningCurves,minAngle,maxAngle):
    nbrCells, nbrAngle, nbrRepeats = np.shape(tuningCurves)
    angls = np.linspace(minAngle,maxAngle,nbrAngle)*np.pi/180
    cv = np.zeros(nbrCells)
    for i in range(nbrCells):
        meanFR = np.mean(tuningCurves[i,:,:],axis=1)
        if (np.sum(meanFR) > 0.0):
            cv[i] = np.abs(np.sum(meanFR*np.exp(2j*angls))/np.sum(meanFR))
    return(1- cv)
#------------------------------------------------------------------------------
    # aproximation of the circular variance with centred tuning curves
    # see Ringach et al.(2002)
    # for tuning curves, estimated over sinus/cosinus gratings
    # calculating the baseline(baseL), after Sadeh et al. 2014
    # TODO: make it work for gabor estimatet tuning curves
def aproxCircularVariance(tuningCurves,minAngle,maxAngle):
    nbrOfCells,nbrOfSteps = np.shape(tuningCurves)
    orientations = np.linspace(minAngle,maxAngle,nbrOfSteps+1)
    circVar = np.zeros(nbrOfCells)
    for i in range(nbrOfCells):
        baseL = np.mean(tuningCurves[i])
        idx = np.where(tuningCurves[i] >= baseL)
        b1 = np.abs(orientations[idx[0][0]])
        b2 = np.abs(orientations[idx[0][len(idx[0])-1]])
        bMean = (b1+b2)/2.0
        c = baseL/np.max(tuningCurves[i])
        circVar[i] = (np.sin(bMean)**2 /(bMean**2 + 2*np.pi*bMean*c))
    return( 1 - circVar)
#------------------------------------------------------------------------------
def calcOSISadeh(tuningCurves,minAngle,maxAngle):
    #print(np.shape(tuningCurves))
    nbrOfCells,nbrOfSteps = np.shape(tuningCurves)
    orientations = np.linspace(minAngle,maxAngle,nbrOfSteps+1)
    orienSteps = np.abs(orientations[1] - orientations[0])
    circVar = np.zeros(nbrOfCells)
    for i in range(nbrOfCells):
        r_pref = np.max(tuningCurves[i])
        idx_pref = np.where(tuningCurves[i] == np.max(tuningCurves[i]))
        idx_pref = int(np.mean(idx_pref[0]))
        idx_orth = idx_pref - 90.0/orienSteps
        r_orth = tuningCurves[i,int(idx_orth)]
        circVar[i] = (r_pref - r_orth) / (r_pref + r_orth)
    return(circVar)
#------------------------------------------------------------------------------
    # orientation Bandwidth after Ringach et al.(2002)
def calculateOrientationBandwidth(tuningCurves,minAngle,maxAngle):
    nbrOfCells,nbrOfSteps = np.shape(tuningCurves)
    orientations = np.linspace(minAngle,maxAngle,nbrOfSteps+1)    
    bW = np.zeros(nbrOfCells)
    for i in range(nbrOfCells):
        baseL = np.mean(tuningCurves[i])
        idx = np.where(tuningCurves[i] >= baseL)
        b1 = np.abs(orientations[idx[0][0]])
        b2 = np.abs(orientations[idx[0][len(idx[0])-1]])
        bMean = (b1+b2)/2.0
        c = baseL/np.max(tuningCurves[i])
        bW[i] = bMean*(1+c)*(1-(1/np.sqrt(2)))
    return(bW)
#------------------------------------------------------------------------------
def estimateOrientationBandwidth(relativTCExc,minAngle,maxAngle,path):
    # algorithm to estimate the OBW:
    # get the index of the first point, what is over 1/sqrt(2)
    # get the index of the first pont, what is below 1/sqrt(2)
    # calculate the mean of both to get an aproximated value for the OBW
    # because not all TCs are symmetrical 
    # -> calculate OBW for the left and the right edge of the TC and take the mean 

    nbrOfNeurons,nbrOfSteps = np.shape(relativTCExc)
    orientations = np.linspace(minAngle,maxAngle,nbrOfSteps+1)
    orienSteps = np.abs(orientations[1] - orientations[0])
    halfO = np.zeros(nbrOfNeurons)
    for i in range(nbrOfNeurons):
        maxFr = np.max(relativTCExc[i])
        idxMax= np.where(relativTCExc[i] == maxFr)[0]
        idxMax = (int(np.mean(idxMax)))        
        idxMaxHalf = np.asarray(np.where(relativTCExc[i] >= (maxFr/np.sqrt(2.0))))[0]
        idxMinHalf = np.asarray(np.where(relativTCExc[i] <= (maxFr/np.sqrt(2.0))))[0]
        
        # TC with high frequency and periodic asymptote, the boarders can be again over 1/sqrt(2)
        # look for the point under 1/sqrt(2) what is nearest on the maximum, for the left edge!
        diff_Idx = idxMax - idxMinHalf

        if (len(diff_Idx[diff_Idx>0]) >0): # check if a left edge exist
            # if exist, get the index
            idx = list(diff_Idx).index(min(diff_Idx[diff_Idx>0]))
            idxMinHalfL = idxMinHalf[idx]
            idxMaxHalfL = idxMinHalfL+1
        else:
            # if not, take the first index as index
            idxMinHalfL = 0
            idxMaxHalfL = idxMinHalfL+1

        # look for the point under 1/sqrt(2) what is nearest on the maximum, for the right edge!

        if (len(diff_Idx[diff_Idx<0]) >0): # check if a right edge exist
            # if, get the index
            idx = list(diff_Idx).index(max(diff_Idx[diff_Idx<0]))
            idxMinHalfR = idxMinHalf[idx]
            idxMaxHalfR = idxMinHalfR-1
        else:
            # if not, take the last index as index
            idxMinHalfR = len(relativTCExc[i])-1
            idxMaxHalfR = idxMinHalfR-1

        maxHalfL = (np.abs(idxMaxHalfL - idxMax)) * orienSteps # upper OBW for the left edge
        minHalfL = (np.abs(idxMinHalfL - idxMax)) * orienSteps # lower OBW for the left edge
        
        maxHalfR = (np.abs(idxMaxHalfR - idxMax)) * orienSteps # upper OBW for the right edge
        minHalfR = (np.abs(idxMinHalfR - idxMax)) * orienSteps # lower OBW for the right edge

        obwL = (maxHalfL+minHalfL)/2.
        obwR = (maxHalfR+minHalfR)/2.

        halfO[i] = (obwL+obwR)/2.
    return(halfO)
#--------------------------------------------------------------------------------
def plotOrientationBandwith(halfO,path,matter):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(halfO,'o')
    plt.xlabel('neuron index')
    plt.ylabel('orientation bandwith [degree]')
    ax.annotate('mean: %f'%np.mean(halfO),xy=(1, 2), xytext=(140, np.max(halfO)+5.0))
    plt.savefig('./Output/TC_'+path+'/BW_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)

    hist = np.histogram(halfO,9)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(halfO,color='lightgrey',linewidth=2.0,bins=np.arange(0.0,60.0,4))
    plt.axvline(x=np.mean(halfO), color = 'k',ls='dashed',linewidth=5.0)
    plt.xlabel('orientation bandwith ')
    plt.ylabel('number of neurons')
    plt.title('mean: %f'%np.mean(halfO))
    #labelsX = ['0$^\circ$','20$^\circ$','40$^\circ$','60$^\circ$']
    #plt.xticks(np.linspace(0,60,4),labelsX,fontsize = 20,fontweight = 'bold')#np.linspace(0,60,4))
    #plt.xlim(0.0,60.0)
    #plt.yticks(np.linspace(0,40,5),np.linspace(0,40,5),fontsize = 20,fontweight = 'bold')
    #plt.ylim(0,40)
    #ax.annotate('mean: %f'%np.mean(halfO),xy=(1, 1.0), xytext=(np.mean(halfO),np.mean(hist[0])))
    plt.savefig('./Output/TC_'+path+'/BW_hist_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotCircularVariance(circVar,path,matter):
    plt.figure()
    plt.hist(circVar,9)
    plt.xlabel('Circular Variance')
    #plt.xlim(0.0,1.0)
    plt.ylabel('# of Cells')
    plt.savefig('./Output/TC_'+path+'/CV_hist_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotOSI(circVar,path,matter):
    hist = np.histogram(circVar,9)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(circVar,9)
    plt.xlabel('OSI')
    plt.xlim(0.0,1.0)
    plt.ylabel('# of Cells')
    plt.axvline(x=np.mean(circVar), color = 'k',ls='dashed',linewidth=5.0)
    ax.annotate('mean: %f'%np.mean(circVar),xy=(1, 1.0), xytext=(np.mean(circVar),np.max(hist[0])))
    plt.savefig('./Output/TC_'+path+'/OSI_hist_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotOrientHist(params,path):
    orientations = params[:,1]
    stepSize = 22.5 #one bin = stepSize degress
    orientations = orientations/np.pi * 180.0
    plt.figure()
    plt.hist(orientations)#,bins=np.arange(0,180+stepSize,stepSize))
    plt.xlabel('Orientation [degrees]',fontsize=23,weight = 'bold')
    plt.ylabel('# of Neurons',fontsize=23,weight = 'bold')
    #plt.xlim(xmin=0,xmax=180)
    #plt.ylim(ymin=0,ymax=40)
    plt.savefig('./Output/TC_'+path+'/orient_Hist.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotMeanTC(tuningCurves,tc_gExc,tc_gInh,contLVL,shift,sadeOSI,bwCalcFull,matter,path):
    meanBW = np.mean(bwCalcFull,axis=0)
    meanOSI = np.mean(sadeOSI,axis=0)
    plt.figure(figsize=(4,5))    
    nbrCells,nbrAngl,nbrLVL = np.shape(tuningCurves)
    anglStep = 360.0/nbrAngl
    shiftStep = int(shift/anglStep)
    markers=['-.','--','-','-o',':'] #['o','^','+','<','>','X','H']
    maxInput = 100
    step = maxInput/nbrLVL
    total = step+(4*step)
    for i in range(1,nbrLVL-1):
        contLevel = (step + (i*step))/total *100
        meanTC = np.mean(tuningCurves[:,:,i],axis=0) 
        length = len(meanTC)
        meanTC = np.roll(meanTC,len(meanTC)-shiftStep)
        x = np.linspace(0,len(meanTC),len(meanTC))
        plt.plot(x,meanTC,markers[i-1],label=str(int(contLevel))+'%')#,marker=markers[i]) #color='seagreen',
    plt.legend()
    plt.ylabel('mean firing rate',fontsize=13)
    plt.xlabel('orientation [degrees]',fontsize=13)
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim(0,len(meanTC))
    plt.ylim(ymin=0.0)#,ymax=330)
    plt.savefig('./Output/TC_'+matter+'/'+path+'/TC_Mean_relative_Contrast.jpg',dpi=300,bbox_inches='tight')

    ymax = 150.0

    plt.figure(figsize=(10,8))
    std = np.std(tuningCurves[:,:,contLVL],axis=0,ddof=1)
    #std = np.concatenate((std[0:(len(std)/2)-1],std[(len(std)/2)+1:len(std)]))
    meanTC = np.mean(tuningCurves[:,:,contLVL],axis=0) 
    meanTC = np.roll(meanTC,len(meanTC)-shiftStep)
    length = len(meanTC)
    #meanTC = np.concatenate((meanTC[0:(length/2)-1],meanTC[(length/2)+1:length]))
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.plot(x,meanTC,'b-o',lw=5,ms=11)
    #plt.plot(x,meanTC+std/2,'r-')
    #plt.plot(x,meanTC-std/2,'r-')
    plt.fill_between(x,meanTC+std,meanTC-std,color='red',alpha='0.15')
    #plt.plot(nbrAngl/2 - (meanBW[contLVL]/anglStep),meanTC[((nbrAngl-1)/2)- (meanBW[contLVL]/anglStep)],'gx')
    #ymax = meanTC[nbrAngl/2- (meanBW[contLVL]/anglStep)] /np.max(meanTC+std/2) 
    #plt.axvline(x =(nbrAngl/2-(meanBW[contLVL]/anglStep)),ymin=0.0,ymax=1.0,color='g', linestyle='--',lw=4) #=meanTC[nbrAngl/2- (meanBW[contLVL]/anglStep)] 
    #plt.text(nbrAngl/2-(meanBW[contLVL]/anglStep +15),ymax-5,r"$\overline{OBW}= "+str(np.round(meanBW[contLVL],3))+" ^{\circ}$",fontsize=20)#np.max(meanTC+std)+(np.max(meanTC+std)/100*1)
    #plt.text(len(meanTC)-13,ymax-5,r"$\overline{OSI}= "+str(np.round(meanOSI[contLVL],3))+" $",fontsize=20)
    plt.ylabel('mean firing rate',fontsize=23,weight = 'bold')
    plt.xlabel('orientation [degrees]',fontsize=23,weight = 'bold')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=20)
    plt.yticks(np.linspace(0,ymax,3),fontsize=20)
    plt.xlim(0,len(meanTC))
    plt.ylim(ymin=0.0,ymax=ymax)
    plt.savefig('./Output/TC_'+matter+'/'+path+'/TC_Mean_relative.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)


    plt.figure(figsize=(10,8))
    std = np.std(tc_gExc[:,:,contLVL],axis=0,ddof=1)
    meanTC = np.mean(tc_gExc[:,:,contLVL],axis=0) 
    meanTC = np.roll(meanTC,len(meanTC)-shiftStep)
    length = len(meanTC)
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.plot(x,meanTC,'b-o',lw=5,ms=11)
    plt.fill_between(x,meanTC+std,meanTC-std,color='red',alpha='0.15')
    plt.ylabel('gExc',fontsize=23,weight = 'bold')
    plt.xlabel('orientation [degrees]',fontsize=23,weight = 'bold')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=20)
    #plt.yticks(np.linspace(0,ymax,3),fontsize=20)
    plt.xlim(0,len(meanTC))
    #plt.ylim(ymin=0.0,ymax=6.0)
    plt.savefig('./Output/TC_'+matter+'/'+path+'/TC_Mean_gExc.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)
    
    plt.figure(figsize=(10,8))
    std = np.std(tc_gInh[:,:,contLVL],axis=0,ddof=1)
    meanTC = np.mean(tc_gInh[:,:,contLVL],axis=0)
    meanTC = np.roll(meanTC,len(meanTC)-shiftStep) 
    length = len(meanTC)
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.plot(x,meanTC,'b-o',lw=5,ms=11)
    plt.fill_between(x,meanTC+std,meanTC-std,color='red',alpha='0.15')
    plt.ylabel('gInh',fontsize=23,weight = 'bold')
    plt.xlabel('orientation [degrees]',fontsize=23,weight = 'bold')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=20)
    #plt.yticks(np.linspace(0,ymax,3),fontsize=20)
    plt.xlim(0,len(meanTC))
    #plt.ylim(ymin=0.0,ymax=6.0)
    plt.savefig('./Output/TC_'+matter+'/'+path+'/TC_Mean_gInh.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    plt.close('all')

#------------------------------------------------------------------------------
def startAnalyseTuningCurves():
    plt.rc('xtick',labelsize = 18)
    plt.rc('ytick',labelsize = 18)

    #---- tuning curves over gabor input -----#
#    tcExc = np.load('./work/TuningCurves_Exc.npy')
    orient = np.load('./work/TuningCuver_orientation.npy')
    params = np.load('./work/TuningCurve_ParamsExc.npy')
    tcExcRelativ = np.load('./work/TuningCurvesRelativ_Exc.npy')
    tcExc_gExc = np.load('./work/TuningCurvesRelativ_gExc_Exc.npy')
    tcExc_gInh = np.load('./work/TuningCurvesRelativ_gInh_Exc.npy')
    tcInh = np.load('./work/TuningCurvesRelativ_Inh.npy')
    tcInh_gExc = np.load('./work/TuningCurvesRelativ_gExc_Inh.npy')
    tcInh_gInh = np.load('./work/TuningCurvesRelativ_gInh_Inh.npy')

    nbrCells,nbrAng,nbrContrast = np.shape(tcExcRelativ)
    halfA = nbrAng/2
    
    contLVL = 4#nbrContrast-3
    
    circVarExcFull = np.zeros((nbrCells,nbrContrast))
    circVarExcHalf = np.zeros((nbrCells,nbrContrast))
    sadeOSI = np.zeros((nbrCells,nbrContrast))
    bwEstFull =np.zeros((nbrCells,nbrContrast))
    bwEstHalf =np.zeros((nbrCells,nbrContrast))
    bwCalcFull =np.zeros((nbrCells,nbrContrast))
    bwCalcHalf = np.zeros((nbrCells,nbrContrast))

    for i in range(nbrContrast):
        minAngle,maxAngle = [-180.0,180.0]
        circVarExcFull[:,i] = calcCircularVarianceRingach(tcExcRelativ[:,:,i],minAngle,maxAngle)  


        tcExcRelat = tcExcRelativ[:,halfA/2:(nbrAng-halfA/2),i]
        minAngle,maxAngle = [-90.0,90.0]
        circVarExcHalf[:,i] = calcCircularVarianceRingach(tcExcRelat,minAngle,maxAngle)  

        minAngle,maxAngle = [-180.0,180.0]
        sadeOSI[:,i] = calcOSISadeh(tcExcRelativ[:,:,i],minAngle,maxAngle)
        

        tcExcRelat = tcExcRelativ[:,halfA/2:(nbrAng-halfA/2),i]
        minAngle,maxAngle = [-90.0,90.0]
        bwEstHalf[:,i] = estimateOrientationBandwidth(tcExcRelat,minAngle,maxAngle,'gabor')        

        minAngle,maxAngle = [-180.0,180.0]
        bwEstFull[:,i] = estimateOrientationBandwidth(tcExcRelativ[:,:,i],minAngle,maxAngle,'gabor')        

        tcExcRelat = tcExcRelativ[:,halfA/2:(nbrAng-halfA/2),i]
        minAngle,maxAngle = [-90.0,90.0]
        bwCalcHalf[:,i] = calculateOrientationBandwidth(tcExcRelat,minAngle,maxAngle)        

        minAngle,maxAngle = [-180.0,180.0]
        bwCalcFull[:,i] = calculateOrientationBandwidth(tcExcRelativ[:,:,i],minAngle,maxAngle)        

    np.save('./work/OrientBandwithEst_gabor',bwEstFull)
    np.save('./work/OSI_Sade',sadeOSI)

    sortTC_gExc = np.zeros((nbrCells,nbrAng)) 
    sortTC_gInh = np.zeros((nbrCells,nbrAng))
    for i in range(nbrCells):  
        idx = np.argsort(tcExc_gExc[i,:,contLVL])
        sortTC_gExc[i] = tcExc_gExc[i,idx,contLVL]
        sortTC_gInh[i] = tcExc_gInh[i,idx,contLVL]
        


    #---plot data---#

    plt.figure()
    plt.plot(np.mean(sortTC_gExc,axis=0),'o',color='seagreen',label='gExc')
    plt.plot(np.mean(sortTC_gInh,axis=0),'^',color='orange',label='gInh')
    plt.savefig('./Output/TC_gabor/Currents.jpg',dip=300,bbox_inches='tight', pad_inches = 0.1)

    shift = 0.0
    plotOrientHist(params,'gabor')
    plotMeanTC(tcExcRelativ,tcExc_gExc,tcExc_gInh,contLVL,shift,sadeOSI,bwEstFull,'gabor','excitatory')
    plotMeanTC(tcInh,tcInh_gExc,tcInh_gInh,contLVL,shift,sadeOSI,bwEstFull,'gabor','inhibitory')

    plotCircularVariance(circVarExcFull[:,contLVL],'gabor','CalcRingach')
    plotOSI(1-circVarExcFull[:,contLVL],'gabor','CalcRingach')

    plotCircularVariance(circVarExcHalf[:,contLVL],'gabor','CalcRingach_pi')
    plotOSI(1-circVarExcHalf[:,contLVL],'gabor','CalcRingach_pi')

    plotOSI(sadeOSI[:,contLVL],'gabor','CalcSadeh')
    plotOrientationBandwith(bwEstHalf[:,contLVL],'gabor','est_Pi')
    plotOrientationBandwith(bwEstFull[:,contLVL],'gabor','est')
    plotOrientationBandwith(bwCalcHalf[:,contLVL],'gabor','calc_Pi')
    plotOrientationBandwith(bwCalcFull[:,contLVL],'gabor','calc')

    plt.figure()
    plt.plot(np.mean(bwEstFull,axis=0),'-o')
    plt.savefig('./Output/TC_gabor/MeanbandWith')
    
    plt.figure()
    plt.plot(np.mean(sadeOSI,axis=0),'-o')
    plt.savefig('./Output/TC_gabor/MeanOSI')
#------------------------------------------------------------------------------
def startAnalyse_Sinus():
    print('Start analysis sinus gratings')
    #---- tuning curves over sinus gratings ---#
    tcExc = np.load('./work/TuningCurves_sinus_Exc.npy')
    tcInh = np.load('./work/TuningCurves_sinus_Inh.npy')
    tCExcSinRelativ = np.load('./work/TuningCurvesRelativ_sinus_Exc.npy')
    tCInhSinRelativ = np.load('./work/TuningCurvesRelativ_sinus_Inh.npy')
    relaticTC_exc_gE = np.load('./work/TuningCurvesRelativ_sinus_Exc_gE.npy')
    relaticTC_exc_gI = np.load('./work/TuningCurvesRelativ_sinus_Exc_gI.npy')
    nbrCells,nbrPatches,nbrContrast = np.shape(tCExcSinRelativ)

    halfA = int(nbrPatches/2)

    circVarExcFull = np.zeros((nbrCells,nbrContrast))
    circVarExcHalf = np.zeros((nbrCells,nbrContrast))
    sadeOSI = np.zeros((nbrCells,nbrContrast))
    bwEstFull =np.zeros((nbrCells,nbrContrast))
    bwEstHalf =np.zeros((nbrCells,nbrContrast))
    bwCalcFull =np.zeros((nbrCells,nbrContrast))
    bwCalcHalf = np.zeros((nbrCells,nbrContrast))
    cv = np.zeros((nbrCells,nbrContrast))

    cvI= np.zeros((int(nbrCells/4),nbrContrast))
    bwEstHalf_I =np.zeros((int(nbrCells/4),nbrContrast))

    for i in range(nbrContrast):

        minAngle,maxAngle = [0.0,360.0]
        cv[:,i] = calcCVSinus(tcExc[:,:,i,:],minAngle,maxAngle)
        cvI[:,i] = calcCVSinus(tcInh[:,:,i,:],minAngle,maxAngle)

        minAngle,maxAngle = [-90.0,90.0]
        circVarExcFull[:,i] = calcCircularVarianceRingach(tCExcSinRelativ[:,:,i],minAngle,maxAngle)  

        tcExcRelat = tCExcSinRelativ[:,int(halfA/2):(nbrPatches-int(halfA/2)),i]
        minAngle,maxAngle = [-90.0,90.0]
        circVarExcHalf[:,i] = calcCircularVarianceRingach(tcExcRelat,minAngle,maxAngle)  

        minAngle,maxAngle = [-90.0,90.0]
        sadeOSI[:,i] = calcOSISadeh(tCExcSinRelativ[:,:,i],minAngle,maxAngle)
        

        tcExcRelat = tCExcSinRelativ[:,int(halfA/2):(nbrPatches-int(halfA/2)),i] # calculate the OBW on the relative TC with -90 + 90 degree  
        minAngle,maxAngle = [-90.0,90.0]
        bwEstHalf[:,i] = estimateOrientationBandwidth(tcExcRelat,minAngle,maxAngle,'sinus')        

        #tcExcRelat = tCInhSinRelativ[:,int(halfA/2):(nbrPatches-int(halfA/2)),i] #TODO: Problem with inhibitory data ?!
        #minAngle,maxAngle = [-90.0,90.0]
        #bwEstHalf_I[:,i] = estimateOrientationBandwidth(tcExcRelat,minAngle,maxAngle,'sinus')        

        minAngle,maxAngle = [-180.0,180.0]
        bwEstFull[:,i] = estimateOrientationBandwidth(tCExcSinRelativ[:,:,i],minAngle,maxAngle,'sinus')        

        tcExcRelat = tCExcSinRelativ[:,int(halfA/2):(nbrPatches-int(halfA/2)),i]
        minAngle,maxAngle = [-90.0,90.0]
        bwCalcHalf[:,i] = calculateOrientationBandwidth(tcExcRelat,minAngle,maxAngle)        

        minAngle,maxAngle = [-180.0,180.0]
        bwCalcFull[:,i] = calculateOrientationBandwidth(tCExcSinRelativ[:,:,i],minAngle,maxAngle)        

    np.save('./work/OrientBandwithEst_Sinus',bwEstFull)
    np.save('./work/OrientBandwithEst_Half_Sinus',bwEstHalf)
    np.save('./work/OSI_Sade_Sinus',sadeOSI)

    contLVL = 5

    sortTC_gExc = np.zeros((nbrCells,nbrPatches)) 
    sortTC_gInh = np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):  
        idx = np.argsort(relaticTC_exc_gE[i,:,contLVL])
        sortTC_gExc[i] = relaticTC_exc_gE[i,idx,contLVL]
        sortTC_gInh[i] = relaticTC_exc_gI[i,idx,contLVL]

    #plotOrientHist(params,'sinus')

    plt.figure()
    plt.plot(np.mean(sortTC_gExc,axis=0),'o',color='seagreen',label='gExc')
    plt.plot(np.mean(sortTC_gInh,axis=0),'^',color='orange',label='gInh')
    plt.savefig('./Output/TC_sinus/Currents.jpg',dip=300,bbox_inches='tight', pad_inches = 0.1)

    shift = 90.0
    plotMeanTC(tCExcSinRelativ,relaticTC_exc_gE,relaticTC_exc_gI,contLVL,shift,sadeOSI,bwEstHalf,'sinus','excitatory')

    plotCircularVariance(circVarExcFull[:,contLVL],'sinus','CalcRingach')
    plotOSI(1-circVarExcFull[:,contLVL],'sinus','CalcRingach')

    plotCircularVariance(circVarExcHalf[:,contLVL],'sinus','CalcRingach_pi')
    plotOSI(1-circVarExcHalf[:,contLVL],'sinus','CalcRingach_pi')


    plotCircularVariance(cv[:,contLVL],'sinus','CalcRingachCOMPLETE')
    plotOSI(1-cv[:,contLVL],'sinus','CalcRingachCOMPLETE')

    plotCircularVariance(cvI[:,contLVL],'sinus','CV_Inhib')
    plotOSI(1-cvI[:,contLVL],'sinus','OSI_Inhib')

    plotOSI(sadeOSI[:,contLVL],'sinus','CalcSadeh')
    plotOrientationBandwith(bwEstHalf[:,contLVL],'sinus','est_Pi')
    plotOrientationBandwith(bwEstFull[:,contLVL],'sinus','est')
    plotOrientationBandwith(bwCalcHalf[:,contLVL],'sinus','calc_Pi')
    plotOrientationBandwith(bwCalcFull[:,contLVL],'sinus','calc')
    plotOrientationBandwith(bwEstHalf_I[:,contLVL],'sinus','est_Pi_Inhib')

    plt.figure()
    plt.plot(np.mean(bwEstHalf,axis=0),'-o')
    plt.savefig('./Output/TC_sinus/MeanbandWith')
    
    plt.figure()
    plt.plot(np.mean(sadeOSI,axis=0),'-o')
    plt.savefig('./Output/TC_sinus/MeanOSI_sade')

    plt.figure()
    plt.plot(np.mean(cv,axis=0),'o-')
    plt.savefig('./Output/TC_sinus/MeanCV')

    plt.figure()
    plt.plot(np.mean(1-cv,axis=0),'-o')
    plt.savefig('./Output/TC_sinus/MeanOSI')
#------------------------------------------------------------------------------
def startAnalyse_Sinus_pos():
    print('Start analysis sinus gratings')
    #---- tuning curves over sinus gratings ---#
    tcExc = np.load('./work/TuningCurves_sinus_Exc_pos.npy')
    #tcInh = np.load('./work/TuningCurves_sinus_Inh_pos.npy')
    tCExcSinRelativ = np.load('./work/TuningCurvesRelativ_sinus_Exc_pos.npy')
    #tCInhSinRelativ = np.load('./work/TuningCurvesRelativ_sinus_Inh_pos.npy')
    relaticTC_exc_gE = np.load('./work/TuningCurvesRelativ_sinus_Exc_gE_pos.npy')
    relaticTC_exc_gI = np.load('./work/TuningCurvesRelativ_sinus_Exc_gI_pos.npy')
    nbrCells,nbrPatches = np.shape(tCExcSinRelativ)

    halfA = nbrPatches/2

    circVarExcFull = np.zeros((nbrCells))
    circVarExcHalf = np.zeros((nbrCells))
    sadeOSI = np.zeros((nbrCells))
    bwEstFull =np.zeros((nbrCells))
    bwEstHalf =np.zeros((nbrCells))
    bwCalcFull =np.zeros((nbrCells))
    bwCalcHalf = np.zeros((nbrCells))
    cv = np.zeros((nbrCells))

    sp = int(halfA/2)

    #cvI= np.zeros((nbrCells/4,nbrContrast))
    #bwEstHalf_I =np.zeros((nbrCells/4,nbrContrast))

    minAngle,maxAngle = [0.0,360.0]
    cv = calcCVSinus(tcExc,minAngle,maxAngle)
    #cvI = calcCVSinus(tcInh,minAngle,maxAngle)

    minAngle,maxAngle = [-180.0,180.0]
    circVarExcFull = calcCircularVarianceRingach(tCExcSinRelativ,minAngle,maxAngle)  

    tcExcRelat = tCExcSinRelativ[:,sp:(nbrPatches-sp)]
    minAngle,maxAngle = [-90.0,90.0]
    circVarExcHalf = calcCircularVarianceRingach(tcExcRelat,minAngle,maxAngle)  

    minAngle,maxAngle = [-180.0,180.0]
    sadeOSI = calcOSISadeh(tCExcSinRelativ,minAngle,maxAngle)
        

    tcExcRelat = tCExcSinRelativ[:,sp:(nbrPatches-sp)]
    minAngle,maxAngle = [-90.0,90.0]
    bwEstHalf = estimateOrientationBandwidth(tcExcRelat,minAngle,maxAngle,'sinus')        

    #tcExcRelat = tCInhSinRelativ[:,sp:(nbrPatches-sp)]
    #minAngle,maxAngle = [-90.0,90.0]
    #bwEstHalf_I = estimateOrientationBandwidth(tcExcRelat,minAngle,maxAngle,'sinus')        

    minAngle,maxAngle = [-180.0,180.0]
    bwEstFull = estimateOrientationBandwidth(tCExcSinRelativ,minAngle,maxAngle,'sinus')        

    tcExcRelat = tCExcSinRelativ[:,sp:(nbrPatches-sp)]
    minAngle,maxAngle = [-90.0,90.0]
    bwCalcHalf = calculateOrientationBandwidth(tcExcRelat,minAngle,maxAngle)        

    minAngle,maxAngle = [-180.0,180.0]
    bwCalcFull = calculateOrientationBandwidth(tCExcSinRelativ,minAngle,maxAngle)        

    np.save('./work/OrientBandwithEst_Sinus',bwEstFull)
    np.save('./work/OSI_Sade_Sinus',sadeOSI)


    sortTC_gExc = np.zeros((nbrCells,nbrPatches)) 
    sortTC_gInh = np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):  
        idx = np.argsort(relaticTC_exc_gE[i,:])
        sortTC_gExc[i] = relaticTC_exc_gE[i,idx]
        sortTC_gInh[i] = relaticTC_exc_gI[i,idx]

    #plotOrientHist(params,'sinus')

    plt.figure()
    plt.plot(np.mean(sortTC_gExc,axis=0),'o',color='seagreen',label='gExc')
    plt.plot(np.mean(sortTC_gInh,axis=0),'^',color='orange',label='gInh')
    plt.savefig('./Output/TC_sinus_pos/Currents.jpg',dip=300,bbox_inches='tight', pad_inches = 0.1)

    shift = 90.0
    #plotMeanTC(tCExcSinRelativ,relaticTC_exc_gE,relaticTC_exc_gI,contLVL,shift,sadeOSI,bwEstHalf,'sinus','excitatory')

    #----------------------------------------------
    meanBW = np.mean(bwEstHalf,axis=0)
    meanOSI = np.mean(sadeOSI,axis=0)
    plt.figure()    
    nbrCells,nbrAngl = np.shape(tCExcSinRelativ)
    anglStep = 360.0/nbrAngl
    shiftStep = int(shift/anglStep)
    ymax = 150.0

    plt.figure(figsize=(10,8))
    std = np.std(tCExcSinRelativ,axis=0,ddof=1)
    meanTC = np.mean(tCExcSinRelativ,axis=0) 
    meanTC = np.roll(meanTC,len(meanTC)-shiftStep)
    length = len(meanTC)
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.plot(x,meanTC,'b-o',lw=5,ms=11)
    plt.fill_between(x,meanTC+std,meanTC-std,color='red',alpha='0.15')
    plt.ylabel('mean firing rate',fontsize=23,weight = 'bold')
    plt.xlabel('orientation [degrees]',fontsize=23,weight = 'bold')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=20)
    plt.yticks(np.linspace(0,ymax,3),fontsize=20)
    plt.xlim(0,len(meanTC))
    plt.ylim(ymin=0.0,ymax=ymax)
    plt.savefig('./Output/TC_sinus_pos/excitatory/TC_Mean_relative.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)


    plt.figure(figsize=(10,8))
    std = np.std(relaticTC_exc_gE,axis=0,ddof=1)
    meanTC = np.mean(relaticTC_exc_gE,axis=0) 
    meanTC = np.roll(meanTC,len(meanTC)-shiftStep)
    length = len(meanTC)
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.plot(x,meanTC,'b-o',lw=5,ms=11)
    plt.fill_between(x,meanTC+std,meanTC-std,color='red',alpha='0.15')
    plt.ylabel('gExc',fontsize=23,weight = 'bold')
    plt.xlabel('orientation [degrees]',fontsize=23,weight = 'bold')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=20)
    #plt.yticks(np.linspace(0,ymax,3),fontsize=20)
    plt.xlim(0,len(meanTC))
    #plt.ylim(ymin=0.0,ymax=6.0)
    plt.savefig('./Output/TC_sinus_pos/excitatory/TC_Mean_gExc.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)
    
    plt.figure(figsize=(10,8))
    std = np.std(relaticTC_exc_gI,axis=0,ddof=1)
    meanTC = np.mean(relaticTC_exc_gI,axis=0)
    meanTC = np.roll(meanTC,len(meanTC)-shiftStep) 
    length = len(meanTC)
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.plot(x,meanTC,'b-o',lw=5,ms=11)
    plt.fill_between(x,meanTC+std,meanTC-std,color='red',alpha='0.15')
    plt.ylabel('gInh',fontsize=23,weight = 'bold')
    plt.xlabel('orientation [degrees]',fontsize=23,weight = 'bold')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180+shift,180+shift,5),fontsize=20)
    #plt.yticks(np.linspace(0,ymax,3),fontsize=20)
    plt.xlim(0,len(meanTC))
    #plt.ylim(ymin=0.0,ymax=6.0)
    plt.savefig('./Output/TC_sinus_pos/excitatory/TC_Mean_gInh.jpg',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    plt.close('all')
    #----------------------------------------------

    plotCircularVariance(circVarExcFull,'sinus_pos','CalcRingach')
    plotOSI(1-circVarExcFull,'sinus_pos','CalcRingach')

    plotCircularVariance(circVarExcHalf,'sinus_pos','CalcRingach_pi')
    plotOSI(1-circVarExcHalf,'sinus_pos','CalcRingach_pi')


    plotCircularVariance(cv,'sinus_pos','CalcRingachCOMPLETE')
    plotOSI(1-cv,'sinus_pos','CalcRingachCOMPLETE')

    #plotCircularVariance(cvI,'sinus_pos','CV_Inhib')
    #plotOSI(1-cvI,'sinus_pos','OSI_Inhib')

    plotOSI(sadeOSI,'sinus_pos','CalcSadeh')
    plotOrientationBandwith(bwEstHalf,'sinus_pos','est_Pi')
    plotOrientationBandwith(bwEstFull,'sinus_pos','est')
    plotOrientationBandwith(bwCalcHalf,'sinus_pos','calc_Pi')
    plotOrientationBandwith(bwCalcFull,'sinus_pos','calc')
    #plotOrientationBandwith(bwEstHalf_I,'sinus_pos','est_Pi_Inhib')

    plt.figure()
    plt.plot(np.mean(bwEstHalf,axis=0),'-o')
    plt.savefig('./Output/TC_sinus_pos/MeanbandWith')
    
    plt.figure()
    plt.plot(np.mean(sadeOSI,axis=0),'-o')
    plt.savefig('./Output/TC_sinus_pos/MeanOSI_sade')

    plt.figure()
    plt.plot(np.mean(cv,axis=0),'o-')
    plt.savefig('./Output/TC_sinus_pos/MeanCV')

    plt.figure()
    plt.plot(np.mean(1-cv,axis=0),'-o')
    plt.savefig('./Output/TC_sinus_pos/MeanOSI')
#------------------------------------------------------------------------------
if __name__ == "__main__":
    #startAnalyseTuningCurves()
    startAnalyse_Sinus()
    #startAnalyse_Sinus_pos()

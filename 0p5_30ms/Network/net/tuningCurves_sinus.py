from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import Gabor_sinus as gabor
import os.path
from net_EB_LGN_DoG import *
from tqdm import tqdm

#-------------------------------------------------------------------------------
def createInput(parameters,maxInput):
    inputIMG = gabor.createGaborMatrix(parameters,s_input,s_input)
    maxVal = np.max(np.abs(inputIMG))
    #print(np.shape(inputIMG))
    sizeX,sizeY=np.shape(inputIMG)
    #print(inputPatch[:,:,1])
    #plt.figure()
    #plt.imshow(inputIMG,cmap=plt.get_cmap('gray'))
    #plt.savefig('Output/tmp/img_'+str(parameters[1])+'_'+str(parameters[5])+'_.png')

    popIMG.r = (inputIMG/maxVal)*maxInput

#--------------------------------------------------------------------------------
def runNet(maxDegree,stepDegree,stepPhase,stepFrequency,duration,maxInput):
    compile()
    loadWeights()
    phaseShifts = np.pi / float(stepPhase)
    nbrOfInputs = int(maxDegree/stepDegree)
    frequShift = (0.15 - 0.05)/float(stepFrequency)  #from 0.5 to 1.5 * s_input#
    repeats = 50
    contrastLVLs = 7

    V1Mon=Monitor(popE1,['spike','g_Exc','g_Inh'])
    rec_frEx= np.zeros((n_E1,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_gExc= np.zeros((n_E1,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_gInh= np.zeros((n_E1,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))

    InhibMon=Monitor(popIL1,['spike'])
    rec_frInh= np.zeros((n_I1,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    orientation = np.zeros((nbrOfInputs))
    maxFrExc = np.zeros(n_E1)
    maxFrInh = np.zeros(n_I1)
    
    ### lagged LGN cells over syn-Delays
    # add some extra delays to implement "lagged" LGN-Cells -> additional delay depends on t_delay !
    projInput_LGN_ON.delay = np.load('./work/LGN_ON_delay.npy')
    projInput_LGN_OFF.delay = np.load('./work/LGN_OFF_delay.npy')
    
    for i in range(nbrOfInputs):
        orientation[i] = stepDegree*np.pi/180.0*i
        print("Orientation: "+str(orientation[i]))
        for k in range(stepPhase):
            for f in range(stepFrequency):
                for c in range(contrastLVLs):
                    freq = (0.05+ (0.05*f))*s_input 
                    phase = phaseShifts*k
                    #parameters = np.array((1.,0.0,0.13,0.2,np.pi/2.0,np.pi/2.0,s_input,s_input,0.0))
                    parameters = np.array((1.,0.0,0.13,0.2,0.1*s_input,np.pi/2.0,s_input,s_input,0.0))
                    parameters[1] = orientation[i]
                    parameters[4] = freq
                    parameters[5] = phase
                    parameters[6] = s_input/2
                    parameters[7] = s_input/2
                    inputFR = maxInput/contrastLVLs + (maxInput/contrastLVLs)*c
                    #print('InputFR = ', inputFR)
                    for r in range(repeats):
                        createInput(parameters,inputFR)
                        simulate(duration)
                        spikesEx = V1Mon.get('spike')
                        gExcEx = V1Mon.get('g_Exc')
                        gInhEx = V1Mon.get('g_Inh')
                        spikesInh = InhibMon.get('spike')
                        for j in range(n_E1):
                            rateEx = len(spikesEx[j])*1000/duration
                            rec_frEx[j,i,k,f,c,r] = rateEx
                            rec_Exc_gExc[j,i,k,f,c,r] = np.mean(gExcEx[:,j])
                            rec_Exc_gInh[j,i,k,f,c,r] = np.mean(gInhEx[:,j])

                            if (j < (n_I1)):
                                rateInh = len(spikesInh[j])*1000/duration
                                rec_frInh[j,i,k,f,c,r] = rateInh
                        #print(np.mean(rec_frEx[:,i,k,f,c,r]))

        #-save Data for later statistical analysis-#
    np.save('./work/TuningCurve_sinus_frEx',rec_frEx)
    np.save('./work/TuningCurve_sinus_frInhib',rec_frInh)
    np.save('./work/TuningCurve_sinus_gExc',rec_Exc_gExc)
    np.save('./work/TuningCurve_sinus_gInh',rec_Exc_gInh)
    np.save('./work/TuningCurve_sinus_orientation',orientation)
    return(rec_frEx,rec_frInh,orientation)
#--------------------------------------------------------------------------------
def calcTuningC():
    frExc = np.load('./work/TuningCurve_sinus_frEx.npy')
    nbrOfNeurons,nbrOfPatches = np.shape(frExc)
    frInhib= np.load('./work/TuningCurve_sinus_frInhib.npy')
    orientations = np.load('./work/TuningCuver_sinus_orientation.npy')
    tuningExc = np.zeros((nbrOfNeurons,nbrOfPatches))
    for i in range(nbrOfNeurons):
        plt.figure()
        plt.plot(frExc[i])
        plt.savefig('./Output/TC_sinus/TC_'+str(i)+'.png',bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
#--------------------------------------------------------------------------------
def calcTuningCC(fr):
    #estimate the best fit for tuning curve over all parameters -> every cell can have different parameter
    nb_cells,nb_degree,nb_phases,nb_freq,trials = np.shape(fr)
    tuningC = np.zeros((nb_cells,nb_degree,trials))
    meanFR = np.mean(fr,axis=4)
    parameters = np.zeros((nb_cells,3)) # orientation, phase, frequency

    for i in range(nb_cells):
        goodIndx = (np.where(meanFR[i] ==np.max(meanFR[i])))
        sumFR = np.sum(meanFR[i,goodIndx[1],goodIndx[2]],axis=1)
        probBestIdx = np.where(sumFR ==np.max(sumFR)) # Index with probaly the best tuning Curve
        orientID = goodIndx[0][probBestIdx[0][0]]
        phaseID = goodIndx[1][probBestIdx[0][0]]
        freqID = goodIndx[2][probBestIdx[0][0]]
        tuningC[i] = fr[i,:,phaseID,freqID,:]
        parameters[i] =[orientID,phaseID,freqID]

    return(tuningC,parameters)
#--------------------------------------------------------------------------------
def calcTuningC_variable(data,params):
    #get the curve of the neuron variable, depending on the fit of the tuning curve
    nb_cells,nb_degree,nb_phases,nb_freq,trials = np.shape(data)
    tuningC = np.zeros((nb_cells,nb_degree,trials))

    for i in range(nb_cells):
        _, phaseID,freqID =params[i,:]
        tuningC[i] = data[i,:,int(phaseID),int(freqID),:]
    return(tuningC)
#--------------------------------------------------------------------------------
def calcRelativTuningCurve(tuningCurve):
    # shift the values of the neuron fire rates, that they are sorted relative to the optimal orientation in the middle
    nbrOfNeurons,nbrOfPatches = np.shape(tuningCurve)
    relativTC = np.zeros((nbrOfNeurons,nbrOfPatches))
    for i in range(nbrOfNeurons):
        orientIdx = np.where(tuningCurve[i] == np.max(tuningCurve[i]))#Index for orientation, wehre the fire rate is maximum
        relativTC[i] = np.roll(tuningCurve[i,:], int(nbrOfPatches/2) - orientIdx[0][0])
    return(relativTC)
#--------------------------------------------------------------------------------
def plotSingleTuningCurves(tuningCurves,path):
    nbrOfNeurons,nbrOfPatches = np.shape(tuningCurves)
    for i in range(nbrOfNeurons):
        plt.figure()
        plt.plot(tuningCurves[i]/np.max(tuningCurves[i]))
        plt.ylabel('nomralized fire rate')
        plt.savefig('./Output/TC_sinus/'+path+'/TC_'+str(i)+'.png',bbox_inches='tight', pad_inches = 0.1)
#--------------------------------------------------------------------------------
def plotAllTuningCurves(tuningCurves,path,maxDegree):
    nbrOfNeurons,nbrOfPatches = np.shape(tuningCurves)
    normedTCLoc = np.zeros((nbrOfNeurons,nbrOfPatches))
    normedTCGlo = np.zeros((nbrOfNeurons,nbrOfPatches))

    upBound = maxDegree/2
    lowBound = maxDegree/2 *-1
    
    plt.figure()
    for i in range(nbrOfNeurons):
        plt.plot(tuningCurves[i])
    plt.ylabel('nomralized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,nbrOfPatches,5),np.linspace(lowBound,upBound,5))
    plt.savefig('./Output/TC_sinus/'+path+'/TC_ALL_relative.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    for i in range(nbrOfNeurons):
        #std = np.std(tuningCurves[i],ddof=1)
        #mean = np.mean(tuningCurves[i])
        normedTCLoc[i,:] =tuningCurves[i]/np.max(tuningCurves[i])#(tuningCurves[i]-mean)/(std**2) 
        plt.plot(normedTCLoc[i,:])
    plt.ylabel('nomralized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,nbrOfPatches,5),np.linspace(lowBound,upBound,5))
    plt.ylim(0,1.3)
    plt.savefig('./Output/TC_sinus/'+path+'/TC_ALL_relativeNormedLokal.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    for i in range(nbrOfNeurons):
        #std = np.std(tuningCurves[i],ddof=1)
        #mean = np.mean(tuningCurves[i])
        normedTCGlo[i,:] =tuningCurves[i]/np.max(tuningCurves)#(tuningCurves[i]-mean)/(std**2) 
        plt.plot(normedTCGlo[i,:])
    plt.ylabel('nomralized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,nbrOfPatches,5),np.linspace(lowBound,upBound,5))
    plt.ylim(0,1.0)
    plt.savefig('./Output/TC_sinus/'+path+'/TC_ALL_relativeNormedGlobal.png',bbox_inches='tight', pad_inches = 0.1)

    std = np.std(tuningCurves,axis=0,ddof=1)
    #std = np.concatenate((std[0:(len(std)/2)-1],std[(len(std)/2)+1:len(std)]))
    meanTC = np.mean(tuningCurves,axis=0) 
    length = len(meanTC)
    #meanTC = np.concatenate((meanTC[0:(length/2)-1],meanTC[(length/2)+1:length]))
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.figure()
    plt.plot(x,meanTC,'-o')
    #plt.plot(x,meanTC+std/2,'r-')
    #plt.plot(x,meanTC-std/2,'r-')
    plt.fill_between(x,meanTC+std/2,meanTC-std/2,color='red',alpha=0.15)
    plt.ylabel('normalized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(lowBound,upBound,5))
    plt.xlim(0,len(meanTC))
    plt.ylim(ymin=0.0)
    plt.savefig('./Output/TC_sinus/'+path+'/TC_Mean_relative.png',bbox_inches='tight', pad_inches = 0.1)

    std = np.std(normedTCLoc,axis=0,ddof=1)
    #std = np.concatenate((std[0:(len(std)/2)-1],std[(len(std)/2)+1:len(std)]))
    meanTC = np.mean(normedTCLoc,axis=0) 
    length = len(meanTC)
    #meanTC = np.concatenate((meanTC[0:(length/2)-1],meanTC[(length/2)+1:length]))
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.figure()
    plt.plot(x,meanTC,'-o')
    #plt.plot(x,meanTC+std/2,'r-')
    #plt.plot(x,meanTC-std/2,'r-')
    plt.fill_between(x,meanTC+std/2,meanTC-std/2,color='red',alpha=0.15)
    plt.ylabel('nomralized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(lowBound,upBound,5))
    plt.xlim(0,len(meanTC))    
    plt.ylim(0,1.0)
    plt.savefig('./Output/TC_sinus/'+path+'/TC_Mean_relativeNormedLokal.png',bbox_inches='tight', pad_inches = 0.1)

    std = np.std(normedTCGlo,axis=0,ddof=1)
    #std = np.concatenate((std[0:(len(std)/2)-1],std[(len(std)/2)+1:len(std)]))
    meanTC = np.mean(normedTCGlo,axis=0) 
    length = len(meanTC)
    #meanTC = np.concatenate((meanTC[0:(length/2)-1],meanTC[(length/2)+1:length]))
    x = np.linspace(0,len(meanTC),len(meanTC))
    plt.figure()
    plt.plot(x,meanTC,'-o')
    #plt.plot(x,meanTC+std/2,'r-')
    #plt.plot(x,meanTC-std/2,'r-')
    plt.fill_between(x,meanTC+std/2,meanTC-std/2,color='red',alpha=0.15)
    plt.ylabel('nomralized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(lowBound,upBound,5))
    plt.xlim(0,len(meanTC))
    plt.ylim(0,1.0)
    plt.savefig('./Output/TC_sinus/'+path+'/TC_Mean_relativeNormedGlobal.png',bbox_inches='tight', pad_inches = 0.1)
#--------------------------------------------------------------------------------
def plotGabor(parameters):
    #parameters =np.array((1.,np.pi/2.0,0.2,0.2,np.pi/2.0,np.pi/2.0,6.0,6.0,0.0))
    inputIMG = gabor.createGaborMatrix(parameters,s_input,s_input)
    plt.figure()
    plt.imshow(inputIMG,cmap=plt.get_cmap('gray'),interpolation='none')
    plt.savefig('./Output/TC_sinus/TC_Gabor.png',bbox_inches='tight', pad_inches = 0.1)
#--------------------------------------------------------------------------------
def plotOrientationBandwith(halfO):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(halfO,'o')
    plt.xlabel('neuron index')
    plt.ylabel('orientation bandwith [degree]')
    ax.annotate('mean: %f'%np.mean(halfO),xy=(1, 2), xytext=(140, np.max(halfO)+1.0))
    plt.savefig('./Output/TC_sinus/BW.png',bbox_inches='tight', pad_inches = 0.1)

    hist = np.histogram(halfO,9)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(halfO,9,color='lightgrey',linewidth=2.0)
    plt.axvline(x=np.mean(halfO), color = 'k',ls='dashed',linewidth=4.0)
    #plt.xlabel('orientation bandwith [degree]')
    #plt.ylabel('number of neurons')
    labelsX = ['0$^\circ$','20$^\circ$','40$^\circ$','60$^\circ$']
    plt.xticks(np.linspace(0,60,4),labelsX,fontsize = 20,fontweight = 'bold')#np.linspace(0,60,4))
    plt.xlim(0.0,60.0)
    plt.yticks(np.linspace(0,40,5),np.linspace(0,40,5),fontsize = 20,fontweight = 'bold')
    plt.ylim(0,40)
    #ax.annotate('mean: %f'%np.mean(halfO),xy=(1, 2), xytext=(np.mean(halfO)+1.0,np.max(hist[0])+2.0))
    plt.savefig('./Output/TC_sinus/BW_hist.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def plotMeanTC(tuningCurves,path):
    plt.figure()
    nbrLVL = np.shape(tuningCurves)[2]
    for i in range(nbrLVL):
        meanTC = np.mean(tuningCurves[:,:,i],axis=0) 
        length = len(meanTC)
        meanTC = np.concatenate((meanTC[0:(int(length/2))-1],meanTC[(int(length/2))+1:length]))
        x = np.linspace(0,len(meanTC),len(meanTC))
        plt.plot(x,meanTC,'-o')
    plt.ylabel('normalized fire rate')
    plt.xlabel('Stim. Orien. relative [deg]')
    plt.xticks(np.linspace(0,len(meanTC),5),np.linspace(-180,180,5))
    plt.xlim(0,len(meanTC))
    plt.ylim(ymin=0.0)
    plt.savefig('./Output/TC_sinus/'+path+'/TC_Mean_relative_Contrast.png',bbox_inches='tight', pad_inches = 0.1)
#--------------------------------------------------------------------------------
def startTuningCurves(duration,maxInput):
    print('start to determine the tuning Curves over sinus gratings')
    print('Simulation time : '+str(duration)+' ms')
    print('maximum Input: '+str(maxInput))
    print('------------------------------------------')
    if not os.path.exists('./Output/TC_sinus/'):
        os.mkdir('./Output/TC_sinus/')
    if not os.path.exists('./Output/TC_sinus/excitatory/'):
        os.mkdir('./Output/TC_sinus/excitatory/')
    if not os.path.exists('./Output/TC_sinus/inhibitory/'):
        os.mkdir('./Output/TC_sinus/inhibitory/')

    maxDegree = 360
    orienSteps = 8#2#5
    phaseSteps = 8#8
    frequSteps = 4#4
    frExc,frInh,orient = runNet(maxDegree,orienSteps,phaseSteps,frequSteps,duration,maxInput)    

    frExc = np.load('./work/TuningCurve_sinus_frEx.npy')
    frInh = np.load('./work/TuningCurve_sinus_frInhib.npy')
    orient =np.load('./work/TuningCurve_sinus_orientation.npy')
    gExc_Exc = np.load('./work/TuningCurve_sinus_gExc.npy')
    gInh_Exc = np.load('./work/TuningCurve_sinus_gInh.npy')


    nb_cells,nb_degree,nb_phases,nb_freq,nb_contrast,trials = np.shape(frExc)

    tuningCExc = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    tuningCgExcE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    tuningCgInhE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))

    relativTCExc = np.zeros((nb_cells,nb_degree,nb_contrast))
    relativTCgExcE = np.zeros((nb_cells,nb_degree,nb_contrast))
    realtivTCgInhE = np.zeros((nb_cells,nb_degree,nb_contrast))

    tuningCInhib = np.zeros((int(nb_cells/4),nb_degree,nb_contrast,trials))
    relativTCInhib = np.zeros((int(nb_cells/4),nb_degree,nb_contrast))
   
    parameters_Exc = np.zeros((nb_cells,3,nb_contrast))

    for i in range(nb_contrast):
        tuningCExc[:,:,i,:],parameters_Exc[:,:,i] = calcTuningCC(frExc[:,:,:,:,i,:])
        tuningCgExcE[:,:,i,:] = calcTuningC_variable(gExc_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i])
        tuningCgInhE[:,:,i,:] = calcTuningC_variable(gInh_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i])
 
        #tuningCInhib[:,:,i,:] = frInh[:,:,:,:,i,:]

        meanCExc = np.mean(tuningCExc,axis=3)
        #meanCInhib = np.mean(tuningCInhib,axis=3)
        meanCgExc = np.mean(tuningCgExcE,axis=3)
        meanCgInh = np.mean(tuningCgInhE,axis=3)
        relativTCExc[:,:,i] = calcRelativTuningCurve(meanCExc[:,:,i])
        #relativTCInhib[:,:,i] = calcRelativTuningCurve(meanCInhib[:,:,i])
        relativTCgExcE[:,:,i] = calcRelativTuningCurve(meanCgExc[:,:,i])
        realtivTCgInhE[:,:,i] = calcRelativTuningCurve(meanCgInh[:,:,i])


    np.save('./work/TuningCurves_sinus_Exc',tuningCExc)
    #np.save('./work/TuningCurves_sinus_Inh',tuningCInhib)
    np.save('./work/TuningCurves_sinus_Exc_gE',tuningCgExcE)
    np.save('./work/TuningCurves_sinus_Exc_gI',tuningCgInhE)
    np.save('./work/TuningCurves_sinus_Exc_parameters',parameters_Exc)

    np.save('./work/TuningCurvesRelativ_sinus_Exc',relativTCExc)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_gE',relativTCgExcE)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_gI',realtivTCgInhE)
    #np.save('./work/TuningCurvesRelativ_sinus_Inh',relativTCInhib)


    plotSingleTuningCurves(relativTCExc[:,:,nb_contrast-1],'excitatory')
    #plotSingleTuningCurves(relativTCInhib[:,:,nb_contrast-1],'inhibitory')
    plotAllTuningCurves(relativTCExc[:,:,nb_contrast-1],'excitatory',maxDegree)
    #plotAllTuningCurves(relativTCInhib[:,:,nb_contrast-1],'inhibitory',maxDegree)
    plotMeanTC(relativTCExc,'excitatory')
#------------------------------------------------------------------------------
if __name__=="__main__":
    data = (sys.argv[1:])
    duration = 125.0
    maxInput = 1
    if len(data) > 0:
        duration = float(data[0])
        maxInput = float(data[1])

    #try:
    startTuningCurves(duration,maxInput)
    #except:
    #    print('Error in validate tuning curves')

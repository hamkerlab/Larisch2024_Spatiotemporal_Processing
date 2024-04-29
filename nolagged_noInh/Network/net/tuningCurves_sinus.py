from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib.pyplot as plt
import Gabor_sinus as gabor
import os.path

### Create the SNN without the RGC to show some sinus gratings to obtain the orientation selectivity
## See Larisch et al. (2021) Plos Comp. for a deeper description

#------------------------global Variables------------------------------------
patchsize = 18#8
n_LGN = patchsize*patchsize*2
nbr_V1N = patchsize*patchsize# because ON-OFF 
nbr_IL1N = int(nbr_V1N/4)
#---------------------------------neuron definitions-------------------------

## Neuron Model for LGN/Input Layer ##
params = """
EL = -70.6
VTrest = -50.4
taux = 15.0
"""

inpt_eqs ="""
    dg_exc/dt = EL/1000 : min=EL, init=-70.6
    Spike =if g_exc > -50.4: 1 else: 0.0
    dresetvar / dt = 1/(1.0) * (-resetvar)
    dxtrace /dt = - xtrace /taux
    """

spkNeurLGN = Neuron(parameters=params,
                          equations=inpt_eqs,
                          reset="""g_exc=EL 
                                   Spike = 1.0
                                   resetvar=1.0
                                   xtrace+=1/taux""", 
                          spike="""g_exc > VTrest""")

## Neuron Model for V1-Layer, after Clopath et al.(2008) ##
params = """
gL = 30.0
DeltaT = 2.0 
tauw = 144.0 
a = 4.0 
b = 0.0805 
EL = -70.6 
C = 281.0 
tauz = 40.0
tauVT= 50.0
Isp = 400.0
VTMax = -30.4 
VTrest = -50.4
taux = 15.0 
tauLTD = 10.0
tauLTP= 7.0 
taumean = 750.0 
tau_gExc = 1.0
tau_gInh = 10.0 
"""

neuron_eqs = """
noise = Normal(0.0,1.0)
dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc-g_Inh: init = -70.6
dvmean/dt = ((vm - EL)**2 - vmean)/taumean    :init = 0.0
dumeanLTD/dt = (vm - umeanLTD)/tauLTD : init=-70.0
dumeanLTP/dt = (vm - umeanLTP)/tauLTP : init =-70.0
dxtrace /dt = (- xtrace )/taux
dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw : init = 0.0
dz/dt = if state==1:-z+Isp-10 else:-z/tauz  : init = 0.0
dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT  : init=-50.4
dg_Exc/dt = 1/tau_gExc * (-g_Exc)
dg_Inh/dt = 1/tau_gInh*(-g_Inh)
state = if state > 0: state-1 else:0
Spike = 0.0
dresetvar / dt = 1/(1.0) * (-resetvar)
vmTemp = vm
           """
#
#if state>=2:0 else: if state==1:-5 + (VTrest - VT)/tauVT else:
spkNeurV1 = Neuron(parameters = params,equations=neuron_eqs,spike="""(vm>VT) and (state == 0.0)""",
                         reset="""vm = 29.4
                                  state = 2.0                      
                                  Spike = 1.0
                                  resetvar = 1.0
                                  xtrace+= 1/taux""")

#----------------------------------synapse definitions----------------------

#----- Synapse from Poisson to Input-Layer -----#
inputSynapse =  Synapse(
    parameters = "",
    equations = "",
    pre_spike = """
        g_target += w
                """
)

#-----------------------population defintions-----------------------------------#
popInput = PoissonPopulation(geometry=(patchsize,patchsize,2),rates=50.0)
popLGN = Population(geometry=(patchsize,patchsize,2),neuron=spkNeurLGN )
popV1 = Population(geometry=nbr_V1N, neuron = spkNeurV1)
popIL1 = Population(geometry=nbr_IL1N, neuron = spkNeurV1)

#-----------------------projection definitions----------------------------------
#projPreLayer_PostLayer
projInput_LGN = Projection(
    pre = popInput,
    post = popLGN,
    target = 'exc',
    synapse = inputSynapse
).connect_one_to_one(weights = 30.0)

projLGN_V1 = Projection(
    pre=popLGN, 
    post=popV1, 
    target='Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projLGN_IL1 = Projection(
    pre=popLGN, 
    post=popIL1, 
    target='Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projV1_IL1 = Projection(
    pre = popV1,
    post = popIL1,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projIL1_V1 = Projection(
    pre = popIL1,
    post= popV1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projIL1_Lat = Projection(
    pre = popIL1,
    post = popIL1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

def loadWeights():
    projLGN_V1.w  = np.loadtxt('Input_network/V1weight.txt')
    projLGN_IL1.w = np.loadtxt('Input_network/InhibW.txt')
    projV1_IL1.w  = np.loadtxt('Input_network/V1toIN.txt')
    projIL1_V1.w  = np.loadtxt('Input_network/INtoV1.txt')
    projIL1_Lat.w = np.loadtxt('Input_network/INLat.txt')

#-------------------------------------------------------------------------------
def createInput(parameters,maxInput):
    inputIMG = gabor.createGaborMatrix(parameters,patchsize,patchsize)
    maxVal = np.max(np.abs(inputIMG))
    sizeX,sizeY=np.shape(inputIMG)
    inputPatch = np.zeros((sizeX,sizeY,2))
    inputPatch[:,:,0] =np.clip(inputIMG,0,np.max(inputIMG))
    inputPatch[:,:,1] =np.abs(np.clip(inputIMG,np.min(inputIMG),0))
    #print(inputPatch[:,:,1])
    #plt.figure()
    #plt.imshow(inputIMG,cmap=plt.get_cmap('gray'))
    #plt.savefig('Output/img_'+str(parameters[1])+'_'+str(parameters[5])+'_.png')

    popInput.rates = (inputPatch/maxVal)*maxInput

#--------------------------------------------------------------------------------
def runNet(maxDegree,stepDegree,stepPhase,stepFrequency,duration,maxInput):
    compile()
    loadWeights()
    phaseShifts = np.pi / float(stepPhase)
    nbrOfInputs = int(maxDegree/stepDegree)
    frequShift = (0.15 - 0.05)/float(stepFrequency)  #from 0.5 to 1.5 * patchsize#
    repeats = 50
    contrastLVLs = 7

    V1Mon=Monitor(popV1,['spike','g_Exc','g_Inh'])
    rec_frEx= np.zeros((nbr_V1N,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_gExc= np.zeros((nbr_V1N,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_gInh= np.zeros((nbr_V1N,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))

    InhibMon=Monitor(popIL1,['spike'])
    rec_frInh= np.zeros((nbr_IL1N,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    orientation = np.zeros((nbrOfInputs))
    maxFrExc = np.zeros(nbr_V1N)
    maxFrInh = np.zeros(nbr_IL1N)
    
    
    for i in range(nbrOfInputs):
        orientation[i] = stepDegree*np.pi/180.0*i
        print("Orientation: "+str(orientation[i]))
        for k in range(stepPhase):
            for f in range(stepFrequency):
                for c in range(contrastLVLs):
                    freq = (0.05+ (0.05*f))*patchsize 
                    phase = phaseShifts*k
                    #parameters = np.array((1.,0.0,0.13,0.2,np.pi/2.0,np.pi/2.0,patchsize,patchsize,0.0))
                    parameters = np.array((1.,0.0,0.13,0.2,0.1*patchsize,np.pi/2.0,patchsize,patchsize,0.0))
                    parameters[1] = orientation[i]
                    parameters[4] = freq
                    parameters[5] = phase
                    parameters[6] = patchsize/2
                    parameters[7] = patchsize/2
                    inputFR = maxInput/contrastLVLs + (maxInput/contrastLVLs)*c
                    for r in range(repeats):
                        createInput(parameters,inputFR)
                        simulate(duration)
                        spikesEx = V1Mon.get('spike')
                        gExcEx = V1Mon.get('g_Exc')
                        gInhEx = V1Mon.get('g_Inh')
                        spikesInh = InhibMon.get('spike')
                        for j in range(nbr_V1N):
                            rateEx = len(spikesEx[j])*1000/duration
                            rec_frEx[j,i,k,f,c,r] = rateEx
                            rec_Exc_gExc[j,i,k,f,c,r] = np.mean(gExcEx[:,j])
                            rec_Exc_gInh[j,i,k,f,c,r] = np.mean(gInhEx[:,j])
                            if (j < (nbr_IL1N)):
                                rateInh = len(spikesInh[j])*1000/duration
                                rec_frInh[j,i,k,f,c,r] = rateInh

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
def plotGabor(parameters):
    #parameters =np.array((1.,np.pi/2.0,0.2,0.2,np.pi/2.0,np.pi/2.0,6.0,6.0,0.0))
    inputIMG = gabor.createGaborMatrix(parameters,patchsize,patchsize)
    plt.figure()
    plt.imshow(inputIMG,cmap=plt.get_cmap('gray'),interpolation='none')
    plt.savefig('./Output/TC_sinus/TC_Gabor.png',bbox_inches='tight', pad_inches = 0.1)
#--------------------------------------------------------------------------------
def startTuningCurves(duration,maxInput):
    print('start to determine the tuning Curves over sinus gratings')
    print('Simulation time : '+str(duration)+' ms')
    print('maximum Input: '+str(maxInput)+ ' Hz')
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
    parameters_Inh = np.zeros((int(nb_cells/4),3,nb_contrast))


    for i in range(nb_contrast):
        tuningCExc[:,:,i,:],parameters_Exc[:,:,i] = calcTuningCC(frExc[:,:,:,:,i,:])
        tuningCInhib[:,:,i,:], parameters_Inh[:,:,i] = calcTuningCC(frInh[:,:,:,:,i,:])
        tuningCgExcE[:,:,i,:] = calcTuningC_variable(gExc_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i])
        tuningCgInhE[:,:,i,:] = calcTuningC_variable(gInh_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i])
 
        #tuningCInhib[:,:,i,:] = frInh[:,:,:,:,i,:]

        meanCExc = np.mean(tuningCExc,axis=3)
        meanCInhib = np.mean(tuningCInhib,axis=3)
        meanCgExc = np.mean(tuningCgExcE,axis=3)
        meanCgInh = np.mean(tuningCgInhE,axis=3)
        relativTCExc[:,:,i] = calcRelativTuningCurve(meanCExc[:,:,i])
        relativTCInhib[:,:,i] = calcRelativTuningCurve(meanCInhib[:,:,i])
        relativTCgExcE[:,:,i] = calcRelativTuningCurve(meanCgExc[:,:,i])
        realtivTCgInhE[:,:,i] = calcRelativTuningCurve(meanCgInh[:,:,i])


    np.save('./work/TuningCurves_sinus_Exc',tuningCExc)
    np.save('./work/TuningCurves_sinus_Inh',tuningCInhib)
    np.save('./work/TuningCurves_sinus_Exc_gE',tuningCgExcE)
    np.save('./work/TuningCurves_sinus_Exc_gI',tuningCgInhE)
    np.save('./work/TuningCurves_sinus_Exc_parameters',parameters_Exc)
    np.save('./work/TuningCurves_sinus_Inh_parameters',parameters_Inh)

    np.save('./work/TuningCurvesRelativ_sinus_Exc',relativTCExc)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_gE',relativTCgExcE)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_gI',realtivTCgInhE)
    np.save('./work/TuningCurvesRelativ_sinus_Inh',relativTCInhib)

#------------------------------------------------------------------------------
if __name__=="__main__":
    data = (sys.argv[1:])
    duration = 125.0
    maxInput = 100.0#75.0
    if len(data) > 0:
        duration = float(data[0])
        maxInput = float(data[1])

    #try:
    startTuningCurves(duration,maxInput)
    #except:
    #    print('Error in validate tuning curves')

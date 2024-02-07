import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import Gabor as gabor
from ANNarchy import *
#------------------------global Variables------------------------------------
duration = 500#200#125
patchsize = 12#8
numberOfNeurons = patchsize*patchsize# because ON-OFF 
#----------------------------------------------------------------------------
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
    dxtrace /dt = 1/taux * (- xtrace )
    """

spkNeurLGN = Neuron(parameters=params,
                          equations=inpt_eqs,
                          reset="""g_exc=EL 
                                   Spike = 1.0
                                   resetvar=1.0
                                   xtrace+=(-xtrace+1)/taux""", 
                          spike="""g_exc > VTrest""")

## Neuron Model for V1-Layer, after Clopath et al.(2008) ##
params = """
gL = 30.0
DeltaT = 2.0 
tauw = 144.0 
a = 4.0 
b = 0.805 
EL = -70.6 
C = 281.0 
tauz = 40.0
tauVT= 50.0
Isp = 400.0
VTMax = 30.4 
VTrest = -50.4
taux = 15.0 
tauLTD = 10.0
tauLTP= 7.0 
taumean = 1200.0 
tau_gEx = 1.1
tau_gInh = 1.1 
"""

neuron_eqs = """
dvm/dt =if state>=2:0 else: 1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z) + g_Exc - g_Inh  : init = -70.6
dwad/dt = 1/tauw * (a * (vm - EL) - wad) : init = 0.0
dz/dt = -z/tauz  : init = 0.0
dVT/dt =if state>=1:0 else: (VTrest - VT)/tauVT  : init=-50.4
dvmean/dt = 1/taumean * ( (vm - EL ) **2.0 - vmean)    :init = 70.0

dg_Exc/dt = 1/tau_gEx * (-g_Exc)
dg_Inh/dt = 1/tau_gInh*(-g_Inh)

Spike =if vm > VT : 1 else: 0.0
state = if state >0: state-1 else : 0
dresetvar / dt = 1/(1.0) * (-resetvar)
dumeanLTD/dt = 1/tauLTD * (vm - umeanLTD) : init=-70.0
dumeanLTP/dt = 1/tauLTP * (vm - umeanLTP) : init =-70.0
dxtrace /dt = 1/taux * (- xtrace )
           """
spkNeurV1 = Neuron(parameters = params,equations=neuron_eqs,spike="""vm>VT""",
                         reset="""vm = 29.4
                                  state = 2.0
                                  wad += b
                                  z = Isp
                                  VT = VTMax
                                  Spike = 1.0
                                  resetvar = 1.0
                                  xtrace+=(-xtrace+1)/taux""")

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
popV1 = Population(geometry=numberOfNeurons, neuron = spkNeurV1)
popInhibit = Population(geometry=numberOfNeurons/4, neuron = spkNeurV1)

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


projV1_Inhib = Projection(
    pre = popV1,
    post = popInhibit,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projInhib_V1 = Projection(
    pre = popInhibit,
    post= popV1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projInhib_Lat = Projection(
    pre = popInhibit,
    post = popInhibit,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

#----------------------------further functions---------------------------------
def loadWeights():
    projLGN_V1.w = np.loadtxt('Input_network/V1weight.txt')
    projV1_Inhib.w = np.loadtxt('Input_network/V1toIN.txt')
    projInhib_V1.w = np.loadtxt('Input_network/INtoV1.txt')
    projInhib_Lat.w = np.loadtxt('Input_network/INLat.txt')
#-------------------------------------------------------------------------------
def createInput(parameters):
    inputIMG = gabor.createGaborMatrix(parameters,patchsize,patchsize)
    maxW = np.max(inputIMG)
    minW = np.min(inputIMG)
    sizeX,sizeY=np.shape(inputIMG)
    inputPatch = np.zeros((sizeX,sizeY,2))
    inputPatch[:,:,0] =np.clip(inputIMG,0,np.max(inputIMG))
    inputPatch[:,:,1] =np.abs(np.clip(inputIMG,np.min(inputIMG),0))
    popInput.rates = inputPatch*125.0
#------------------------------------------------------------------------------
def startDetermineRF():
    # approximate the receptive fields of the inhibitory neurons over the variations of gabor inputs
    # variation over orientation[0,np.pi], position[0,patchsize] and frequency[0.1,np.pi]
    # variation  
    print('Start to approximate receptive filds of inhibitory neurons')
    compile()
    loadWeights()
    #--------init neuron recording--------#
    stepSizeO=20
    stepSizeS = 1
    nbrOfInputs = (180/stepSizeO) * ((patchsize/stepSizeS)**2) *2 #* (np.pi/(np.pi/4)**2)
    print(nbrOfInputs)
    InhibMon=Monitor(popInhibit,['spike','g_Exc','g_Inh'])
    rec_frInh= np.zeros((numberOfNeurons/4,nbrOfInputs+1))
    parameterArray = np.zeros((nbrOfInputs+1,9))
    i = 0
    for x in np.arange(0,patchsize,stepSizeS): #shift over x-axis
        for y in np.arange(0,patchsize,stepSizeS): #shift over y-axis
            for a in np.arange(0,180,stepSizeO): #orientation shift
                for ph in np.arange(np.pi/2,np.pi+np.pi,np.pi): #phase shift
                    #for f in np.arange(0.0,np.pi,np.pi/4): #frequenvy shift
                    parameters = np.array((1.,a*np.pi/180.0,0.2,0.4,np.pi/2.0,ph,x,y,0.0))    
                    parameterArray[i,:] = parameters
                    createInput(parameters)
                    simulate(duration)
                    spikesInh = InhibMon.get('spike')
                    for j in range(numberOfNeurons/4):
                        rateInh = len(spikesInh[j])*1000/duration
                        rec_frInh[j,i] = rateInh
                    i+=1
    print(i)
    #--- match the approximated gabor parameter to the inhibitory neuron ---#
    matchParameters = np.zeros((numberOfNeurons/4,9))
    for i in range(numberOfNeurons/4):
        maxPatchNbr = np.where(rec_frInh[i,:] == np.max(rec_frInh[i,:]))
        matchParameters[i,:] = np.sum(parameterArray[maxPatchNbr[0],:],axis=0)/len(maxPatchNbr[0])
        #matchParameters[i,:] = parameterArray[maxPatchNbr[0][0]]
    testGab = gabor.createGaborMatrix(matchParameters[12],patchsize,patchsize)
    plt.figure()
    plt.imshow(testGab,cmap=plt.get_cmap('gray'),interpolation='none')
    plt.savefig('./test.png')

    np.save('./work/GaborTest_Parameters',parameterArray)
    np.save('./work/parameter_inhib2',matchParameters)
    print('Finish!')
#------------------------------------------------------------------------------
if __name__=="__main__":
    startDetermineRF()

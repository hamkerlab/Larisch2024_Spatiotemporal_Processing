import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt

#
# model, consisting of RGC, LGN and V1 L4
#
#

#from ANNarchy import Neuron,PoissonPopulation,Population,Projection,Synapse,Uniform
from ANNarchy import *
import numpy as np
#------------------------global Variables------------------------------------
filter_size= 5 # size of the DoG Filter for RGC
filter_LGN = 5 # first dimension of the LGN RF => filter_LGN x filter_LGN == input matrix to one(!) LGN cell
center_LGN = 3

sigma_1 = 0.3*filter_size
sigma_2 = 0.4*filter_size
h_LGN = 18 # first dimension of the LGN population !
stride_RGC = (int(sigma_1)+1)
stride_LGN = 1

n_LGN = h_LGN*h_LGN*2 # total number of all LGN neurons 
n_E1 = h_LGN*h_LGN # number of neurons in Layer E1
n_I1 = int(n_E1/4) # number of neurons on Layer I1



h_RGC = filter_LGN+(h_LGN*stride_LGN -1 ) # first dimension of the complete (!) RGC population
s_input = filter_size+((stride_RGC*h_RGC)-1) #(filter_size*filter_LGN) - (filter_size-stride_RGC)
print('Input size=%i'%s_input)

#---------------------------------neuron definitions-------------------------
## adaptive exponential-integrate and fire neuron for RGC ##

spkNeurRGC = Neuron(
    parameters="""
        v_rest = -58        :population
        cm = 200.0          :population
        tau_m = 9.3667      :population
        tau_syn_E = 5.0     :population    
        tau_syn_I = 5.0     :population    
        e_rev_E = 0.0       :population
        e_rev_I = -80.0     :population
        tau_w = 120         :population
        a = 0.03            :population
        b = 100             :population
        gL = 10             :population
        i_offset = 0.0      :population
        delta_T = 2.0       :population
        v_thresh = -50.4    :population    
        v_reset = -46.0     :population
        v_spike = -40.0     :population
    """,
    equations= """
        I = g_exc - g_inh + i_offset            
        dv/dt = ( -gL*(v-v_rest) + gL*delta_T*exp((v-v_thresh)/delta_T) + I - w)/cm  : init=-58      
        tau_w * dw/dt = a * (v - v_rest) - w           
        tau_syn_E * dg_exc/dt = - g_exc : exponential
        tau_syn_I * dg_inh/dt = - g_inh : exponential
    """,
    spike = """
        v >= v_spike
    """,
    reset = """         
        v = v_reset
        w += b"""
)
# Leaky Integrate and Fire LGN Neuron
spkNeurLGN = Neuron(
    parameters="""
        v_rest = -60        :population
        cm = 200.0          :population
        tau_m = 9.3667      :population
        tau_syn_E = 5.0     :population    
        tau_syn_I = 5.0     :population    
        e_rev_E = 0.0       :population
        e_rev_I = -80.0     :population
        tau_w = 50         :population
        a = 2.0            :population
        b = 10             :population
        gL = 10             :population
        i_offset = 0.0      :population
        delta_T = 2.0       :population
        v_thresh = -50.4    :population    
        v_reset = -46.0     :population
        v_spike = -40.0     :population
    """,
    equations= """
        I = g_exc - g_inh + i_offset            
        dv/dt = ( -gL*(v-v_rest) + gL*delta_T*exp((v-v_thresh)/delta_T) + I - w)/cm  : init=-60       
        tau_w * dw/dt = a * (v - v_rest) - w           
        tau_syn_E * dg_exc/dt = - g_exc : exponential
        tau_syn_I * dg_inh/dt = - g_inh : exponential
    """,
    spike = """
        v >= v_spike
    """,
    reset = """         
        v = v_reset
        w += b"""
)

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

# Rate-Neuron
rateNeuron = Neuron(
                equations=""" r = sum(exc) """ #*5
            )

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

popIMG = Population(geometry=(s_input,s_input),neuron=Neuron(parameters="r = 0.0")) # population to set the input image/ pixel value determines the firing rate
# DoG projection (see below)
popON = Population(geometry=(h_RGC,h_RGC),neuron=rateNeuron) # ON-Center Population
popOFF = Population(geometry=(h_RGC,h_RGC),neuron=rateNeuron) # OFF-Center Population
# currentInjection (see below)
popRGC = Population(geometry=(h_RGC,h_RGC,2),neuron=spkNeurRGC)
popLGN = Population(geometry=(h_LGN,h_LGN,2),neuron=spkNeurLGN ) #  additional population to create the x-trace
popE1 = Population(geometry=n_E1,neuron=spkNeurV1, name="E1")
popIL1 = Population(geometry=n_I1, neuron = spkNeurV1, name="I1")


#---------------- DoG Custom Projection Definition ----------------------------#
    # create the connectivity pattern from input to RGC
def DoG_pattern(prePop, postPop, sigma_1, sigma_2, polar, amplitude, stride): # polar means polarity -> 1: ON-Center -1: OFF-Center

    # projection definition for a DoG-Filter

    # First: create a DoG filter as template for the weight matrices
    r = np.arange(filter_size)+1 # radius
    x = np.tile(r, [filter_size,1]) # copy for each row
    y = x.T
    spatial_diff = (x - filter_size/2 - 0.5)**2 + (y - filter_size/2 - 0.5)**2 # center the DoG

    dog_filter = 1/np.sqrt(2*np.pi) * (1/sigma_1 * np.exp(-spatial_diff/(2*(sigma_1**2))) - 1/sigma_2 * np.exp(-spatial_diff/(2*(sigma_2**2))))

    center_mask = dog_filter>0.0 # where the DoG is over zero, there is the center ?!

    dog_filter -= np.mean(dog_filter[:])
    dog_filter /= np.amax(dog_filter[:])

    dog_filter = dog_filter*amplitude*polar 

    dog_filter = np.reshape(dog_filter,filter_size*filter_size)

    # create the weight matrices depending on the DoG-Filter
    synapses = CSR()

    str_X = stride[0]
    str_Y = stride[1]
    # iterate through all postsynaptic neurons
    for w_post in range(postPop.geometry[0]):
        for h_post in range(postPop.geometry[1]):
            delays = np.random.randint(30,50,(filter_size,filter_size))# Delays on DoG
            delays[center_mask] = 1.0 #very small delays for DoG-center
            delays = np.reshape(delays,filter_size*filter_size)

            # get a populationview of the prePop for the corresponding neurons
            post_rank = postPop.rank_from_coordinates((w_post,h_post))
            subPrePop = prePop[0+w_post*str_X:filter_size+w_post*str_X,0+h_post*str_Y:filter_size+h_post*str_Y]      
            
            pre_ranks=subPrePop.rank # get the ranks

            weights= dog_filter# set the weights by use the reshaped the DoG-Filter #[1 for i in range(len(pre_ranks)) ]#
            delays = delays# [0 for i in range(len(pre_ranks)) ] # set delays to zero


            synapses.add(post_rank,pre_ranks,weights,delays)

    return synapses
#----------------------- Custom projection from RGC to LGN ---------------------#
# create the connectivity from the LGN to the nxn RGC population
def RGCtoLGN(prePop,postPop, s, c, stride):
    synapses = CSR()

    str_X = stride[0]
    str_Y = stride[1]


    x, y = np.meshgrid(np.linspace(-1,1,filter_LGN), np.linspace(-1,1,filter_LGN))
    d = np.sqrt(x*x+y*y) 
    sigma, mu = 1.2, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )*10.0

    #weights = np.ones((filter_LGN,filter_LGN))*10
    #weights[1,1] *=3.0 # improve weights from the center RGC 
    weights = g*8.0
    print(weights)
    weights = np.reshape(weights,filter_LGN*filter_LGN)
    for w_post in range(postPop.geometry[0]):
        for h_post in range(postPop.geometry[1]):
            post_rank = postPop.rank_from_coordinates((w_post,h_post,0),local=True)
            subPrePop_surr = prePop[0+(w_post*str_X):filter_LGN+(w_post*str_X),0+(h_post*str_Y):filter_LGN+(h_post*str_Y),s] # RGC for surround
            subPrePop_cent = prePop[0+(w_post*str_X):filter_LGN+(w_post*str_X),0+(h_post*str_Y):filter_LGN+(h_post*str_Y),c] # RGC for center

            pos = int(filter_LGN/2)-1 # start position for the LGN center connections, for even numbers
            cor_geomX = np.asarray([pos,pos+center_LGN]) + (w_post*str_X)
            cor_geomY = np.asarray([pos,pos+center_LGN]) + (h_post*str_Y)

            old_Rank_pop= prePop[cor_geomX[0]:cor_geomX[1],cor_geomY[0]:cor_geomY[1],s]
            new_Rank_pop= prePop[cor_geomX[0]:cor_geomX[1],cor_geomY[0]:cor_geomY[1],c]

            # list of ranks for the center neurons
            # old center ranks to find them in the list
            center_ranks_old = old_Rank_pop.rank
            # new center ranks to set them in the list
            center_ranks_new = new_Rank_pop.rank

            # list of ranks for the surround neurons (it is a list! always!)
            pre_ranks_surr= subPrePop_surr.rank
            # list of ranks for the neurons to connect with for the custom connectivity pattern
            pre_ranks = pre_ranks_surr

            # get the index of the old center cells
            i_dx =np.where(np.isin(pre_ranks,center_ranks_old))[0]
            for i in range(len(i_dx)):
                # set the new center cells on the correct positions in the rank list
                pre_ranks[i_dx[i]] = center_ranks_new[i] # set center neuron in a 3x3 grid

            delays = np.random.randint(1,3,(filter_LGN*filter_LGN))#np.ones((filter_LGN*filter_LGN))
            
            synapses.add(post_rank,pre_ranks,weights,delays)

    return synapses

#-----------------------projection definitions----------------------------------
projDoG_ON = Projection(
    pre = popIMG,
    post= popON,
    target = 'exc'
).connect_with_func(method=DoG_pattern, sigma_1=sigma_1, sigma_2=sigma_2, polar=1,amplitude=50,stride=(stride_RGC,stride_RGC))

projDoG_OFF = Projection(
    pre = popIMG,
    post= popOFF,
    target = 'exc'
).connect_with_func(method=DoG_pattern, sigma_1=sigma_1, sigma_2=sigma_2, polar=-1,amplitude=50,stride=(stride_RGC,stride_RGC))

popRGC_ON = popRGC[:,:,0]#ON-Center RGC Population
projCurrInject_ON = CurrentInjection(
    pre = popON,
    post = popRGC_ON,
    target='exc'
).connect_current()

popRGC_OFF = popRGC[:,:,1]#OFF-Center RGC Population
projCurrInject_OFF = CurrentInjection(
    pre = popOFF,
    post = popRGC_OFF,
    target='exc'
).connect_current()

popLGN_ON = popLGN[:,:,0]#ON-Center LGN Population
projInput_LGN_ON = Projection(
    pre = popRGC,
    post = popLGN_ON,
    target = 'exc',
    synapse = inputSynapse
).connect_with_func(method=RGCtoLGN, s=1, c=0, stride=(stride_LGN,stride_LGN))

popLGN_OFF = popLGN[:,:,1]#OFF-Center LGN Population
projInput_LGN_OFF = Projection(
    pre = popRGC,
    post = popLGN_OFF,
    target = 'exc',
    synapse = inputSynapse
).connect_with_func(method=RGCtoLGN, s=0, c=1, stride=(stride_LGN,stride_LGN))

## projections from LGN to V1 L4
projLGN_V1 = Projection(
    pre=popLGN, 
    post=popE1, 
    target='Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,4.0), delays = Uniform(1.0, 25) )

projLGN_Inh = Projection(
    pre=popLGN, 
    post=popIL1, 
    target='Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0), delays = Uniform(1.0, 25) )

projV1_Inhib = Projection(
    pre = popE1,
    post = popIL1,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))#, delays = Uniform(1.0, 2) )

projInhib_V1 = Projection(
    pre = popIL1,
    post= popE1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))#, delays = Uniform(1.0, 2) )

projInhib_Lat = Projection(
    pre = popIL1,
    post = popIL1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))#, delays = Uniform(1.0, 2) )



def loadWeights():
    projLGN_V1.w = np.loadtxt('Input_network/V1weight.txt')
    projLGN_Inh.w = np.loadtxt('Input_network/InhibW.txt')
    projV1_Inhib.w = np.loadtxt('Input_network/V1toIN.txt')
    projInhib_V1.w = np.loadtxt('Input_network/INtoV1.txt')#*0.0
    projInhib_Lat.w = np.loadtxt('Input_network/INLat.txt')#*0.0


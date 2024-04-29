from ANNarchy import *
setup(dt=1.0,seed=101)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from net_EB_LGN_DoG import *
import Gabor_sinus as grating
import pickle
from tqdm import tqdm

#####
# Present moving gratings
# to measure DS 
#####


def createInput(degree,totalT,presentT,spat_speed,amplitude=0.3,s_f=1.12):
   
    #spat_speed -> speed of grating
    edge = int(s_input*0.5)
    image_size = s_input+edge  #h_LGN +2
    #s_f = 1.12#0.048*s_input#0.3#
    n_img = int(1000/presentT)   # number of patches, which are neccessary to present one second
    n_img_total = n_img*int(totalT/1000)
    step = ((image_size/(n_img-1))/(s_f))*(spat_speed) #((image_size/(n_img-1))/(0.12*image_size))*spat_speed
    #0.048*image_size
    # parameters for sinus grating               #0.12*image_size#np.pi/2.0
    parameters = np.array((1.0,0.0,0.13,0.2,s_f,np.pi/2.0,image_size,image_size,0.0))
    
    parameters[1] = 0.0#degree*np.pi/180

    # create a episode of images, present the grating at the orientation, sequence contain the moving
    img_list = np.zeros((n_img_total,s_input,s_input))
    for i in range(n_img_total): 
        parameters[6] = i*step# x_center position
        img = grating.createGaborMatrix(parameters,image_size,image_size)
        img = ndimage.rotate(img,degree,reshape=False,mode='wrap')
        #print(np.max(img),np.min(img))   
        
        img += 1#np.abs(np.min(img))
        img *= amplitude
        #print(np.max(img),np.min(img))
        img  = img[int(edge/2):int(edge/2)+s_input,int(edge/2):int(edge/2)+s_input]
        img_list[i] = img

    return(img_list)        

def main():

    if not os.path.exists('Output/direct_gratingSinus/'):
        os.mkdir('Output/direct_gratingSinus/')

    repeats = 5
    totalT = 3000
    n_spatF = 5
    spat_speed = np.linspace(2,6,n_spatF) #[0.25,0.5,1,2,4,8,16,32] #np.linspace(2,8,n_spatF)#2  # spatial Frequency in Hz ->speed (!)
    print(spat_speed)
    n_spatF = len(spat_speed)
    presentT = 20#int(1000/spat_speed)
    

    s_deg = 18 # 18 degree steps
    n_degrees = int(360/s_deg) # number of steps

    amp_start= 0.11#0.16
    amplitude_step = 0.02#0.051
    n_amplitudes = 1

    # start the simulation
    compile()
    loadWeights()

    n_sf_steps = 5
    sf_steps = np.linspace(1,5,n_sf_steps) # spatial Frequency in cycl/pixel (cycl/img)

    mon_LGN = Monitor(popLGN,['spike'])
    mon_E1 = Monitor(popE1,['spike','vm','g_Exc','g_Inh'])
    mon_IL1 = Monitor(popIL1,['spike'])#,'vm','g_Exc','g_Inh'])


    rec_LGN = np.zeros((n_degrees,repeats,n_LGN))

    rec_E1 = np.zeros((n_degrees,repeats,n_E1))
    rec_E1_spikes = np.zeros((n_degrees,repeats,n_E1, totalT))
    rec_membPotEx = np.zeros((n_degrees,repeats,totalT,n_E1))
    rec_gExcEx = np.zeros((n_degrees,repeats,totalT,n_E1))
    rec_gInhEx = np.zeros((n_degrees,repeats,totalT,n_E1))


    rec_IL1 = np.zeros((n_degrees,repeats,n_I1))
    rec_I1_spikes = np.zeros((n_degrees,repeats,n_I1, totalT))
    rec_membPotIL1 = np.zeros((n_degrees,repeats,totalT,n_I1))
    rec_gExIL1 = np.zeros((n_degrees,repeats,totalT,n_I1))
    rec_gInIL1 = np.zeros((n_degrees,repeats,totalT,n_I1))

   
    ### lagged LGN cells over syn-Delays
    # add some extra delays to implement "lagged" LGN-Cells -> additional delay depends on t_delay !
    projInput_LGN_ON.delay = np.load('./work/LGN_ON_delay.npy')
    projInput_LGN_OFF.delay = np.load('./work/LGN_OFF_delay.npy')

    l_degrees = np.linspace(0,360-s_deg,n_degrees,dtype='int32')

    
    sim_parameter = {"totalT": totalT, "presenT": presentT,"speed":spat_speed, "n_degree":n_degrees,"spatFreq":sf_steps}
    pbar = tqdm(total=(n_amplitudes*n_spatF*n_sf_steps*repeats*n_degrees))

    for a in range(n_amplitudes):
        for sf in range(n_spatF):
            for tF in range(n_sf_steps):
                for r in range(repeats):
                    # choose randomly a degree
                    np.random.shuffle(l_degrees)            
                    for d in l_degrees:
                
                        # create a bar on a random degree
                        inp_list = createInput(d,totalT,presentT,spat_speed[sf],amp_start,sf_steps[tF])
                        n_ele = np.shape(inp_list)[0]             
                        for inp in range(n_ele):
                            popIMG.r = inp_list[inp]
                            simulate(presentT)

                        spk_LGN = mon_LGN.get('spike')
                        spk_E1 = mon_E1.get('spike')
                        spk_I1 = mon_IL1.get('spike')
                    


                        vmEx = mon_E1.get('vm')
                        gExcEx = mon_E1.get('g_Exc')
                        gInhEx = mon_E1.get('g_Inh')
                        

                        for n in range(n_LGN):
                            spk = spk_LGN[n]
                            rec_LGN[int(d//s_deg),r,n] = (len(spk)/totalT)*1000

                        for n in range(n_E1):
                            spk = spk_E1[n]
                            point_list = np.zeros(totalT)
                            point_list[spk] = 1
                            rec_E1_spikes[int(d//s_deg),r,n] = point_list
                            rec_E1[int(d//s_deg),r,n] = (len(spk)/totalT)*1000

                            if n < int(n_E1/4):
                                spk = spk_I1[n]
                                point_list = np.zeros(totalT)
                                point_list[spk] = 1
                                rec_I1_spikes[int(d//s_deg),r,n] = point_list
                                rec_IL1[int(d//s_deg),r,n] = (len(spk)/totalT)*1000


                        rec_membPotEx[int(d//s_deg),r] = vmEx#[0:totalT]
                        rec_gExcEx[int(d//s_deg),r] = gExcEx#[0:totalT]
                        rec_gInhEx[int(d//s_deg),r] = gInhEx#[0:totalT]

             
                        reset()
                        pbar.update(1)


                np.save('./work/directGrating_Sinus_SpikeCount_LGN_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_LGN, allow_pickle=False)
                np.save('./work/directGrating_Sinus_SpikeCount_E1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_E1, allow_pickle=False)
                np.save('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_E1_spikes, allow_pickle=False)
                np.save('./work/directGrating_Sinus_SpikeCount_I1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_IL1, allow_pickle=False)
                np.save('./work/directGrating_Sinus_SpikeTimes_I1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_I1_spikes, allow_pickle=False)

                np.save('./work/directGrating_Sinus_MembranPot_E1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_membPotEx, allow_pickle=False)
                np.save('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_gExcEx, allow_pickle=False)
                np.save('./work/directGrating_Sinus_gInh_E1_amp%i_spatF%i_tempF%i'%(a,sf,tF),rec_gInhEx, allow_pickle=False)

    pbar.close()
    np.savetxt('./work/directGrating_Sinus_parameter.txt',[totalT,1,presentT])
    pickle.dump(sim_parameter,open('./work/directGrating_Sinus_parameter.p',"wb"))

#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

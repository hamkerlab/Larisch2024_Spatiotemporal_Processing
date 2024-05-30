from ANNarchy import *
#setup(dt=1.0,seed=101)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np
from net_EB_LGN_DoG import *
import Gabor as bar
from scipy.special import factorial


#####
# Present moving gratings
# to measure DS 
#####


def createInput(degree,totalT,presentT):
   
    n_img = int(totalT/presentT)   # number of patches, which are neccessary
    speed = (h_LGN/(n_img-1))

    image_size = h_LGN +2 # for the 3x3 DoG-Filter

    # parameters for sinus grating
    parameters = np.array((1.,0.0,0.13,1.0,0.12*image_size,np.pi/2.0,image_size,image_size/2,1.0))
    parameters[1] = 0.0#degree*np.pi/180

    # create a episode of images, present the grating at the orientation, sequence contain the moving
    img_list = np.zeros((n_img,image_size,image_size))
    for i in range(n_img): 
        parameters[6] = i*speed# x_center position
        img = bar.createGaborMatrix(parameters,image_size,image_size)
        img = ndimage.rotate(img,degree,reshape=False,mode='wrap')
        maxVal = np.max(np.abs(img))    
        img /= maxVal    
        img_list[i] = img
        
        #print(np.max(img), np.min(img))
        #plt.figure()
        #plt.imshow(img,cmap='gray',interpolation='none')
        #plt.savefig('Output/direct_gratingBar/inpt_d1_%i_%i'%(degree,i))
        #plt.close()

    return(img_list)        

def main():

   
    if not os.path.exists('Output/direct_gratingBar/'):
        os.mkdir('Output/direct_gratingBar/')

    repeats = 10
    totalT = 2000
    presentT = 100
    
    maxInput = 70

    s_deg = 18 # 18 degree steps
    n_degrees = int(360/18) # number of steps

    # start the simulation
    compile()
    loadWeights()

    ### lagged LGN cells over syn-Delays
    # add some extra delays to implement "lagged" LGN-Cells -> additional delay depends on t_delay !
    n_LGN = 18*18*2
    lagged_cells = np.random.choice(n_LGN,int((n_LGN)*0.6),replace=False ) # chose random some "lagged" LGN cells
    d_old = np.asarray(projInput_LGN.delay) # get the old delays
    d_old[lagged_cells] = d_old[lagged_cells]+50 # add a delay of n ms
    #projInput_LGN.delay = d_old


    #mon_LGN = Monitor(popLGN,['spike'])
    mon_E1 = Monitor(popE1,['spike','vm','g_Exc','g_Inh'])
    mon_IL1 = Monitor(popIL1,['spike'])#,'vm','g_Exc','g_Inh'])
    mon_E2 = Monitor(popE2,['spike'])#,'vm','g_Exc','g_Inh'])
    mon_IL2 = Monitor(popIL2,['spike'])#,'vm','g_Exc','g_Inh'])


    rec_E1 = np.zeros((n_degrees,repeats,n_E1))
    rec_membPotEx = np.zeros((n_degrees,repeats,totalT,n_E1))
    rec_gExcEx = np.zeros((n_degrees,repeats,totalT,n_E1))
    rec_gInhEx = np.zeros((n_degrees,repeats,totalT,n_E1))

    rec_IL1 = np.zeros((n_degrees,repeats,n_I1))
    rec_E2 = np.zeros((n_degrees,repeats,n_E2))
    rec_IL2 = np.zeros((n_degrees,repeats,n_I2))



    l_degrees = np.linspace(0,360-s_deg,n_degrees,dtype='int32')



    for r in range(repeats):
        # choose randomly a degree
        np.random.shuffle(l_degrees)     
        if r%(repeats//10) == 0:
            print('On repetition = %i'%r)        
        for d in l_degrees:
  
            # create a bar on a random degree
            inp_list = createInput(d,totalT,presentT)
            n_ele = np.shape(inp_list)[0]             
            for inp in range(n_ele):
                popIMG.r = inp_list[inp]
                simulate(presentT)

            spk_E1 = mon_E1.get('spike')
            spk_E2 = mon_E2.get('spike')
            spk_I1 = mon_IL1.get('spike')
            spk_I2 = mon_IL2.get('spike')                        


            vmEx = mon_E1.get('vm')
            gExcEx = mon_E1.get('g_Exc')
            gInhEx = mon_E1.get('g_Inh')
            
            for n in range(n_E1):
                spk = spk_E1[n]
                rec_E1[int(d//s_deg),r,n] = len(spk)

                spk = spk_E2[n]
                rec_E2[int(d//s_deg),r,n] = len(spk)

                if n < int(n_E1/4):
                    spk = spk_I1[n]
                    rec_IL1[int(d//s_deg),r,n] = len(spk)

                    spk = spk_I2[n]
                    rec_IL2[int(d//s_deg),r,n] = len(spk)

            rec_membPotEx[int(d//s_deg),r] = vmEx
            rec_gExcEx[int(d//s_deg),r] = gExcEx
            rec_gInhEx[int(d//s_deg),r] = gInhEx

            #reset()
            


    np.save('./work/directGrating_Bar_SpikeCount_E1',rec_E1)
    np.save('./work/directGrating_Bar_SpikeCount_I1',rec_IL1)
    np.save('./work/directGrating_Bar_SpikeCount_E2',rec_E2)
    np.save('./work/directGrating_Bar_SpikeCount_I2',rec_IL2)


    np.save('./work/directGrating_Bar_MemnranPot_E1',rec_membPotEx)
    np.save('./work/directGrating_Bar_gExc_E1',rec_gExcEx)
    np.save('./work/directGrating_Bar_gInh_E1',rec_gInhEx)
#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

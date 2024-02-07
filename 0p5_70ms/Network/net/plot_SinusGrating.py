from ANNarchy import *
setup(dt=1.0,seed=101)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np
from net_EB_LGN_DoG import *
import Gabor_sinus as grating
from scipy.special import factorial


#####
# Present moving gratings
# to measure DS 
#####


def createInput(degree,totalT,presentT,speed,amplitude=0.3,s_f=1.12):
   
    #speed -> speed of grating
    edge = int(s_input*0.5)
    image_size = s_input+edge  #h_LGN +2
    #s_f = 1.12#0.048*s_input#0.3#
    n_img = int(1000/presentT)   # number of patches, which are neccessary to present one second
    n_img_total = n_img*int(totalT/1000)
    step = ((image_size/(n_img-1))/(s_f))*(speed) #((image_size/(n_img-1))/(0.12*image_size))*speed
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
        img /= np.amax(img)
        img *= amplitude
        #print(np.max(img),np.min(img))
        img  = img[int(edge/2):int(edge/2)+s_input,int(edge/2):int(edge/2)+s_input]
        img_list[i] = img

        plt.figure()
        plt.imshow(img_list[i],cmap='gray',interpolation='none')
        plt.colorbar()
        plt.savefig('Output/direct_gratingSinus/input/amp_%f/speed_%i/spatFreq_%f/grating_degree%i_%i'%(amplitude,speed,s_f,degree,i))
        plt.close()

    return(img_list)          

def main():

    if not os.path.exists('Output/direct_gratingSinus/'):
        os.mkdir('Output/direct_gratingSinus/')

    repeats = 2#200
    totalT = 3000
    n_spatF = 7
    speed = np.linspace(2,5,n_spatF)#2  # spatial Frequency in Hz ->speed (!)
    presentT = 20#int(1000/speed)
    

    s_deg = 24#18 # 18 degree steps
    n_degrees = 4#int(360/s_deg) # number of steps

    amp_start= 0.08#0.06
    amplitude_step = 0.02#0.051
    n_amplitudes = 1

    # start the simulation
    compile()
    loadWeights()

    n_sf_steps = 7
    sf_steps = np.linspace(0.75,2.25,n_sf_steps) # spatial Frequency in cycl/pixel (cycl/img)



    l_degrees = np.linspace(0,360-s_deg,n_degrees,dtype='int32')    

    if not os.path.exists('Output/direct_gratingSinus/input/'):
       os.mkdir('Output/direct_gratingSinus/input/')  

    for a in range(n_amplitudes):
        amp = amp_start+((a+1)*amplitude_step)
        if not os.path.exists('Output/direct_gratingSinus/input/amp_%f'%amp):
            os.mkdir('Output/direct_gratingSinus/input/amp_%f'%amp)    
        print('----')
        for sf in range(n_spatF):
            if not os.path.exists('Output/direct_gratingSinus/input/amp_%f/speed_%i'%(amp,speed[sf])):
                os.mkdir('Output/direct_gratingSinus/input/amp_%f/speed_%i'%(amp,speed[sf]))     
            for tF in range(n_sf_steps):
                if not os.path.exists('Output/direct_gratingSinus/input/amp_%f/speed_%i/spatFreq_%f'%(amp,speed[sf],sf_steps[tF])):
                    os.mkdir('Output/direct_gratingSinus/input/amp_%f/speed_%i/spatFreq_%f'%(amp,speed[sf],sf_steps[tF]))  

                for d in l_degrees:                
                    # create a bar on a random degree
                    inp_list = createInput(d,totalT,presentT,speed[sf],amp,sf_steps[tF])


#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from scipy import ndimage, misc
import numpy as np
import os
import Gabor_sinus as grating


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

        #plt.figure()
        #plt.imshow(img_list[i],cmap='gray',interpolation='none',vmin=0.0,vmax=0.2)
        #plt.colorbar()
        #plt.savefig('Output/direct_gratingSinus/inpt_d1_degree_%i_spatFreq_%i_sF_%f_%i.png'%(degree,spat_speed,s_f,i))
        #plt.close()

    return(img_list)      

def createVideo():

    print('Make a nice video with a sinus grating')



    input_imgs = []
    s_input = 48

    repeats = 2#200
    totalT = 1000
    n_spatF = 7
    speed = np.linspace(2,5,n_spatF)#2  # spatial Frequency in Hz ->speed (!)
    presentT = 20#int(1000/speed)
    

    s_deg = 24#18 # 18 degree steps
    n_degrees = int(360/s_deg) # number of steps

    amp_start= 0.08#0.06
    amplitude_step = 0.02#0.051
    n_amplitudes = 1

    n_sf_steps = 7
    sf_steps = np.linspace(0.75,2.25,n_sf_steps) # spatial Frequency in cycl/pixel (cycl/img)



    l_degrees = np.linspace(0,360-s_deg,n_degrees,dtype='int32')    

    if not os.path.exists('Output/direct_gratingSinus/input/'):
       os.mkdir('Output/direct_gratingSinus/input/')  

    for a in range(n_amplitudes):
        amp = amp_start+((a+1)*amplitude_step)
        if not os.path.exists('Output/direct_gratingSinus/input/amp_%f'%amp):
            os.mkdir('Output/direct_gratingSinus/input/amp_%f'%amp)    

        np.random.shuffle(l_degrees)

        for d in l_degrees:                
            # create a bar on a random degree
            spat_speed = speed[0] 
            s_f = sf_steps[3]
            degree = d

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

        
            for i in range(n_img_total): 
                parameters[6] = i*step# x_center position
                img = grating.createGaborMatrix(parameters,image_size,image_size)
                img = ndimage.rotate(img,degree,reshape=False,mode='wrap')
                #print(np.max(img),np.min(img))   
                
                img += 1#np.abs(np.min(img))
                img *= amp
                #print(np.max(img),np.min(img))
                img  = img[int(edge/2):int(edge/2)+s_input,int(edge/2):int(edge/2)+s_input]
                input_imgs.append(img)

    input_imgs = np.asarray(input_imgs)



    ## create the figure with the subplot structure
    fig, ax = plt.subplots()
    
    im1 = ax.imshow(input_imgs[0], cmap=plt.get_cmap('gray'),aspect='auto',vmin=0.0,vmax=0.27)
    ax.axis('off')

    def run(it):
        # update the data 
        im1.set_data(input_imgs[it])
        return im1,input_imgs

    print('Start to create movie')

    ani = animation.FuncAnimation(fig,run,frames=np.arange(0,len(input_imgs)),interval=1200)
    #writer = PillowWriter(fps=20)
    writervideo = animation.FFMpegWriter(fps=2) 
    ani.save('./Output/sinus_input.mp4', dpi=80, writer=writervideo)


if __name__=="__main__":

    createVideo()

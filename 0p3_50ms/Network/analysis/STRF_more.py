import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import leastsq, curve_fit
from tqdm import tqdm, trange
import pickle

def checkBounds(parameters):

    parameters[0] = np.clip(parameters[0],3,44)
    parameters[1] = np.clip(parameters[1],1,8)
    parameters[2] = np.clip(parameters[2],2,44)
    parameters[3] = np.clip(parameters[3],0.1,24)
    parameters[4] = np.clip(parameters[4],-180,360)

    return(parameters)

def get1DGabor(parameters, x_size=48):

    parameters = checkBounds(parameters)

    x_0    = parameters[0]
    K      = parameters[1]
    w      = parameters[2]
    f_opt  = parameters[3]
    phi    = parameters[4]


    x = np.linspace(x_size-1,0,x_size)

    gabor = K*np.exp(-(2*(x-x_0)/w)**2)*np.cos(2*np.pi*f_opt*(x-x_0) + phi)

    return(gabor)


def get1DGaborParams(x, x_0, K, w, f_opt, phi, x_size=48):

    return(K*np.exp(-(2*(x/x_0)/w)**2)*np.cos(2*np.pi*f_opt*(x-x_0) + phi))

def setSubplotDimension(a):
    x = 0
    y = 0
    if ( (a % 1) == 0.0):
        x = a
        y =a
    elif((a % 1) < 0.5):
        x = round(a) +1
        y = round(a)
    else :
        x = round(a)
        y = round(a)
    return (x,y)   


def startAnalyze():
    name = 'E1'
    list_data = np.load('./work/STRF_data_'+name+'.npy')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20   

    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')

    print(np.shape(preff_E1))

    print(np.shape(list_data))  

    max_prefD = preff_E1[0,:,2]
    step_degree = int(360/n_steps)
    max_prefD = max_prefD*step_degree
    ### the moving direction is orthogonal to the stimulus orientation so add 90 degree 
    max_prefD = (max_prefD)%360

    
    #max_prefD = max_prefD[:49]
    max_prefD = max_prefD%180

    overX = np.where(np.abs(max_prefD - 90)>=(90/2))[0]
    overY = np.where(np.abs(max_prefD - 90)<(90/2))[0]

  
    n_cells, steps, x_dim, y_dim = np.shape(list_data)
    strf_list = np.zeros((n_cells, steps, x_dim))
    for i in range(n_cells):
        list_data[i] = list_data[i]-np.mean(list_data[i]) # seat mean to zero
        list_data[i] /=np.max(list_data[i])
        if i in overX:
            strf_list[i] = np.sum(list_data[i],axis=2)
        if i in overY:
            strf_list[i] = np.sum(list_data[i],axis=1)

    np.save('./work/STRF_list', strf_list)

#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyze()


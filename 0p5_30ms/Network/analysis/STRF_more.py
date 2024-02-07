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

    max_args = (np.where(list_data[0] == np.max(list_data[0])))
    max_y = max_args[2]
    max_t = max_args[0]
    #print(np.where(list_data[0] == np.min(list_data[0])))

    for x in overX:
        data_c = list_data[x]
        
        plt.figure()
        plt.imshow(np.mean(data_c,axis=2), cmap=plt.get_cmap('RdBu',9), aspect='auto')
        plt.savefig('./Output/STRF/cell_oY_%i_STRF'%(x))
        plt.close()

    for y in overY:
        data_c = list_data[y]
        
        plt.figure()
        plt.imshow(np.mean(data_c,axis=1), cmap=plt.get_cmap('RdBu',9), aspect='auto')
        plt.savefig('./Output/STRF/cell_oX_%i_STRF'%(y))
        plt.close()

    return -1

    for i in range(49):
        data_c = list_data[i]
    
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.mean(data_c,axis=2), cmap=plt.get_cmap('RdBu',9), aspect='auto')
        plt.subplot(122)
        plt.imshow(np.mean(data_c,axis=1), cmap=plt.get_cmap('RdBu',9), aspect='auto')
        plt.savefig('./Output/STRF/cell%i_STRF'%(i))
        plt.close()

    return -1

    curve = np.sum(list_data[3],axis=-1)

    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(1,2,1)
    plt.imshow(curve,cmap='gray', aspect='auto', interpolation='none')
    fig.add_subplot(1,2,2)
    plt.plot(curve[:,max_y],'--o')
    plt.savefig('./Output/STRF/test.png')

    print(np.shape(curve))
    plt.figure()
    plt.plot(np.linspace(0,x_dim-1,x_dim),np.squeeze(curve[max_t,:]),'--o')
    plt.savefig('./Output/STRF/test_spatial')
    """
    for i in range(x_dim):
        plt.figure()
        plt.plot(curve[:,i],'--o')
        plt.ylim(np.min(curve),np.max(curve))
        plt.savefig('./Output/STRF/test_%i'%i)
        plt.close()
    """

    fft = np.fft.fftshift(np.fft.fft2(curve)).real
    fft = fft[0:150,0:24]

    plt.figure()
    plt.imshow(fft, cmap='gray',interpolation='none', aspect='auto')
    plt.colorbar()
    plt.savefig('./Output/STRF/fft2')


    fittParameters = [x_dim/2, 1, 4.8, 2/x_dim, np.pi*180/2]
    fittParameters[0] = np.random.randint(3,x_dim-3)
    fittParameters[1] = np.random.randint(1,8)
    fittParameters[2] = np.random.uniform(3,x_dim)
    fittParameters[3] = np.random.uniform(0.1,4)
    fittParameters[4] = np.random.uniform(np.pi/10,np.pi*2)


    testParams = [x_dim/2, 1, 44, 1/24, 45/np.pi]

    fit_gabor = get1DGabor(testParams)
    plt.figure()
    plt.plot(fit_gabor,'-o')
    plt.savefig('./Output/test_gabor')


    n_trys = 100

    final_params_cells = []
    #pbar = tqdm( total = ((steps-150)*n_trys*n_cells) )
    for c in range(n_cells):
        curve = np.sum(list_data[c],axis=-1)

        print('Cell: ',c)
        gabor_init = get1DGabor(fittParameters)
        final_params = []
        pbar = tqdm( total = ((steps-150)*n_trys) ,ncols=80, mininterval=1)
        for t in range(150,steps):
            gabor_init = get1DGabor(fittParameters)
            best_params = fittParameters
            data = np.squeeze(curve[t,:])
            init_error = np.mean( np.sqrt((gabor_init - data)**2))
            act_error = init_error
            for r in range(n_trys):
                
                fittParameters[0] = np.random.randint(3,x_dim-3)
                fittParameters[1] = np.random.randint(1,8)
                fittParameters[2] = np.random.uniform(3,x_dim)
                fittParameters[3] = np.random.uniform(0.1,4)
                fittParameters[4] = np.random.uniform(0.0,np.pi*2)

                errfunc = lambda fittParameters,data: (get1DGabor(fittParameters) - data)**2 
                params, success = leastsq(errfunc,fittParameters, args=(data),maxfev=10000)
                new_error = np.mean(np.sqrt((get1DGabor(params) - data)**2))

                if new_error < act_error:
                    best_params = params
                    act_error = new_error
                pbar.update(1)
            final_params.append(best_params)
        final_params_cells.append(final_params)

        #fit_gabor = get1DGabor(best_params)
        #fig = plt.figure(figsize=(12,6))
        #fig.add_subplot(1,2,1)
        #plt.plot(np.linspace(0,x_dim-1,x_dim),fit_gabor,'--o')
        #plt.ylim(-7,7)
        #fig.add_subplot(1,2,2)
        #plt.plot(np.linspace(0,x_dim-1,x_dim),data,'--o')
        #plt.ylim(-7,7)
        #plt.savefig('./Output/STRF/test_spatial_fitt_%i'%(t))            
        #plt.close()

    np.save('./work/1DGabor_Fitt',final_params_cells)


#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyze()


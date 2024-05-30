import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import least_squares
from tqdm import tqdm, trange
import multiprocessing as mp
import sys

def checkBounds(parameters, x_dim):

    parameters[0] = np.clip(parameters[0],5,40)
    parameters[1] = np.clip(parameters[1],0,15)
    parameters[2] = np.clip(parameters[2],5,35)
    parameters[3] = np.clip(parameters[3],0.1,3)
    parameters[4] = np.clip(parameters[4],-100,250)#,10, 350)

    return(parameters)

def get1DGabor(parameters, x_size):

    parameters = checkBounds(parameters,x_size)

    x_0    = parameters[0]
    K      = parameters[1]
    w      = parameters[2]
    f_opt  = parameters[3]/x_size
    phi    = (parameters[4]/180.0)*np.pi


    x = np.linspace(x_size-1,0,x_size)

    gabor = K*np.exp(-(2*(x-x_0)/w)**2)*np.cos(2*np.pi*f_opt*(x-x_0) + phi)

    return(gabor)


def get1DGaborParams(x, x_0, K, w, f_opt, phi, x_size):

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


def fitt1D_Gabor(curve, send_end):
    
    n_c = np.shape(curve)[0]
    steps = np.shape(curve)[1]
    x_dim = np.shape(curve)[2]
    final_params = np.zeros((n_c,steps,5))

    n_trys = 500#2000

    fittParameters = [x_dim/2, 1, 4.8, 2/x_dim, np.pi*180/2]
    fittParameters[0] = np.random.randint(10,40)
    fittParameters[1] = np.random.randint(0,15)
    fittParameters[2] = np.random.uniform(5,35)
    fittParameters[3] = np.random.uniform(0.1,3)
    fittParameters[4] = np.random.uniform(-100,250)

    gabor_init = get1DGabor(fittParameters,x_dim)
    #pbar = tqdm( total = (steps*n_trys*n_c) ,ncols=80, mininterval=2)
    for c in range(n_c):
        gabor_init = get1DGabor(fittParameters, x_dim)
        best_params = fittParameters
        for t in range(steps):
            data = np.squeeze(curve[c,t,:])
            init_error = np.sqrt(np.mean((gabor_init - data)**2))
            act_error = init_error
            for r in range(n_trys):
                    
                fittParameters[0] = np.random.randint(10,40)
                fittParameters[1] = np.random.randint(0,15)
                fittParameters[2] = np.random.uniform(5,35)
                fittParameters[3] = np.random.uniform(0.1,3)
                fittParameters[4] = np.random.uniform(-100,250)

                #errfunc = lambda fittParameters,data: (get1DGabor(fittParameters, x_dim) - data)**2
                def errfunc(fittParameters):
                    return (get1DGabor(fittParameters, x_dim) - data)**2

                results = least_squares(errfunc,fittParameters, method = 'trf')
                params = results['x']
                new_error = np.sqrt(np.mean((get1DGabor(params,x_dim) - data)**2))

                if new_error < act_error:
                    best_params = params
                    act_error = new_error
                #pbar.update(1)
            final_params[c,t] = best_params

    final_params = final_params

    send_end.send(final_params)

def startAnalyze():
    name = 'E1'
    strf_list = np.load('./work/STRF_list.npy')

    n_cells, n_steps ,x_dim = np.shape(strf_list)

    testParams = [x_dim/2, 1, 44, 1/24, 45/np.pi]

    fit_gabor = get1DGabor(testParams, x_dim)
    plt.figure()
    plt.plot(fit_gabor,'-o')
    plt.savefig('./Output/test_gabor')


    number_of_worker_threads = int(n_cells/2)
    number_of_neurons_per_worker_run = 2
    threads = [None]*number_of_worker_threads
    nNeuronsLeft = n_cells
    currNeuron = 0
    endReached = False


    final_params_cells = np.zeros((n_cells,75,5))

    strf_list = strf_list[:,150:]
    strf_list = strf_list[:,0::2]


    while(nNeuronsLeft > 0):
        start = currNeuron    
        pipe_list = []
        
        print('creating worker threads')


        for thr in range(0, number_of_worker_threads):
            recv_end, send_end = mp.Pipe(False) #create a pipeline whit the two ends recv_end, send_end
            tmp = currNeuron + number_of_neurons_per_worker_run

            if (tmp >= n_cells):	#last run here! create only one more worker whit the last neurons
                endReached = True
                tmp = n_cells
                last_run_number_of_neurons = tmp - currNeuron

            threads[thr] = mp.Process(target=fitt1D_Gabor, args=(strf_list[currNeuron:tmp], send_end))#create the thread and give the worker the send_end of the pipe
            pipe_list.append(recv_end)#add the recv_end to the empty list


            threads[thr].start()#... and start the worker
            currNeuron = tmp
            if endReached:
                number_of_worker_threads = thr
                break

        print('worker threads running..\n')

        for thr in range(0, number_of_worker_threads):
            threads[thr].join()	#join for all worker, all have written the result in the pipeline
            sys.stdout.write('.')
            sys.stdout.flush()


        print('\n')
        print('all worker threads are finished, acquiring results..')

        result_list = [result.recv() for result in pipe_list]#get the results from the pipeline
        print('results recived, saving them..')

        for i in range(0, len(result_list)):#fill the results in the result arrays
            if endReached and (i == (len(result_list)-1)):
                end = start+last_run_number_of_neurons
            else:
                end = start+number_of_neurons_per_worker_run
            final_params_cells[ start:end, : ] = result_list[i]
            start = end
		
        print('all results saved..')
		
        nNeuronsLeft = n_cells - currNeuron
        print('#--- ' + str(round(((float(currNeuron)/n_cells)*100), 2)) + '% fitted ---#')
        if not endReached:
            print('starting again..')



    np.save('./work/1DGabor_Fitt',final_params_cells)

def plotFit():
    
    name = 'E1'
    list_data = np.load('./work/STRF_data_'+name+'.npy')
    strf_list = np.load('./work/STRF_list.npy')

    list_data = strf_list[:,150:]
    list_data = list_data[:,0::2]

    print(np.shape(list_data))
    n_cells, n_steps, x_dim = np.shape(list_data)

    params = np.load('./work/1DGabor_Fitt.npy')
    print(np.shape(params))
    
    n_cells, n_steps, n_params = np.shape(params)
    """
    for i in range(n_cells):
        list_data[i] = list_data[i]-np.mean(list_data[i]) # seat mean to zero
        list_data[i] /=np.max(list_data[i])
    """
    fig = plt.figure(figsize=(15,15))
    for i in range(n_cells):
        fig.add_subplot(7,7,i+1)
        plt.plot(params[i,:,4])
    plt.savefig('./Output/STRF/phi_allCells.png')            
    plt.close()

    fitt_errors = np.zeros((n_cells,n_steps))
    for i in tqdm(range(n_cells)):
        if not os.path.exists('Output/STRF/gabor_cell'+str(i)+'/'):
            os.mkdir('Output/STRF/gabor_cell'+str(i)+'/')
        curve = list_data[i] #np.sum(list_data[i],axis=-1)


        fig = plt.figure(figsize=(20,5))
        fig.add_subplot(1,5,1)
        plt.plot(params[i,:,0],'--o')
        plt.ylabel('x_0')
        plt.xlabel('time steps')
        fig.add_subplot(1,5,2)
        plt.plot(params[i,:,1],'--o')
        plt.ylabel('K')
        plt.xlabel('time steps')
        fig.add_subplot(1,5,3)
        plt.plot(params[i,:,2],'--o')
        plt.ylabel('w')
        plt.xlabel('time steps')
        fig.add_subplot(1,5,4)
        plt.plot(params[i,:,3],'--o')
        plt.ylabel('f_opt')
        plt.xlabel('time steps')
        fig.add_subplot(1,5,5)
        plt.plot(params[i,:,4],'--o')
        plt.ylabel('phi')
        plt.xlabel('time steps')
        plt.savefig('./Output/STRF/gabor_cell'+str(i)+'/params')
        plt.close()


        for t in range(n_steps):
            data = np.squeeze(curve[t])
            fit_gabor = get1DGabor(params[i,t],x_dim)
            fitt_errors[i,t] = np.sqrt(np.mean((fit_gabor - data)**2))
            fig = plt.figure(figsize=(12,6))
            fig.add_subplot(1,2,1)
            plt.plot(np.linspace(0,x_dim-1,x_dim),fit_gabor,'--o')
            plt.ylim(-20,20)
            fig.add_subplot(1,2,2)
            plt.plot(np.linspace(0,x_dim-1,x_dim),data,'--o')
            plt.ylim(-20,20)
            plt.savefig('./Output/STRF/gabor_cell'+str(i)+'/test_spatial_fitt_%i'%(t))            
            plt.close()

        plt.figure()
        plt.plot(fitt_errors[i],'-o')
        plt.savefig('./Output/STRF/gabor_cell'+str(i)+'/errors')
        plt.close()
        

    plt.figure()
    plt.plot(np.mean(fitt_errors,axis=1),'o')
    plt.savefig('./Output/STRF/gabor_meanFitt_Error') 
    plt.close()    

    ## calculate the difference between different t
    diff_phi = np.zeros((n_cells,n_steps-1,n_params))
    for c in range(n_cells):
        diff_phi[c] = [np.abs( params[c,t,:] - params[c,t-1,:]) for t in range(1,n_steps) ]

        plt.figure(figsize=(15,5))
        plt.subplot(151)
        plt.plot(diff_phi[c,:,0],'-o')
        plt.ylabel(r'$\Delta$ $x_0$')
        plt.xlabel('time step')
        plt.subplot(152)
        plt.plot(diff_phi[c,:,1],'-o')
        plt.ylabel(r'$\Delta$ $K$')
        plt.xlabel('time step')
        plt.subplot(153)
        plt.plot(diff_phi[c,:,2],'-o')
        plt.ylabel(r'$\Delta$ $w$')
        plt.xlabel('time step')
        plt.subplot(154)
        plt.plot(diff_phi[c,:,3],'-o')
        plt.ylabel(r'$\Delta$ $f_{opt}$')
        plt.xlabel('time step')
        plt.subplot(155)
        plt.plot(diff_phi[c,:,4],'-o')
        plt.ylabel(r'$\Delta$ $\phi$')
        plt.xlabel('time step')
        plt.savefig('./Output/STRF/gabor_cell'+str(c)+'/distance')
        plt.close()
        ## set phi = 0 where delta_phi >200, because this is a bug through the fitting
        diff_phi[c,diff_phi[c,:,4]>180,4] = 0 

    plt.figure()
    plt.plot(np.max(diff_phi[:,:,4],axis=1),'o')
    plt.ylabel(r'max $\Delta$ $\phi$')
    plt.xlabel('cell index')
    plt.savefig('./Output/STRF/maxDeltaPhi') 
    plt.close

    np.save('./work/1DGabor_Fitt_Delta',diff_phi)

#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyze()
    plotFit()


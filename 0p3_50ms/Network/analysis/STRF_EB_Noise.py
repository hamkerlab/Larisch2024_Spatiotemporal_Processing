import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

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


def fastSTA(sta_Input,sta_frExc):
    sta_input = sta_input[:,:,:,0] - sta_input[:,:,:,1] 
    n_cells = np.shape(sta_frExc)[0]
    n_stim,h,w = np.shape(sta_input)

    sta_cells = np.zeros((n_cells,h,w))

    for i in range(n_cells):
        a_i =  np.where(sta_frExc[i]>0)
        sta_cells = np.sum(sta_input[a_i])*sta_frExc[a_i]

def calcSTRF(sta_Input,spk_pop, n_cells, time_back, name):
    # calculate the spatio-temporal RF:
    # per cell, have a [X,Y,T] matrix
    # sum over T -> STA  !
    # sum over Y -> STRF !
    # see Ohazawa et al. (1996)

    print('Calculate the STRF for '+name)
    n_stim ,s_X,s_Y = np.shape(sta_Input)
    n_stim, duration = (n_stim,50)
    total_t = n_stim*duration
  

    fr_STA = np.zeros(n_stim)
    max_tS = duration*n_stim # total simulation time
    
    spkC = np.zeros((n_cells,n_stim))
    exc_spks = np.zeros((total_t,n_cells),dtype='int32') # matrix to say if a neuron spiked (1) or not (0) at a specific time point (t)
    inp_bin = np.zeros((n_stim,duration,s_X,s_Y))
    temporal_offset = 0
    for i in range(n_stim):
        evtL = spk_pop[i]   
        for c in range(n_cells):
            if (len(evtL[c]) != 0 ):
                spkC[c,i] = len(evtL[c]) # number of spikes
                for e_t in evtL[c]:
                    exc_spks[e_t,c] = 1

        inp = sta_Input[i]
        inp_bin[i] = np.repeat(inp[np.newaxis,:,:],duration,axis=0)
        temporal_offset += duration

    print('Mean activity = %f'%(np.mean(spkC)))
    print('Max spike count = %f'%(np.max(spkC)))

    
    inp_bin = np.reshape(inp_bin,(n_stim*duration,s_X,s_Y))
    list_input = []
    list_inputON=[]
    list_inputOFF=[]

    for i in range(n_cells):
        print('on cell %i'%(i))
        event_w = np.zeros((time_back,s_X,s_Y)) # summed input for the STRF
        event_ON = np.zeros((time_back,s_X,s_Y)) # summed input for the STRF
        event_OFF = np.zeros((time_back,s_X,s_Y)) # summed input for the STRF
        
        
        spike_times = np.where(exc_spks[time_back:,i] == 1)[0]+time_back
        for s_t in spike_times: # for each time point of spike
            i_bin = inp_bin[s_t-time_back:s_t]
            event_w += i_bin # save for STRF

            # divide input into white and dark bars
            on_bin =  (i_bin>0.0)*0.5
            off_bin = (i_bin<0.0)*-0.5

            event_ON += on_bin
            event_OFF += off_bin

        event_w /= len(spike_times)
        event_ON/= len(spike_times)
        event_OFF/= len(spike_times)


        list_input.append(event_w)

        list_inputON.append(event_ON)
        list_inputOFF.append(event_OFF)

    np.save('./work/STRF_data_'+name,list_input)
    np.save('./work/STRF_data_ON_'+name,list_inputON)
    np.save('./work/STRF_data_OFF_'+name,list_inputOFF)

    print('Finish with creating data')
    return()


def plot(list_STA, list_STRF,list_STRF_ON,list_STRF_OFF, name):
    fig = plt.figure(figsize=(8,8))
    print(np.shape(list_STA))
    x,y = setSubplotDimension(np.sqrt(np.shape(list_STA)[0]))
    for i in range(np.shape(list_STA)[0]):
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(list_STA[i],cmap=mp.cm.Greys_r,aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STRF/STRF_'+name+'/STAEX_NOISE_Test.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    fig = plt.figure(figsize=(8,8))
    maxV = np.max(np.max(list_STRF,axis=1),axis=1)
    minV = np.min(np.min(list_STRF,axis=1),axis=1)
    x,y = setSubplotDimension(np.sqrt(np.shape(list_STRF)[0]))
    for i in range(np.shape(list_STRF)[0]):
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(list_STRF[i],cmap=plt.get_cmap('gray',7),aspect='auto',interpolation='none',vmin=minV[i],vmax=maxV[i])
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STRF/STRF_'+name+'/STRF_NOISE_Test.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)
    

    fig = plt.figure(figsize=(8,8))
    maxV = np.max(np.max(list_STRF_ON,axis=1),axis=1)
    minV = np.min(np.min(list_STRF_ON,axis=1),axis=1)
    x,y = setSubplotDimension(np.sqrt(np.shape(list_STRF_ON)[0]))
    for i in range(np.shape(list_STRF_ON)[0]):
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(list_STRF_ON[i],cmap=plt.get_cmap('gray',7),aspect='auto',interpolation='none',vmin=minV[i],vmax=maxV[i])
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STRF/STRF_'+name+'/STRF_NOISE_ON_Test.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    fig = plt.figure(figsize=(8,8))
    maxV = np.max(np.max(list_STRF_OFF,axis=1),axis=1)
    minV = np.min(np.min(list_STRF_OFF,axis=1),axis=1)
    x,y = setSubplotDimension(np.sqrt(np.shape(list_STRF_OFF)[0]))
    for i in range(np.shape(list_STRF_OFF)[0]):
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(list_STRF_OFF[i],cmap=plt.get_cmap('gray',7),aspect='auto',interpolation='none',vmin=minV[i],vmax=maxV[i])
        #plt.axis('equal')
    #plt.axis('equal')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STRF/STRF_'+name+'/STRF_NOISE_OFF_Test.jpg',bbox_inches='tight', pad_inches = 0.1,dpi=300)

def plotData(name):

    if not os.path.exists('Output/STRF/STRF_'+name+'/'):
        os.mkdir('Output/STRF/STRF_'+name+'/')

    list_input = np.load('./work/STRF_data_'+name+'.npy')
    list_input_ON = np.load('./work/STRF_data_ON_'+name+'.npy')
    list_input_OFF = np.load('./work/STRF_data_OFF_'+name+'.npy')
   
    
    t_n = 0#114

    #list_input = list_input[:,:,:,:,0] - list_input[:,:,:,:,1]
    n_cells,n_time =np.shape(list_input)[0:2]
    list_STA = []
    list_STRF = []
    list_STRF_ON = []
    list_STRF_OFF = []    
    for i in range(n_cells):
        list_STA.append(np.sum(list_input[i],axis=0))
        list_STRF.append(np.sum(list_input[i],axis=2))
        list_STRF_ON.append(np.sum(list_input_ON[i],axis=2))
        list_STRF_OFF.append(np.sum(list_input_OFF[i],axis=2))

    list_STRF = np.asarray(list_STRF)
    print(np.shape(list_STRF))

    l_ON_STRF = np.zeros((n_cells,n_time))
    l_OFF_STRF = np.zeros((n_cells,n_time))

    for i in range(n_cells):
        l_argmax = np.asarray(np.where(list_STRF[i,:,:] == np.max(list_STRF[i,:,:])))[1] # ON part
        l_argmin = np.asarray(np.where(list_STRF[i,:,:] ==  np.min(list_STRF[i,:,:])))[1]# OFF part
        l_ON_STRF[i] = list_STRF[i,:,l_argmax[0]]
        l_OFF_STRF[i] = list_STRF[i,:,l_argmin[0]]

    l_list_ON = np.mean(list_STRF_ON,axis=2)
    l_list_OFF = np.mean(list_STRF_OFF,axis=2)

    for i in range(1):
        plotLinesON = l_ON_STRF[i]
        plotLinesON = plotLinesON[::-1]
        plotLinesOFF= l_OFF_STRF[i]
        plotLinesOFF= plotLinesOFF[::-1]

        plt.figure()
        plt.plot(plotLinesON/np.max(np.abs(plotLinesON)),'--',color='steelblue', label='ON')
        #plt.hlines(0,xmin=0,xmax=n_time)
        plt.plot(plotLinesOFF/np.max(np.abs(plotLinesOFF)),'--',color='tomato', label='OFF')
        plt.xlabel('time [ms]')
        plt.ylabel('normalized amplitude')
        #plt.xticks(np.linspace(0,len(l_ON_STRF[i]),7),np.linspace(300,0,7))
        plt.hlines(0,0,150,'gray','--')
        plt.legend()
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Noise_TimeS_'+str(i))
        plt.close()

        plt.figure()
        plt.subplot(211)
        plt.plot(l_list_ON[i]/np.max(np.abs(l_list_ON[i])),'--',color='steelblue', label='ON')
        plt.ylabel('normalized amplitude')
        plt.legend()
        plt.subplot(212)
        plt.plot(l_list_OFF[i]/np.max(np.abs(l_list_OFF[i])),'--',color='tomato', label='OFF')
        plt.xlabel('time [ms]')
        plt.ylabel('normalized amplitude')
        plt.legend()
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Noise_TimeS_list2_'+str(i))
        plt.close()


    mean_lON = np.mean(l_ON_STRF,axis=0)
    mean_lOFF = np.mean(l_OFF_STRF,axis=0)

    std_lON = np.std(l_ON_STRF,axis=0,ddof=1)
    std_lOFF = np.std(l_OFF_STRF,axis=0,ddof=1)

    x = np.linspace(0,n_time-1,n_time)

    plt.figure()
    plt.plot(mean_lON,'--',color='steelblue',label='ON input')
    plt.fill_between(x,mean_lON-std_lON, mean_lON+std_lON,color='steelblue',alpha=0.2 )
    plt.plot(mean_lOFF,'--',color='tomato',label='OFF input')
    plt.fill_between(x,mean_lOFF-std_lOFF, mean_lOFF+std_lOFF,color='tomato',alpha=0.2 )
    #plt.hlines(0,xmin=0,xmax=n_time)
    plt.legend()
    plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Noise_TimeS_allCmean')


    print(np.shape(list_input))
    n_cells,duration,h,w = np.shape(list_input)
    print(duration)
    tm_STA = np.zeros((n_cells,duration-1))
    for c in range(n_cells):    
        for t in range(1,duration):
            sta_old = np.reshape(list_input[c,t-1],(h*w) )
            sta_new = np.reshape(list_input[c,t],(h*w))
            tm_STA[c,t-1] = np.dot(sta_old,sta_new)/(np.linalg.norm(sta_old)*np.linalg.norm(sta_new))
        


    plt.figure()
    plt.plot(np.mean(tm_STA,axis=0))
    plt.savefig('./Output/STRF/STRF_'+name+'/STRF_NOISE_TM_Test.jpg')

    plot(list_STA,list_STRF,list_STRF_ON,list_STRF_OFF,name)
    
    strf = list_STRF[0]
    strf = np.mean(strf,axis=1)
    plt.figure()
    plt.plot(strf)
    #plt.hlines(0.5,0.0,len(strf))
    plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Time_cell0.jpg')


def startAnalyze():

    name_l = ['RGC','LGN','E1'] 

    if not os.path.exists('Output/STRF/'):
        os.mkdir('Output/STRF/')

    sta_Input = np.load('./work/STRF_Input.npy')


    time_back = 300 #ms time to look back

    spk_RGC = np.load('./work/STRF_Spk_RGC.npy')
    calcSTRF(sta_Input,spk_RGC,9,time_back,name_l[0]) # STRF for LGN
    plotData(name_l[0])

    spk_LGN = np.load('./work/STRF_Spk_LGN.npy')
    calcSTRF(sta_Input,spk_LGN,9,time_back,name_l[1]) # STRF for LGN
    plotData(name_l[1])
    
    spk_E1 = np.load('./work/STRF_Spk_E1.npy')
    calcSTRF(sta_Input,spk_E1,9,time_back,name_l[2]) # STRF for E1
    plotData(name_l[2])

#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyze()


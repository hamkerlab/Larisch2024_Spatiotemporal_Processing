import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

#import seaborn as sns
#import pandas as pd
import numpy as np
import os
import pickle
from scipy import signal
from tqdm import tqdm

###
# Analyse DS, but use the preff. O from TC to determine preff D
###

def load_Spk_Data():

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)

    ## initialize LGN data
    test_data = np.load('./work/directGrating_Sinus_SpikeCount_LGN_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells = np.shape(test_data)    
    spkC_LGN_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells))

    ## initialize E1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeCount_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells = np.shape(test_data)    
    spkC_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells))

    ## initialize I1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeCount_I1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells = np.shape(test_data)    
    spkC_I1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                spkC_LGN_all[a,sf,tf] = np.load('./work/directGrating_Sinus_SpikeCount_LGN_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                spkC_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeCount_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                spkC_I1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeCount_I1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(spkC_E1_all, spkC_I1_all,spkC_LGN_all)

def load_Time_Data():

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)

    #np.load('./work/directGrating_Sinus_SpikeTimes_E1.npy')
    ## initialize E1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    print(np.shape(test_data))
    n_degree, repeats, n_cells, total_t  = np.shape(test_data)    
    spkT_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells, total_t ))

    ## initialize I1 data
    test_data = np.load('./work/directGrating_Sinus_SpikeTimes_I1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    n_degree, repeats, n_cells,total_t  = np.shape(test_data)    
    spkT_I1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, n_cells, total_t ))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                spkT_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeTimes_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                spkT_I1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_SpikeTimes_I1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(spkT_E1_all, spkT_I1_all)

def load_Current_Data():

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']


    n_amp = 1
    n_spatF = len(lvl_speed)
    n_tempF = len(lvl_spFrq)


    ## initialize current data
    test_data = np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(0,0,0)) # dummy data to get shapes 
    print(np.shape(test_data))
    n_degree, repeats, total_t, n_cells = np.shape(test_data)    
    gEx_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, total_t, n_cells))
    gIn_E1_all = np.zeros((n_amp,n_spatF, n_tempF, n_degree, repeats, total_t, n_cells))


    for a in range(n_amp):
        for sf in range(n_spatF):
            for tf in range(n_tempF):
                gEx_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_gExc_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))
                gIn_E1_all[a,sf,tf] =  np.load('./work/directGrating_Sinus_gInh_E1_amp%i_spatF%i_tempF%i.npy'%(a,sf,tf))    

    return(gEx_E1_all,gIn_E1_all)

def main():
    print('Analyse direction selectivity by preseting a moving Sinus')

    if not os.path.exists('Output/DSI_TC/'):
        os.mkdir('Output/DSI_TC/')


    spkC_E1_all, spkC_I1_all, spkC_LGN_all= load_Spk_Data()


    ### get preff O. 
    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')
    params_E1 = params_E1[:,0,5]
    orienSteps = 8
    preffO_E1 = params_E1*orienSteps

    print('Mean FR LGN = ',np.mean(spkC_LGN_all))
    n_contrast, n_speeds, n_spatF,n_degress,repeats,n_cellsE1 = np.shape(spkC_E1_all)


    #sim_param = np.loadtxt('./work/directGrating_Sinus_parameter.txt')
    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']

    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]


    w_I1 = np.loadtxt('./Input_network/InhibW.txt')
    n_post,n_pre = np.shape(w_I1)
    rf_I1 = np.reshape(w_I1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_I1 = rf_I1[:,:,:,0] - rf_I1[:,:,:,1]


    dsi_kim = np.zeros((n_contrast,n_cellsE1))#DSI after Kim and Freeman (2016)// 1-(R_null/R_pref)

    preff_E1 = np.zeros((n_contrast,n_cellsE1,3)) # save the indices(!) of preffered speed[0], spatF[1], direction[2]

    for i in range(n_contrast):

        if not os.path.exists('Output/DSI_TC/LVL_%i'%(i)):
            os.mkdir('Output/DSI_TC/LVL_%i'%(i))

        step_degree = int(360/n_degress) 

        ## get through each cell and look, where the cells has spiked most
        for c in range(n_cellsE1):
            spkC_E1 = spkC_E1_all[i,:,:,:,:,c]
            spkC_E1 = np.mean(spkC_E1,axis=3)
            ## get the direction wich is close as possible to pref O + 90°
            
            pref_O = preffO_E1[c]
            ## each pref_0 can be combined with two pref D -> choose the one with higher activity!
            pref_D1 = pref_O
            pref_D2 = (pref_O+180)%360            

            pref_D1_arg = int(pref_D1//step_degree)
            spkC_E1_pref1 = spkC_E1[:,:,pref_D1_arg]

            pref_D2_arg = int(pref_D2//step_degree)
            spkC_E1_pref2 = spkC_E1[:,:,pref_D2_arg]

            pref_speed1, pref_spatF1 = np.where(spkC_E1_pref1 == np.max(spkC_E1_pref1))
            pref_speed2, pref_spatF2 = np.where(spkC_E1_pref2 == np.max(spkC_E1_pref2))

            spk_cell1 = spkC_E1[int(pref_speed1[0]),int(pref_spatF1[0])]
            spk_cell2 = spkC_E1[int(pref_speed2[0]),int(pref_spatF2[0])]
            if np.max(spk_cell1)>np.max(spk_cell2):
                preff_E1[i,c,:] = pref_speed1[0], pref_spatF1[0], pref_D1_arg #pref_degree[0]
            else:
                preff_E1[i,c,:] = pref_speed2[0], pref_spatF2[0], pref_D2_arg


            ##calculate the DSI
            spk_cell = spkC_E1[int(preff_E1[i,c,0]),int(preff_E1[i,c,1])]
            #spk_cell = spk_cell[0]
            preff_arg = int(preff_E1[i,c,2])
            preff_D = preff_arg*step_degree
            null_D = (preff_D+180)%360
            null_arg = int(null_D/step_degree)
            if spk_cell[preff_arg] < spk_cell[null_arg]:
                preff_t = np.copy(null_arg)
                null_arg = np.copy(preff_arg)
                preff_arg = preff_t

            dsi_kim[i,c] = 1 - (spk_cell[null_arg]/spk_cell[preff_arg])            

        plt.figure()
        plt.hist(dsi_kim[i])
        plt.ylabel('# of cells')
        plt.xlabel('DSI Kim')
        plt.savefig('Output/DSI_TC/LVL_%i/Hist_dsi_kim.png'%(i))
        plt.close()                

    np.save('./work/dsi_kim_Cells_TC',dsi_kim,allow_pickle=False)
    np.save('./work/dsi_preff_E1_TC',preff_E1,allow_pickle=False)
#------------------------------------------------------------------------------
def calcCrossCorr(x,y,t_win):
    # calculate the crosscorrelation for a specific time window
    crossCorr = np.zeros(t_win*2+1)

    c = 0
    for i in range(t_win,0,-1): # first 100 shifts
        y_calc = y[i:]
        x_calc = x[:len(y_calc)]
        crossCorr[c] = np.corrcoef(x_calc,y_calc)[0,1]
        c+=1
    ## identic ##
    crossCorr[c] = np.corrcoef(x,y)[0,1]
    c+=1

    for i in range(1,t_win+1):
        x_calc = x[i:] 
        y_calc = y[:len(x_calc)]
        crossCorr[c] = np.corrcoef(x_calc,y_calc)[0,1]
        c+=1

    return(crossCorr)


def reject_outliers(data, m=3.):
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def moreDSI():

    if not os.path.exists('Output/direct_gratingSinus/maxDSI'):
        os.mkdir('Output/direct_gratingSinus/maxDSI')

    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    


    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    n_post,n_pre = np.shape(w_E1)
    rf_E1 = np.reshape(w_E1, (n_post, int(np.sqrt(n_pre/2)), int(np.sqrt(n_pre/2)),2) )
    rf_E1 = rf_E1[:,:,:,0] - rf_E1[:,:,:,1]

    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')
    print(np.shape(preff_E1))

    dsi_kim = np.load('./work/dsi_kim_Cells_TC.npy')
    print(np.shape(dsi_kim))


    mse_list = np.load('./work/rmseFFT2D.npy')
    ampl_list = np.load('./work/ampDiffFFT2D.npy')
    print(np.shape(mse_list), np.shape(ampl_list))

    mse_min = np.where(mse_list<0.035)[0]
    dsi_min = dsi_kim[0, mse_min]
    mse_min = mse_list[mse_min]
    mse_max = np.where(mse_list>0.17)[0]
    dsi_max = dsi_kim[0, mse_max]
    mse_max = mse_list[mse_max]

    tfd_min = np.where(ampl_list<0.01)[0]
    dsi_min = dsi_kim[0, tfd_min]
    tfd_min = ampl_list[tfd_min]
    tfd_max = np.where(ampl_list>0.3)[0]
    print(tfd_max)
    dsi_max = dsi_kim[0, tfd_max]
    tfd_max = ampl_list[tfd_max]

    n_cells = n_post
    ## save how many cells have a specific speed or spatF level (depending on max FR)
    n_speed = len(lvl_speed)
    n_spatF = len(lvl_spFrq)

    img_speed_spat = np.zeros((n_speed,n_spatF))
    for c in range(n_cells):
        pref_speed = int(preff_E1[0,c,0])
        pref_spatF = int(preff_E1[0,c,1])
        img_speed_spat[pref_speed,pref_spatF] +=1


    plt.figure()
    plt.imshow(img_speed_spat,cmap='viridis')
    plt.ylabel('speed [Hz]')
    plt.xlabel('sp.Freq. [Cycl/Image]')
    cbar = plt.colorbar()
    cbar.set_label('# of cells')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1,len(lvl_spFrq)), lvl_spFrq)
    plt.yticks(np.linspace(0,len(lvl_speed)-1,len(lvl_speed)), lvl_speed)
    plt.savefig('./Output/DSI_TC/sF_to_speed_imshow')
    plt.close()

    ## load the STRFs
    strf_E1 = np.load('./work/STRF_list.npy')
    print(np.shape(strf_E1))
    dsi_arg = np.argsort(dsi_kim[0][0:25])
    plt.figure(figsize=(12,12))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.title(np.round(dsi_kim[0][dsi_arg[i]],3))
        plt.imshow(strf_E1[dsi_arg[i]],cmap=plt.get_cmap('RdBu',7), interpolation='none', aspect='auto', vmin=-np.max(np.abs(strf_E1[dsi_arg[i]])), vmax=np.max(np.abs(strf_E1[dsi_arg[i]])))
        plt.axis('off')
    plt.savefig('./Output/DSI_TC/STRF')
    plt.close()



    max_prefD = preff_E1[0,:,2]
    step_degree = int(360/n_steps)
    max_prefD = max_prefD*step_degree
    ### the moving direction is orthogonal to the stimulus orientation so add 90 degree 
    max_prefD = (max_prefD+90)%360

    bin_size=10
    bins = np.arange(0,360,bin_size)
    a,b = np.histogram(max_prefD, bins=np.arange(0,360+bin_size,bin_size))
    print(a,b)
    plt.figure()
    ax = plt.subplot(121,projection='polar')
    bars = ax.bar(bins/180*np.pi, a, width=0.2, bottom=0.1)
    ax = plt.subplot(122)
    plt.hist(max_prefD,bins=10)
    plt.savefig('./Output/DSI_TC/hist_PreffD.png')
    plt.close()

    ### compare the pref. orientation with the pref. direction
    params_E1 = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')
    params_E1 = params_E1[:,0,5]
    orienSteps = 8
    preffO_E1 = params_E1*orienSteps

    plt.figure()
    plt.scatter(preffO_E1%180,max_prefD%180)
    plt.xlabel('preffO')
    plt.ylabel('preffD')
    plt.savefig('./Output/DSI_TC/preffO_preffD')
    plt.close()

    print(np.max(max_prefD),np.min(max_prefD))
    print(np.max(preffO_E1),np.min(preffO_E1))

    diff = np.abs((max_prefD%180) - (preffO_E1%180))%90

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(diff, 'o')
    plt.xlabel('cell index')
    plt.ylabel('prefD - preffO')
    plt.subplot(2,1,2)
    plt.hist(diff,7)
    plt.xlabel('prefD - preffO')
    plt.ylabel('# cells')
    plt.savefig('./Output/DSI_TC/preffO_minus_preffD')
    plt.close()

    if not os.path.exists('Output/DSI_TC/E1'):
        os.mkdir('Output/DSI_TC/E1')

    # make the direct plots for preff DSI 

    # print spikes per cell and degree for max_DSI
    alpha = (step_degree*np.arange(0,n_steps)) # list of degrees 
    alpha = np.roll(alpha,int(90/step_degree))   # roll so direction angle is correct
    alpha = np.append(alpha,alpha[0]) # add the first entry at the end for better plotting
    alpha = alpha/180. * np.pi #change in radian
                        
    #spkC_E1_all = np.load('./work/directGrating_Sinus_SpikeCount_E1.npy')
    spkC_E1_all, _, _= load_Spk_Data()
    
    gEx_E1_all, gIn_E1_all = load_Current_Data()
    #gEx_E1_all = np.load('./work/directGrating_Sinus_gExc_E1.npy')
    #gIn_E1_all = np.load('./work/directGrating_Sinus_gInh_E1.npy')



    gEx_pref_E1 = []
    gEx_null_E1 = []
    gIn_pref_E1 = []
    gIn_null_E1 = []

    

    plot_t = 200
    for c in range(n_cells):
        spk_cell = spkC_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,c]
        spk_cell = np.mean(spk_cell,axis=1)
        spk_cell = np.append(spk_cell,spk_cell[0])


        preff_D = int(preff_E1[0,c,2])*step_degree
        null_D = (preff_D+180)%360
        null_arg = int(null_D/step_degree)

        gEx_E1_all_cell = gEx_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,:,c ]
        gEx_E1_all_cell = np.mean(gEx_E1_all_cell,axis=1)
        preff_gEx_resh= gEx_E1_all_cell[int(preff_E1[0,c,2])]
        total_t = len(preff_gEx_resh)
        gEx_pref_E1.append(preff_gEx_resh)
        preff_gEx_resh= np.reshape(preff_gEx_resh,(int(total_t//plot_t),plot_t))
        #print(np.shape(preff_vm_resh))

        null_gEx_resh = gEx_E1_all_cell[null_arg]
        gEx_null_E1.append(null_gEx_resh)
        null_gEx_resh = np.reshape(null_gEx_resh,(int(total_t//plot_t),plot_t))


        gIn_E1_all_cell = gIn_E1_all[0,int(preff_E1[0,c,0]),int(preff_E1[0,c,1]),:,:,:,c ]
        gIn_E1_all_cell = np.mean(gIn_E1_all_cell,axis=1)
        preff_gIn_resh= gIn_E1_all_cell[int(preff_E1[0,c,2])]
        total_t = len(preff_gIn_resh)
        gIn_pref_E1.append(preff_gIn_resh)
        preff_gIn_resh= np.reshape(preff_gIn_resh,(int(total_t//plot_t),plot_t))
        #print(np.shape(preff_vm_resh))

        null_gIn_resh = gIn_E1_all_cell[null_arg]
        gIn_null_E1.append(null_gIn_resh)
        null_gIn_resh = np.reshape(null_gIn_resh,(int(total_t//plot_t),plot_t))

        
        #grid = plt.GridSpec(5,2,wspace=0.4,hspace=0.3)
        #plt.figure(figsize=(14,16))
        #ax = plt.subplot(grid[0,0],projection='polar')
        #ax.plot( alpha,spk_cell,'o-')
        #ax.grid(True)
        #ax.set_xticklabels([r'$\rightarrow$','', r'$\uparrow$', '', r'$\leftarrow$', '', r'$\downarrow$', ''])
        #ax = plt.subplot(grid[0,1])
        #ax.set_title('DSI = %f'%dsi_kim[0,c])
        #ax.imshow(rf_E1[c],cmap='gray',aspect='equal',interpolation='none')
        #ax.axis('off')
        #ax = plt.subplot(grid[1,:])
        #ax.plot(preff_vm_resh[5],color='steelblue',label='pref')#ax.plot(np.mean(preff_vm_resh,axis=0),color='steelblue',label='pref')#
        #ax.plot(null_vm_resh[5],color='tomato',label='null')#ax.plot(np.mean(null_vm_resh,axis=0),color='tomato',label='null')#
        #ax.legend() 
        #ax = plt.subplot(grid[2,:])
        #ax.plot(preff_gEx_resh[5],'--',color='steelblue',label='pref gEx')#ax.plot(np.mean(preff_gEx_resh,axis=0),'--',color='steelblue',label='pref gEx')#
        #ax.plot(null_gEx_resh[5],'--',color='tomato',label='null gEx')#ax.plot(np.mean(null_gEx_resh,axis=0),'--',color='tomato',label='null gEx')#
        #ax.legend() 
        #ax = plt.subplot(grid[3,:])
        #ax.plot(preff_gIn_resh[5],'.-',color='steelblue',label='pref gIn')#ax.plot(np.mean(preff_gIn_resh,axis=0),'.-',color='steelblue',label='pref gIn')#
        #ax.plot(null_gIn_resh[5],'.-',color='tomato',label='null gIn')#ax.plot(np.mean(null_gIn_resh,axis=0),'.-',color='tomato',label='null gIn')#
        #ax.legend() 
        #ax = plt.subplot(grid[4,:])
        #ax.plot(preff_gEx_resh[5] - preff_gIn_resh[5] ,color='steelblue',label='pref gEx - gIn')
        #ax.plot(null_gEx_resh[5] - null_gIn_resh[5] ,color='tomato',label='null gEx - gIn')
        #ax.hlines(0,0,plot_t,color='black')
        #ax.set_ylim(-60,60)
        #ax.legend() 
        #plt.savefig('./Output/DSI_TC/E1/direct_Polar_%i'%(c))
        #plt.close()
        

    gEx_pref_E1 = np.asarray(gEx_pref_E1)
    gEx_null_E1 = np.asarray(gEx_null_E1)
    gIn_pref_E1 = np.asarray(gIn_pref_E1)
    gIn_null_E1 = np.asarray(gIn_null_E1)

    n_cells, n_steps = np.shape(gEx_pref_E1)

    gEx_pref_E1_smal = np.copy(gEx_pref_E1)
    gEx_null_E1_smal = np.copy(gEx_null_E1)
    gIn_pref_E1_smal = np.copy(gIn_pref_E1)
    gIn_null_E1_smal = np.copy(gIn_null_E1)

    gEx_pref_E1_smal = np.reshape(gEx_pref_E1_smal,(n_cells,30,100))
    gEx_null_E1_smal = np.reshape(gEx_null_E1_smal,(n_cells,30,100))
    gIn_pref_E1_smal = np.reshape(gIn_pref_E1_smal,(n_cells,30,100))
    gIn_null_E1_smal = np.reshape(gIn_null_E1_smal,(n_cells,30,100))

    ### calculate the E/I-ratio for preff and null direction
    ratio_pref = np.nanmean(np.mean(gEx_pref_E1_smal,axis=2)/np.mean(gIn_pref_E1_smal,axis=2),axis=1)
    
    ratio_null = np.nanmean(np.mean(gEx_null_E1_smal,axis=2)/np.mean(gIn_null_E1_smal,axis=2),axis=1) #np.mean(gEx_null_E1,axis=1)/np.mean(gIn_null_E1,axis=1)

    low_dsi_arg = np.where(dsi_kim[0] < 0.2)[0]
    mid_low_dsi_arg = np.where((dsi_kim[0] >= 0.2) & (dsi_kim[0] < 0.4))[0]
    mid_dsi_arg = np.where((dsi_kim[0] >= 0.4) & (dsi_kim[0] < 0.6))[0]
    mid_high_dsi_arg = np.where((dsi_kim[0] >= 0.6) & (dsi_kim[0] < 0.8))[0]
    high_dsi_arg = np.where(dsi_kim[0] >= 0.8)[0]


    crossCorr_pref_valid = []
    crossCorr_null_valid = []

    crossCorr_pref_same = []
    crossCorr_null_same = []


    for c in range(n_cells):
        crossCorr = signal.correlate(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),'valid')
        crossCorr_pref_valid.append(crossCorr)

        crossCorr = signal.correlate(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),'valid')
        crossCorr_null_valid.append(crossCorr)


        #crossCorr = signal.correlate(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),'same')
        crossCorr = calcCrossCorr(gEx_pref_E1[c]/np.max(gEx_pref_E1[c]),gIn_pref_E1[c]/np.max(gIn_pref_E1[c]),20 )
        crossCorr_pref_same.append(crossCorr)#/np.max(crossCorr))

        #crossCorr = signal.correlate(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),'same')
        crossCorr = calcCrossCorr(gEx_null_E1[c]/np.max(gEx_null_E1[c]),gIn_null_E1[c]/np.max(gIn_null_E1[c]),20 )
        crossCorr_null_same.append(crossCorr)#/np.max(crossCorr))

    crossCorr_pref_valid = np.asarray(crossCorr_pref_valid)
    crossCorr_null_valid = np.asarray(crossCorr_null_valid)

    crossCorr_pref_same = np.asarray(crossCorr_pref_same)
    crossCorr_null_same = np.asarray(crossCorr_null_same)

    np.save('./work/dircection_corrCorr_Currents_pref_valid',crossCorr_pref_valid )
    np.save('./work/dircection_corrCorr_Currents_null_valid',crossCorr_null_valid )
    np.save('./work/dircection_corrCorr_Currents_pref_same',crossCorr_pref_same )
    np.save('./work/dircection_corrCorr_Currents_null_same',crossCorr_null_same )

    print(np.shape(crossCorr_pref_same), np.shape(crossCorr_null_same))

    
    t_wi = 21#51
    t_total = len(gEx_pref_E1[0])
    lags = signal.correlation_lags(t_wi, t_wi)

    idx_0 = np.where(lags == 0)[0]
    plt.figure()
    plt.hist(crossCorr_pref_same[:,idx_0], histtype='step', label='pref')
    plt.hist(crossCorr_null_same[:,idx_0], histtype='step', label='null')
    plt.legend()
    plt.xlabel('Correlation gEx - gInh')
    plt.ylabel('# Cells')
    plt.savefig('./Output/DSI_TC/hist_Correlation')
    plt.close()

    idx_0 = np.where(lags == 0)[0]

    crossCorr_argmax = np.argmax(crossCorr_pref_same,axis=1)    
    crossCorr_pref_maxT =(lags[crossCorr_argmax])

    crossCorr_argmax = np.argmax(crossCorr_null_same,axis=1)    
    crossCorr_nullmaxT = (lags[crossCorr_argmax])


    plt.figure()
    plt.plot(lags,crossCorr_pref_same[0], label='pref') #[int(t_total/2)-t_wi:int(t_total/2)+t_wi-1]
    plt.plot(lags,crossCorr_null_same[0], label='null') #[int(t_total/2)-t_wi:int(t_total/2)+t_wi-1]
    plt.vlines(0,ymin=0.5,ymax=1.1, linestyles = 'dashed', color='black')
    plt.legend()
    plt.savefig('./Output/DSI_TC/crossCorr')
    plt.close()

    plt.figure()
    plt.plot(np.mean(crossCorr_pref_same,axis=0), label='pref')#[1400:1600]
    plt.plot(np.mean(crossCorr_null_same,axis=0), label='null')#[1400:1600]
    plt.legend()
    plt.savefig('./Output/DSI_TC/crossCorr_MeanALLC')
    plt.close()


    low_dsi_arg = np.where(dsi_kim[0] < 0.2)[0]
    mid_low_dsi_arg = np.where((dsi_kim[0] >= 0.2) & (dsi_kim[0] < 0.4))[0]
    mid_dsi_arg = np.where((dsi_kim[0] >= 0.4) & (dsi_kim[0] < 0.6))[0]
    mid_high_dsi_arg = np.where((dsi_kim[0] >= 0.6) & (dsi_kim[0] < 0.8))[0]
    high_dsi_arg = np.where(dsi_kim[0] >= 0.8)[0]

    print(len(low_dsi_arg), len(mid_dsi_arg),len(high_dsi_arg))

 
    
    crossCorr_pref_low = crossCorr_pref_same[low_dsi_arg]
    crossCorr_pref_mid_low = crossCorr_pref_same[mid_low_dsi_arg]
    crossCorr_pref_mid = crossCorr_pref_same[mid_dsi_arg]
    crossCorr_pref_mid_high = crossCorr_pref_same[mid_high_dsi_arg]
    crossCorr_pref_high = crossCorr_pref_same[high_dsi_arg]

    crossCorr_null_low = crossCorr_null_same[low_dsi_arg]
    crossCorr_null_mid_low = crossCorr_null_same[mid_low_dsi_arg]
    crossCorr_null_mid = crossCorr_null_same[mid_dsi_arg]
    crossCorr_null_mid_high = crossCorr_null_same[mid_high_dsi_arg]
    crossCorr_null_high = crossCorr_null_same[high_dsi_arg]



    ## plot something about the deltaT 
    crossCorr_pref_low_maxT      = reject_outliers(lags[np.argmax(crossCorr_pref_low,axis=1)])
    crossCorr_pref_mid_low_maxT  = reject_outliers(lags[np.argmax(crossCorr_pref_mid_low,axis=1)])
    crossCorr_pref_mid_maxT      = reject_outliers(lags[np.argmax(crossCorr_pref_mid,axis=1)])
    crossCorr_pref_mid_high_maxT = reject_outliers(lags[np.argmax(crossCorr_pref_mid_high,axis=1)])
    crossCorr_pref_high_maxT     = reject_outliers(lags[np.argmax(crossCorr_pref_high,axis=1)])

    crossCorr_null_low_maxT      = reject_outliers(lags[np.argmax(crossCorr_null_low,axis=1)])
    crossCorr_null_mid_low_maxT  = reject_outliers(lags[np.argmax(crossCorr_null_mid_low,axis=1)])
    crossCorr_null_mid_maxT      = reject_outliers(lags[np.argmax(crossCorr_null_mid,axis=1)])
    crossCorr_null_mid_high_maxT = reject_outliers(lags[np.argmax(crossCorr_null_mid_high,axis=1)])
    crossCorr_null_high_maxT     = reject_outliers(lags[np.argmax(crossCorr_null_high,axis=1)])



    data_plot_pref = [crossCorr_pref_low_maxT, crossCorr_pref_mid_low_maxT, crossCorr_pref_mid_maxT, crossCorr_pref_mid_high_maxT, crossCorr_pref_high_maxT]
    data_plot_null = [crossCorr_null_low_maxT, crossCorr_null_mid_low_maxT, crossCorr_null_mid_maxT, crossCorr_null_mid_high_maxT, crossCorr_null_high_maxT]



    data_plot_pref_len = [len(crossCorr_pref_low_maxT), len(crossCorr_pref_mid_low_maxT), len(crossCorr_pref_mid_maxT), len(crossCorr_pref_mid_high_maxT), len(crossCorr_pref_high_maxT)]
    data_plot_null_len = [len(crossCorr_null_low_maxT), len(crossCorr_null_mid_low_maxT), len(crossCorr_null_mid_maxT), len(crossCorr_null_mid_high_maxT), len(crossCorr_null_high_maxT)]


    fig, ax = plt.subplots()
    bp1 = ax.violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),showmeans=False)
    bp2 = ax.violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)),showmeans=False)
    plt.hlines(0,0,11,colors='black',linestyles='dashed')
    #plt.ylim(-15.5,15.5)
    plt.xlabel('DSI')
    plt.ylabel(r'$\Delta$T')
    legend_elements = [Line2D([0], [0], color='steelblue', lw=4, label='pref'), 
                       Line2D([0], [0], color='tomato', lw=4, label='null')]
    ax.legend(handles=legend_elements)
    plt.xticks(np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    plt.savefig('./Output/DSI_TC/violon_deltaT')
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 16))
    bp1 = axes[0].violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),showmeans=False)
    bp2 = axes[0].violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)),showmeans=False)
    axes[0].hlines(0,0,11,colors='black',linestyles='dashed')
    #plt.ylim(-15.5,15.5)
    axes[0].set_xlabel('DSI')
    axes[0].set_ylabel(r'$\Delta$T')
    legend_elements = [Line2D([0], [0], color='steelblue', lw=4, label='pref'), 
                       Line2D([0], [0], color='tomato', lw=4, label='null')]
    axes[0].legend(handles=legend_elements)
    axes[0].set_xticks(np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)))
    axes[0].set_xticklabels(['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    bar1 = axes[1].bar([0,1,2,3,4],data_plot_pref_len, alpha=0.4, color='gray')
    axes[1].set_ylabel('Nbr Cells')
    plt.savefig('./Output/DSI_TC/violon_deltaT_bar',dpi=300,bbox_inches='tight')
    plt.close()


    ## plot something about the deltaT - do it again, but with abs(deltaT)
    crossCorr_pref_low_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_low,axis=1)]))
    crossCorr_pref_mid_low_maxT  = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_mid_low,axis=1)]))
    crossCorr_pref_mid_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_mid,axis=1)]))
    crossCorr_pref_mid_high_maxT = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_mid_high,axis=1)]))
    crossCorr_pref_high_maxT     = reject_outliers(np.abs(lags[np.argmax(crossCorr_pref_high,axis=1)]))

    crossCorr_null_low_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_low,axis=1)]))
    crossCorr_null_mid_low_maxT  = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_mid_low,axis=1)]))
    crossCorr_null_mid_maxT      = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_mid,axis=1)]))
    crossCorr_null_mid_high_maxT = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_mid_high,axis=1)]))
    crossCorr_null_high_maxT     = reject_outliers(np.abs(lags[np.argmax(crossCorr_null_high,axis=1)]))



    data_plot_pref = [crossCorr_pref_low_maxT, crossCorr_pref_mid_low_maxT, crossCorr_pref_mid_maxT, crossCorr_pref_mid_high_maxT, crossCorr_pref_high_maxT]
    data_plot_null = [crossCorr_null_low_maxT, crossCorr_null_mid_low_maxT, crossCorr_null_mid_maxT, crossCorr_null_mid_high_maxT, crossCorr_null_high_maxT]

    data_plot_pref_len = [len(crossCorr_pref_low_maxT), len(crossCorr_pref_mid_low_maxT), len(crossCorr_pref_mid_maxT), len(crossCorr_pref_mid_high_maxT), len(crossCorr_pref_high_maxT)]
    data_plot_null_len = [len(crossCorr_null_low_maxT), len(crossCorr_null_mid_low_maxT), len(crossCorr_null_mid_maxT), len(crossCorr_null_mid_high_maxT), len(crossCorr_null_high_maxT)]



    fig, ax = plt.subplots()
    bp1 = ax.violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)))
    bp2 = ax.violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)))
    #plt.hlines(0,0,11,colors='black',linestyles='dashed')
    plt.ylim(0)
    plt.xlabel('DSI')
    plt.ylabel(r'$\Delta$T')
    plt.xticks(0.25+np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    plt.savefig('./Output/DSI_TC/violon_deltaT_abs',bbox_inches='tight',dpi=300)
    plt.close()


    fig, ax = plt.subplots()
    bp1 = ax.violinplot(data_plot_pref,positions=np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)))
    bp2 = ax.violinplot(data_plot_null,positions=0.5+np.linspace(1,2*len(data_plot_null), len(data_plot_null)))
    ax.set_ylabel(r'$\Delta$T')
    plt.ylim(0.0, 17.5)
    #plt.hlines(0,0,11,colors='black',linestyles='dashed')
    color = 'gray'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.bar(np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),data_plot_pref_len, alpha=0.3, color=color,width=0.3)
    ax2.bar(0.5+np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),data_plot_null_len, alpha=0.3, color=color,width=0.3)
    ax2.set_ylabel('# cells',color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.ylim(0.0, 17.5)
    plt.xlabel('DSI')
    plt.xticks(0.25+np.linspace(1,2*len(data_plot_pref), len(data_plot_pref)),['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])
    plt.savefig('./Output/DSI_TC/violon_deltaT_abs_bar',bbox_inches='tight',dpi=300)
    plt.close()


    plt.figure(figsize=(16,4))
    plt.subplot(1,5,1)
    plt.title('DSI 0.0 - 0.2')
    plt.plot(lags,np.mean(crossCorr_pref_low,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_low,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=1.1,linestyles = 'dashed', color='black')
    plt.legend()
    plt.ylabel('cross correlation', fontsize=13)
    plt.xlabel(r'$\Delta$T')
    plt.subplot(1,5,2)
    plt.title('DSI 0.2 - 0.4')
    plt.plot(lags,np.mean(crossCorr_pref_mid_low,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid_low,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=1.1,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.subplot(1,5,3)
    plt.title('DSI 0.4 - 0.6')
    plt.plot(lags,np.mean(crossCorr_pref_mid,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=1.1,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.subplot(1,5,4)
    plt.title('DSI 0.6 - 0.8')
    plt.plot(lags,np.mean(crossCorr_pref_mid_high,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_mid_high,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=1.1,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.subplot(1,5,5)
    plt.title('DSI 0.8 - 1.0')
    plt.plot(lags,np.mean(crossCorr_pref_high,axis=0), label='pref', linewidth=3) #[1400:1600]
    plt.plot(lags,np.mean(crossCorr_null_high,axis=0), label='null', linewidth=3) #[1400:1600]
    plt.vlines(0,ymin=-0.2,ymax=1.1,linestyles = 'dashed', color='black')
    plt.legend()
    plt.xlabel(r'$\Delta$T')
    plt.yticks([],[])
    plt.savefig('./Output/DSI_TC/crossCorr_Low_Mid_High',dpi=300, bbox_inches='tight')
    plt.close()


def calcTM(rf1,rf2):

    n_pop1 = len(rf1)
    n_pop2 = len(rf2)

    template_match = np.zeros((n_pop1, n_pop2))

    for p1 in range(n_pop1):
        c1 = rf1[p1].flatten()
        for p2 in range(n_pop2):
            c2 = rf2[p2].flatten()
            template_match[p1,p2] = np.dot(c1,c2)/(np.linalg.norm(c1)*np.linalg.norm(c2))

    return(template_match)

def calcPSTH(data,deltaT):
    ## calculate the peristimulus time histogram
    repeats, n_steps = np.shape(data)
    n_bins = int(n_steps//deltaT)
    psth = np.zeros((repeats,n_bins))
    for r in range(repeats):
        spk_times = np.where(data[r] == 1)[0]
        for b in range(n_bins):
            idx = np.where((spk_times>=(0+deltaT*b) ) & (spk_times<(deltaT+deltaT*b) ))[0]
            psth[r,b] = len(idx)
    return(np.mean(psth,axis=0))


def calcDiff(arr_1,arr_2):
    n_1 = len(arr_1)
    n_2 = len(arr_2)
    diff = np.zeros((n_1,n_2))
    for i in range(n_1):
        diff[i,:] = np.abs(arr_1[i] - arr_2)
    return(diff)  


def calcCrossE1(spkC_times_E1, spkC_times_I1, preff_E1, null_idx):


    n_E1 = np.shape(spkC_times_E1)[-2]
    n_I1 = np.shape(spkC_times_I1)[-2]

    max_t_pref_all = np.zeros((n_E1,n_I1))
    max_t_null_all = np.zeros((n_E1,n_I1))
    ### go through all pyr-cells and calculate the cross-correlation between the 
    ## actual pyr-cell and all inhibitory cells for pref and null direction
    
    for cc in tqdm(range(n_E1),ncols=80): #n_E1

        cell = cc

        ### calculate the cross correlation between the actual pyramidial cell and all other inhibitory cells
        ### for null direction
        spk_times_c1 = spkC_times_E1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),int(preff_E1[cell,2]),:,cell,:]
        spk_times_inhib = spkC_times_I1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),int(preff_E1[cell,2]),:,:,:]
        cross_cor_all = []
        correlate_cell_pref = np.zeros(len(spk_times_inhib[0]))
        for c_i in range(len(spk_times_inhib[0])):
            cross_cor = calcCrossCorr(spk_times_c1[0], spk_times_inhib[0,c_i],100)
            cross_cor_all.append(cross_cor)
            idx_st = np.where(spk_times_inhib[0,c_i] == 1)[0]
            correlate_cell_pref[c_i] = np.corrcoef(spk_times_c1[0], spk_times_inhib[0,c_i])[0,1]



        ### calculate the cross correlation between the actual pyramidial cell and all other inhibitory cells
        ### for null direction
        spk_null_c1 = spkC_times_E1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),null_idx[cell],:,cell,:]
        spk_null_inhib = spkC_times_I1[0,int(preff_E1[cell,0]),int(preff_E1[cell,1]),null_idx[cell],:,:,:]
        cross_cor_all_null = []
        correlate_cell_null = np.zeros(len(spk_times_inhib[0]))
        for c_i in range(len(spk_null_inhib[0])):
            cross_cor = calcCrossCorr(spk_null_c1[0], spk_null_inhib[0,c_i],100)
            cross_cor_all_null.append(cross_cor)
            idx_st = np.where(spk_null_inhib[0,c_i] == 1)[0]
            correlate_cell_null[c_i] = np.corrcoef(spk_null_c1[0], spk_null_inhib[0,c_i])[0,1]

        cross_cor_all = np.asarray(cross_cor_all)
        cross_cor_all_null = np.asarray(cross_cor_all_null)

        ### get the max delta_t 
        t_steps = np.linspace(-99,99,200, dtype='int32')
        
        max_t_pref = np.argmax(cross_cor_all[:,1:-1],axis=1)
        #print(max_t_pref)
        max_t_pref = t_steps[max_t_pref]


        max_t_null = np.argmax(cross_cor_all_null[:,1:-1],axis=1)
        max_t_null = t_steps[max_t_null]


        max_t_pref_all[cc] = max_t_pref
        max_t_null_all[cc] = max_t_null


    np.save('./work/DSI_spike_CrossCorr_pref_short',max_t_pref_all)
    np.save('./work/DSI_spike_CrossCorr_null_short',max_t_null_all)


def calcCycleT(psth_pref_c1, psth_window, total_time):

    fft = np.fft.fft(psth_pref_c1)
    fft_real = fft.real # get the real value part
    fft_img = fft.imag # get the imaginary part
    fft_magn = np.sqrt(fft_real**2 +fft_img**2 ) # calculate the magnitude for each frequency

    fft_phase = np.arctan2(fft_img,fft_real) # calculate the phases in radians!!

    fft_freq = np.fft.fftfreq(len(psth_pref_c1),d= psth_window/total_time )
    #print(fft)
    #print(fft_freq)
    #fft_freq = np.fft.fftshift(fft_freq)

    idx_max = np.argmax(np.abs(fft_magn[1:]))

    max_freq =np.abs(fft_freq[idx_max+1])
    #print(idx_max, fft_freq[idx_max+1])
    ### if Frequency is known -> calculate the time for one cycle -> create corresponding many bins and look, for each bin, when is the peak response (!)
    cycl_len = len(psth_pref_c1)/max_freq
    n_cycles = int(len(psth_pref_c1)/cycl_len)
    #print('cycl_len: ',cycl_len, 'n_cycles: ', n_cycles, 'time: ',int(n_cycles*int(cycl_len)))
    cycles = np.linspace(0,len(psth_pref_c1)-1,n_cycles)
    if (n_cycles*int(cycl_len))%len(psth_pref_c1) == 0: # it fits without a cutoff
        sub_psth = np.reshape(psth_pref_c1, (n_cycles,int(cycl_len)))
        cycl_t = np.argmax(sub_psth,axis=1)
    else:
        cut_off = (n_cycles*int(cycl_len))%len(psth_pref_c1) # time steps that did not fitt into a complete cycle
        sub_psth = np.reshape(psth_pref_c1[0:int(n_cycles*int(cycl_len))], (n_cycles,int(cycl_len)))
        #print(cut_off, np.shape(sub_psth), len(psth_pref_c1[0:int(n_cycles*int(cycl_len))]))
        cycl_t = np.argmax(sub_psth,axis=1)

    #print('cycl_t: ', cycl_t, 'mean_t: ', np.mean(cycl_t), 'median_t: ', np.median(cycl_t))

    
    return(fft_freq, fft_magn, fft_phase, cycl_t, cycl_len)

def spikeDSI():

    if not os.path.exists('Output/DSI_TC/Spikes'):
        os.mkdir('Output/DSI_TC/Spikes')

    if not os.path.exists('Output/DSI_TC/Spikes_2'):
        os.mkdir('Output/DSI_TC/Spikes_2')


    spkC_E1_all, spkC_I1_all, _ = load_Spk_Data()

    spkC_times_E1, spkC_times_I1 = load_Time_Data() #np.load('./work/directGrating_Sinus_SpikeTimes_E1.npy')
    #spkC_times_I1 = np.load('./work/directGrating_Sinus_SpikeTimes_I1.npy')

    ## ignore the first 120 ms, because the network needs some time to
    spkC_times_E1 = spkC_times_E1[:,:,:,:,:,:,120:]
    spkC_times_I1 = spkC_times_I1[:,:,:,:,:,:,120:]

    print(np.shape(spkC_times_E1))


    w_E1 = np.loadtxt('./Input_network/V1weight.txt')
    w_I1 = np.loadtxt('./Input_network/InhibW.txt')


    sim_param = pickle.load( open( './work/directGrating_Sinus_parameter.p', "rb" ) )
    print(sim_param)

    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    

    step_degree = int(360/n_steps)

    print(len(lvl_speed), len(lvl_spFrq), n_steps)


    dsi_kim = np.load('./work/dsi_kim_Cells_TC.npy')
    dsi_kim = dsi_kim[0]

    dsi_argmax = np.argsort(dsi_kim*-1)


    preff_E1 = np.load('./work/dsi_preff_E1_TC.npy')
    preff_E1 = np.asarray(preff_E1[0])



    n_E1, n_lgn = np.shape(w_E1)
    n_I1 = len(w_I1)
    w_E1 = np.reshape(w_E1,(n_E1, int(np.sqrt(n_lgn/2)),int(np.sqrt(n_lgn/2)),2))
    w_I1 = np.reshape(w_I1,(n_I1, int(np.sqrt(n_lgn/2)),int(np.sqrt(n_lgn/2)),2))

    rf_E1 = w_E1[:,:,:,0] - w_E1[:,:,:,1]
    rf_I1 = w_I1[:,:,:,0] - w_I1[:,:,:,1]  
    print(np.shape(rf_E1), np.shape(rf_I1))


    ##NOTE: Use the template match between the receptive fields -> change later to the pref. orientations
    temp = calcTM(rf_E1,rf_I1)
    np.save('./work/DSI_RF_tempE1_I1',temp)
    dsi_bins = np.linspace(0,1,6)



    fig, axs = plt.subplots(nrows=6, ncols=13, figsize=(16,10))
    for i in range(6):
        ci = np.random.randint(0,n_E1)#i+7#
        temp_sort = np.argsort(temp[ci]*-1)
        #axs[2].set_visible(False)
        axs[i,0].imshow(rf_E1[ci],cmap='gray', interpolation='none')
        axs[i,0].axis('off')
        axs[i,0].set_title('Pyr cell')
        axs[i,1].vlines(0,0,1, colors='white', linestyles='dashed')
        axs[i,1].axis('off')
        axs[i,2].imshow(rf_I1[temp_sort[0]],cmap='gray', interpolation='none')
        axs[i,2].axis('off')
        axs[i,2].set_title('TM: %.3f'%(temp[ci,temp_sort[0]]))
        axs[i,3].imshow(rf_I1[temp_sort[1]],cmap='gray', interpolation='none')
        axs[i,3].axis('off')
        axs[i,3].set_title('TM: %.3f'%(temp[ci,temp_sort[1]]))

        axs[i,4].vlines(0,0,1, colors='white', linestyles='dashed')
        axs[i,4].axis('off')

        axs[i,5].imshow(rf_I1[temp_sort[38]],cmap='gray', interpolation='none')
        axs[i,5].axis('off')
        axs[i,5].set_title('TM: %.3f'%(temp[ci,temp_sort[38]]))
        axs[i,6].imshow(rf_I1[temp_sort[40]],cmap='gray', interpolation='none')
        axs[i,6].axis('off')
        axs[i,6].set_title('TM: %.3f'%(temp[ci,temp_sort[40]]))
        axs[i,7].imshow(rf_I1[temp_sort[41]],cmap='gray', interpolation='none')
        axs[i,7].axis('off')
        axs[i,7].set_title('TM: %.3f'%(temp[ci,temp_sort[41]]))
        axs[i,8].imshow(rf_I1[temp_sort[42]],cmap='gray', interpolation='none')
        axs[i,8].axis('off')
        axs[i,8].set_title('TM: %.3f'%(temp[ci,temp_sort[42]]))
        axs[i,9].imshow(rf_I1[temp_sort[43]],cmap='gray', interpolation='none')
        axs[i,9].axis('off')
        axs[i,9].set_title('TM: %.3f'%(temp[ci,temp_sort[43]]))

        axs[i,10].vlines(0,0,1, colors='white', linestyles='dashed')
        axs[i,10].axis('off')

        axs[i,11].imshow(rf_I1[temp_sort[-2]],cmap='gray', interpolation='none')
        axs[i,11].axis('off')
        axs[i,11].set_title('TM: %.3f'%(temp[ci,temp_sort[-2]]))
        axs[i,12].imshow(rf_I1[temp_sort[-1]],cmap='gray', interpolation='none')
        axs[i,12].axis('off')
        axs[i,12].set_title('TM: %.3f'%(temp[ci,temp_sort[-1]]))
    plt.savefig('./Output/DSI_TC/Spikes/testRF_tm',dpi=300, bbox_inches='tight')
    plt.close()

   
    max_prefD = preff_E1[:,2]
    step_degree = int(360/n_steps)
    max_prefD = max_prefD*step_degree
    null_D =(max_prefD+180)%360
    null_idx = np.asarray((null_D/step_degree),dtype='int32') ## index of the null-direction


    ## do not forget
    calcCrossE1(spkC_times_E1,spkC_times_I1, preff_E1, null_idx)


    cross_win = 100    
    cor_inhib_all_pref = np.zeros((n_E1,n_I1))
    cor_inhib_all_null = np.zeros((n_E1,n_I1))
    cor_inhib_all_pref_null = np.zeros((n_E1,n_I1))

    crossCorr_inhib_all_pref = np.zeros((n_E1,n_I1,cross_win*2+1))
    crossCorr_inhib_all_null = np.zeros((n_E1,n_I1,cross_win*2+1))
    crossCorr_inhib_all_pref_null = np.zeros((n_E1,n_I1,cross_win*2+1))
    

    print(np.shape(spkC_times_I1))

    ## go over all pyr cells, and calculate the correlation between the inhibitory cells
    for cc in tqdm(range(n_E1), ncols=80):
        spk_times_c1 = spkC_times_E1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]),int(preff_E1[cc,2]),:,cc,:]
        spk_null_c1 = spkC_times_E1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]), null_idx[cc],:,cc,:]
        sort_arg_temp = np.argsort(temp[cc]*-1)
        ## get the spike times for inhib for the pref and null direction, with respect to the acualt pyr-cell        
        spk_times_inhib_pref = spkC_times_I1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]),int(preff_E1[cc,2]),:,:,:]
        spk_times_inhib_null = spkC_times_I1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]),null_idx[cc],:,:,:]
        
        ## sort the spike times via the template match to the actual pyr cell
        #spk_times_inhib_pref = spk_times_inhib_pref[:,sort_arg_temp]
        #spk_times_inhib_null = spk_times_inhib_null[:,sort_arg_temp]
        n_repeats = len(spk_times_inhib_null)

        #### calculate the correlation between the inhibitory cells (sorted!)
        corr_inhibi_pref = np.zeros((n_repeats,n_I1))
        corr_inhibi_pref_null = np.zeros((n_repeats,n_I1))
        crosscorr_inhibi_pref = np.zeros((n_repeats,n_I1,cross_win*2+1))
        crosscorr_inhibi_pref_null = np.zeros((n_repeats,n_I1,cross_win*2+1))

        #print(np.shape(spk_times_c1),np.shape(spk_times_inhib_pref) )

        for r in range(n_repeats):
            for i_c1 in range(n_I1):
                corr_inhibi_pref[r,i_c1] = np.corrcoef(spk_times_c1[r], spk_times_inhib_pref[r,i_c1])[0,1]
                corr_inhibi_pref_null[r,i_c1] = np.corrcoef(spk_times_c1[r], spk_times_inhib_null[r,i_c1])[0,1]

                crosscorr_inhibi_pref[r,i_c1] = calcCrossCorr(spk_times_c1[r], spk_times_inhib_pref[r,i_c1],cross_win)
                crosscorr_inhibi_pref_null[r,i_c1] = calcCrossCorr(spk_times_c1[r], spk_times_inhib_null[r,i_c1], cross_win)

        cor_inhib_all_pref[cc] = np.mean(corr_inhibi_pref,axis=0)
        cor_inhib_all_pref_null[cc] = np.mean(corr_inhibi_pref_null,axis=0)

        crossCorr_inhib_all_pref[cc] = np.mean(crosscorr_inhibi_pref,axis=0)
        crossCorr_inhib_all_pref_null[cc] = np.mean(crosscorr_inhibi_pref_null,axis=0)

    np.save('./work/DSI_corr_Spike_Inhib_pref_short',cor_inhib_all_pref)
    np.save('./work/DSI_corr_Spike_Inhib_pref_null_short',cor_inhib_all_pref_null)

    np.save('./work/DSI_crosscorr_Spike_Inhib_pref_short',crossCorr_inhib_all_pref)
    np.save('./work/DSI_crosscorr_Spike_Inhib_pref_null_short',crossCorr_inhib_all_pref_null)
    

    ## do it again for psth

    psth_window = 5#10
    cross_win = 50    

    ## calculate the correlation between the pref spike of the pyr Cells and all inhib for pref and null direction
    cor_inhib_all_pref = np.zeros((n_E1,n_I1))
    cor_inhib_all_pref_null = np.zeros((n_E1,n_I1))

    crossCorr_inhib_all_pref = np.zeros((n_E1,n_I1,cross_win*2+1))
    crossCorr_inhib_all_pref_null = np.zeros((n_E1,n_I1,cross_win*2+1))

    for cc in tqdm(range(n_E1), ncols=80):
        ## calculate the psth for the actual pyr-cell
        spk_times_c1 = spkC_times_E1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]),int(preff_E1[cc,2]),:,cc,:]
        psth_pref_c1 = calcPSTH(spk_times_c1,psth_window)
        spk_null_c1 = spkC_times_E1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]), null_idx[cc],:,cc,:]
        psth_null_c1 = calcPSTH(spk_null_c1,psth_window)

        psth_pref_c1 = psth_pref_c1/np.max(psth_pref_c1)
        psth_null_c1 = psth_null_c1/np.max(psth_null_c1)


        sort_arg_temp = np.argsort(temp[cc]*-1)

        ## get the spike times for inhib for the pref and null direction, with respect to the actual pyr-cell       
        spk_times_inhib_pref = spkC_times_I1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]),int(preff_E1[cc,2]),:,:,:]
        spk_times_inhib_null = spkC_times_I1[0,int(preff_E1[cc,0]),int(preff_E1[cc,1]),null_idx[cc],:,:,:]
        
        ## sort the spike times via the template match to the actual pyr cell
        #spk_times_inhib_pref = spk_times_inhib_pref[:,sort_arg_temp]
        #spk_times_inhib_null = spk_times_inhib_null[:,sort_arg_temp]

        #### calculate the correlation to the inhibitory cells
        corr_inhibi_pref = np.zeros(n_I1)
        corr_inhibi_pref_null = np.zeros(n_I1)

        crosscorr_inhibi_pref = np.zeros((n_I1,cross_win*2+1))
        crosscorr_inhibi_pref_null = np.zeros((n_I1,cross_win*2+1))


        for inhib_c in range(n_I1):

            psth_c1_pref = calcPSTH(spk_times_inhib_pref[:,inhib_c,:],psth_window)
            psth_c1_null = calcPSTH(spk_times_inhib_null[:,inhib_c,:],psth_window)

            corr_inhibi_pref[inhib_c] = np.corrcoef(psth_pref_c1, psth_c1_pref)[0,1]
            corr_inhibi_pref_null[inhib_c] = np.corrcoef(psth_pref_c1, psth_c1_null)[0,1]

            crosscorr_inhibi_pref[inhib_c] = calcCrossCorr(psth_pref_c1, psth_c1_pref,cross_win)
            crosscorr_inhibi_pref_null[inhib_c] = calcCrossCorr(psth_pref_c1, psth_c1_null,cross_win)

        cor_inhib_all_pref[cc] = corr_inhibi_pref
        cor_inhib_all_pref_null[cc] = corr_inhibi_pref_null
        crossCorr_inhib_all_pref[cc] = crosscorr_inhibi_pref
        crossCorr_inhib_all_pref_null[cc] = crosscorr_inhibi_pref_null

    np.save('./work/DSI_corr_PSTH_Inhib_pref_short',cor_inhib_all_pref)
    np.save('./work/DSI_corr_PSTH_Inhib_pref_null_short',cor_inhib_all_pref_null)                
    np.save('./work/DSI_CrossCorr_PSTH_Inhib_pref_short',crossCorr_inhib_all_pref)
    np.save('./work/DSI_CrossCorr_PSTH_Inhib_pref_null_short',crossCorr_inhib_all_pref_null)   

def moreSpikesCorr():


    cor_inhib_all_pref = np.load('./work/DSI_corr_PSTH_Inhib_pref_short.npy')
    cor_inhib_all_pref_null = np.load('./work/DSI_corr_PSTH_Inhib_pref_null_short.npy')  

    print(np.shape(cor_inhib_all_pref))
    plt.figure()
    plt.subplot(121)
    plt.imshow(cor_inhib_all_pref, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(cor_inhib_all_pref_null, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.savefig('Output/DSI_TC/Spikes_2/Correlation_imshow_complete')    
    plt.close()

    temp = np.load('./work/DSI_RF_tempE1_I1.npy')
    n_pyrCells, n_Inhib = np.shape(temp)    


    dsi_kim = np.load('./work/dsi_kim_Cells_TC.npy')
    dsi_kim = dsi_kim[0]

    n_bins = 5
    dsi_bins = np.linspace(0,1,n_bins+1)

    n_tm_bins = 5
    tm_bound = np.max(np.abs(temp))

    tm_bins = np.linspace(-0.8,0.8,n_tm_bins+1)
    #tm_bins = np.linspace(0,0.8,n_tm_bins+1) 
    print(tm_bins)

    mean_bin_pref_all = []
    std_bin_pref_all  = [] 
    leng_bin_pref_all = []

    mean_bin_null_all = []
    std_bin_null_all  = []
    leng_bin_null_all = []

    for b in range(n_bins):
        idx_bin = np.where((dsi_kim >= dsi_bins[b]) & (dsi_kim < dsi_bins[b+1]))[0]
        #print('idx_bin: ',idx_bin)
        corr_pref = cor_inhib_all_pref[idx_bin]
        corr_pref_null = cor_inhib_all_pref_null[idx_bin]

        corr_pref_sort = np.copy(corr_pref)
        corr_pref_null_sort = np.copy(corr_pref_null)
        for cc in range(len(idx_bin)):
            sort_arg_temp = np.argsort(temp[idx_bin[cc]])
            corr_pref_sort[cc] = corr_pref_sort[cc, sort_arg_temp]
            corr_pref_null_sort[cc] = corr_pref_null_sort[cc, sort_arg_temp]

        x = np.linspace(0,n_Inhib-1,n_Inhib,dtype='int32')

        plt.figure()
        plt.plot(x, np.mean(corr_pref_sort,axis=0),'o-',color='steelblue', label='pref')
        plt.fill_between(x, np.mean(corr_pref_sort,axis=0)-np.std(corr_pref_sort,axis=0,ddof=1), np.mean(corr_pref_sort,axis=0)+np.std(corr_pref_sort,axis=0,ddof=1) ,color='steelblue', alpha=0.5  )
        plt.plot(x, np.mean(corr_pref_null_sort,axis=0),'o-',color='tomato', label='null')
        plt.fill_between(x, np.mean(corr_pref_null_sort,axis=0)-np.std(corr_pref_null_sort,axis=0,ddof=1), np.mean(corr_pref_null_sort,axis=0)+np.std(corr_pref_null_sort,axis=0,ddof=1) ,color='tomato', alpha=0.5  )
        plt.legend()
        plt.hlines(0,0,80)
        plt.ylim(-0.7,0.7)
        plt.savefig('Output/DSI_TC/Spikes_2/mean_corr_bin_%i'%b)


        ## get the values for all the inihbitory in a certain tm-range, for all pyr cells
        bin_corr_pref_bins = []
        bin_corr_null_bins = []
        for t in range(n_tm_bins):
            bin_corr_pref = []
            bin_corr_null = []
            for c in range(len(idx_bin)):
                #print(tm_bins[t], tm_bins[t+1])
                idx_tm = np.where((temp[idx_bin[c]] >= tm_bins[t]) & (temp[idx_bin[c]] < tm_bins[t+1]))[0]
                #print(idx_tm)
                #bin_corr_pref.append(np.asarray(corr_pref[c,idx_tm]))
                bin_corr_pref = np.append(bin_corr_pref,np.asarray(corr_pref[c,idx_tm]))
                bin_corr_null = np.append(bin_corr_null,np.asarray(corr_pref_null[c,idx_tm]))
            bin_corr_pref_bins.append(np.asarray(bin_corr_pref))   
            bin_corr_null_bins.append(np.asarray(bin_corr_null))
        """
        x = np.linspace(0,len(bin_corr_pref_bins)-1,len(bin_corr_pref_bins),dtype='int32')
        #print(len(bin_corr_pref_bins))
        fig, ax = plt.subplots()
        ax.violinplot(bin_corr_pref_bins, x, showmeans=True,showextrema =False)
        ax.violinplot(bin_corr_null_bins, x+0.25, showmeans=True,showextrema =False)
        ax.hlines(0,-1,len(bin_corr_pref_bins), alpha=0.5)
        plt.ylim(-1,1)
        plt.savefig('Output/DSI_TC/Spikes_2/violin_corr_bin_%i'%b)
        plt.close()
        """
        mean_bin_pref =np.zeros(len(bin_corr_pref_bins)) 
        std_bin_pref = np.zeros(len(bin_corr_pref_bins)) 
        leng_bin_pref= np.zeros(len(bin_corr_pref_bins)) 

        mean_bin_null = np.zeros(len(bin_corr_null_bins)) 
        std_bin_null  = np.zeros(len(bin_corr_null_bins)) 
        leng_bin_null = np.zeros(len(bin_corr_null_bins)) 


        for bb in range(len(bin_corr_pref_bins)):
            mean_bin_pref[bb] = np.mean(reject_outliers(bin_corr_pref_bins[bb]))
            std_bin_pref[bb] = np.std(reject_outliers(bin_corr_pref_bins[bb]),ddof=1)
            leng_bin_pref[bb] = len(bin_corr_pref_bins[bb])

            mean_bin_null[bb] = np.mean(reject_outliers(bin_corr_null_bins[bb]))
            std_bin_null[bb] = np.std(reject_outliers(bin_corr_null_bins[bb]),ddof=1)
            leng_bin_null[bb] = len(bin_corr_null_bins[bb])

        mean_bin_pref_all.append(mean_bin_pref)
        std_bin_pref_all.append(std_bin_pref)
        leng_bin_pref_all.append(leng_bin_pref)
        
        mean_bin_null_all.append(mean_bin_null)
        std_bin_null_all.append(std_bin_null)
        leng_bin_null_all.append(leng_bin_null)

        #print(leng_bin_null)

        x = np.linspace(0,len(bin_corr_pref_bins)-1,len(bin_corr_pref_bins),dtype='int32')
        plt.figure()
        plt.scatter(x,mean_bin_pref,c = 'steelblue', s = leng_bin_pref, alpha=0.5, label='pref')
        plt.scatter(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null,c = 'tomato', s = leng_bin_null, alpha=0.5, label='nul')
        plt.hlines(0,-1,len(mean_bin_pref), alpha=0.5)
        plt.ylim(-.75,.75)
        plt.xticks(np.linspace(0,n_tm_bins-1,5),np.linspace(0,0.8,5) )
        plt.legend()
        plt.savefig('Output/DSI_TC/Spikes_2/scatter_corr_bin_size_%i'%b)
        plt.close()

        x = np.linspace(0,len(bin_corr_pref_bins)-1,len(bin_corr_pref_bins),dtype='int32')
        plt.figure()
        plt.errorbar(x,mean_bin_pref,std_bin_pref, linestyle='None', marker='o' ,c = 'steelblue', label='pref')
        plt.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null,std_bin_null, linestyle='None', marker='o' ,c = 'tomato', label='nul')
        plt.hlines(0,-1,len(mean_bin_pref), alpha=0.5)
        plt.ylim(-.75,.75)
        plt.xticks(np.linspace(0,n_tm_bins-1,5),np.linspace(0,0.8,5) )
        plt.legend()
        plt.savefig('Output/DSI_TC/Spikes_2/error_corr_bin_%i'%b)
        plt.close()


    marker_list = ['o','s','<','D']
    x = np.linspace(0,len(bin_corr_pref_bins)-1,len(bin_corr_pref_bins),dtype='int32')
    x_labels= ['%.2f'%v for v in np.linspace(-0.8,0.8,5)]
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2,ncols = 2, figsize=(10,8))
    
    ax1.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[0],std_bin_pref_all[0], linestyle='None', marker= marker_list[0] ,c = 'steelblue', label='pref_low')
    ax1.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[-1],std_bin_pref_all[-1], linestyle='None', marker= marker_list[1] ,c = 'tomato', label='pref_high')
    ax1.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax1.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(-0.8,0.8)
    ax1.legend()
    ax1.set_ylabel('Correlation')

    ax2.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[0],std_bin_null_all[0], linestyle='None', marker= marker_list[2]  ,c = 'navy', label='null_low')
    ax2.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[-1],std_bin_null_all[-1], linestyle='None', marker= marker_list[3] ,c = 'orange', label='null_high')
    ax2.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax2.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(-0.8,0.8)
    ax2.legend()


    ax3.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[0],std_bin_pref_all[0], linestyle='None', marker= marker_list[0] ,c = 'steelblue', label='pref_low')
    ax3.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[0],std_bin_null_all[0], linestyle='None', marker= marker_list[2] ,c = 'navy', label='null_low')
    ax3.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax3.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax3.set_xticklabels(x_labels)
    ax3.set_ylim(-0.8,0.8)
    ax3.legend()
    ax3.set_ylabel('Correlation')
    ax3.set_xlabel('Cosine')

    ax4.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[-1],std_bin_pref_all[-1], linestyle='None', marker= marker_list[1] ,c = 'tomato', label='pref_high')
    ax4.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[-1],std_bin_null_all[-1], linestyle='None', marker= marker_list[3] ,c = 'orange', label='null_high')
    ax4.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax4.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax4.set_xticklabels(x_labels)
    ax4.set_ylim(-0.8,0.8)
    ax4.legend()
    ax4.set_xlabel('Cosine')

    ## set shared axes
    ax1.get_shared_x_axes().join(ax3, ax1) 
    ax1.set_xticklabels([])
    ax2.get_shared_y_axes().join(ax1, ax2) 
    ax2.set_yticklabels([])
    ax2.get_shared_x_axes().join(ax1, ax4) 
    ax2.set_xticklabels([])
    ax4.get_shared_y_axes().join(ax3, ax4) 
    ax4.set_yticklabels([])
    plt.savefig('Output/DSI_TC/Spikes_2/error_corr_bin_low_high',dpi=300,bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(5,3))
    plt.bar(x-0.2, leng_bin_pref_all[0],width=0.4, color='steelblue', label='low DSI')
    plt.bar(x+0.2, leng_bin_pref_all[-1],width=0.4, color='tomato', label='high DSI')    
    plt.ylabel('# pairs')
    plt.xlabel('Cosine')
    plt.legend()
    plt.xticks(np.linspace(0,n_tm_bins-1,5), x_labels)
    plt.savefig('Output/DSI_TC/Spikes_2/template_hists_bar',dpi=300,bbox_inches='tight')
    plt.close()

    x = np.linspace(0,len(bin_corr_pref_bins)-1,len(bin_corr_pref_bins),dtype='int32')
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2,ncols = 2, figsize=(16,10))
    
    ax1.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[0],std_bin_pref_all[0], linestyle='None', marker='o' ,c = 'steelblue', label='pref_low')
    ax1.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[-1],std_bin_pref_all[-1], linestyle='None', marker='o' ,c = 'tomato', label='pref_high')
    ax1.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax1.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax1.set_xticklabels(np.linspace(0,0.8,5))
    ax1.set_ylim(-0.8,0.8)
    ax1.legend()
    ax1_2 = ax1.twinx()
    ax1_2.set_ylabel('# cells', color='gray')
    ax1_2.plot(x, leng_bin_pref_all[0], 'o', color='steelblue', alpha=0.5)
    ax1_2.plot(x, leng_bin_pref_all[-1], 'o', color='tomato', alpha=0.5)   
    ax1_2.tick_params(axis='y', labelcolor='gray')
    ax1_2.set_ylim(-1800,2500)


    ax2.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[0],std_bin_null_all[0], linestyle='None', marker='o' ,c = 'navy', label='null_low')
    ax2.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[-1],std_bin_null_all[-1], linestyle='None', marker='o' ,c = 'orange', label='null_high')
    ax2.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax2.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax2.set_xticklabels(np.linspace(0,0.8,5))
    ax2.set_ylim(-0.8,0.8)
    ax2.legend()
    ax2_2 = ax2.twinx()
    ax2_2.set_ylabel('# cells', color='gray')
    ax2_2.plot(x, leng_bin_null_all[0], 'o', color='navy', alpha=0.5)
    ax2_2.plot(x, leng_bin_null_all[-1], 'o', color='orange', alpha=0.5)   
    ax2_2.tick_params(axis='y', labelcolor='gray')
    ax2_2.set_ylim(-1800,2500)    


    ax3.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[0],std_bin_pref_all[0], linestyle='None', marker='o' ,c = 'steelblue', label='pref_low')
    ax3.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[0],std_bin_null_all[0], linestyle='None', marker='o' ,c = 'navy', label='null_low')
    ax3.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax3.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax3.set_xticklabels(np.linspace(0,0.8,5))
    ax3.set_ylim(-0.8,0.8)
    ax3.legend()
    ax3_2 = ax3.twinx()
    ax3_2.set_ylabel('# cells', color='gray')
    ax3_2.plot(x, leng_bin_pref_all[0], 'o', color='steelblue', alpha=0.5)
    ax3_2.plot(x, leng_bin_null_all[0], 'o', color='navy', alpha=0.5)   
    ax3_2.tick_params(axis='y', labelcolor='gray')
    ax3_2.set_ylim(-1800,2500)    

    ax4.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_pref_all[-1],std_bin_pref_all[-1], linestyle='None', marker='o' ,c = 'tomato', label='pref_high')
    ax4.errorbar(x+np.random.normal(0.0,0.1,len(bin_corr_pref_bins)),mean_bin_null_all[-1],std_bin_null_all[-1], linestyle='None', marker='o' ,c = 'orange', label='null_high')
    ax4.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    ax4.set_xticks(np.linspace(0,n_tm_bins-1,5))
    ax4.set_xticklabels(np.linspace(0,0.8,5))
    ax4.set_ylim(-0.8,0.8)
    ax4.legend()
    ax4_2 = ax4.twinx()
    ax4_2.set_ylabel('# cells', color='gray')
    ax4_2.plot(x, leng_bin_pref_all[-1], 'o', color='tomato', alpha=0.5)
    ax4_2.plot(x, leng_bin_null_all[-1], 'o', color='orange', alpha=0.5)   
    ax4_2.tick_params(axis='y', labelcolor='gray')
    ax4_2.set_ylim(-1800,2500)


    plt.savefig('Output/DSI_TC/Spikes_2/error_corr_bin_low_high_bars',dpi=300,bbox_inches='tight')
    plt.close()




    x = np.linspace(0,len(bin_corr_pref_bins)-1,len(bin_corr_pref_bins),dtype='int32')
    plt.figure()
    plt.subplot(221)
    plt.scatter(x,mean_bin_pref_all[0], c = leng_bin_pref_all[0], cmap='viridis')
    plt.errorbar(x,mean_bin_pref_all[0],std_bin_pref_all[0], linestyle='None', marker=None ,c = 'steelblue', label='pref_low')
    plt.errorbar(x,mean_bin_pref_all[-1],std_bin_pref_all[-1], linestyle='None', marker=None ,c = 'tomato', label='pref_high')
    plt.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    plt.xticks(np.linspace(0,n_tm_bins-1,5),np.linspace(0,0.8,5) )
    plt.ylim(-0.5,0.7)
    plt.legend()
    plt.subplot(222)
    plt.errorbar(x,mean_bin_null_all[0],std_bin_null_all[0], linestyle='None', marker=None ,c = 'navy', label='null_low')
    plt.errorbar(x,mean_bin_null_all[-1],std_bin_null_all[-1], linestyle='None', marker=None ,c = 'orange', label='null_high')
    plt.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    plt.xticks(np.linspace(0,n_tm_bins-1,5),np.linspace(0,0.8,5) )
    plt.ylim(-0.5,0.7)
    plt.legend()
    plt.subplot(223)
    plt.errorbar(x,mean_bin_pref_all[0],std_bin_pref_all[0], linestyle='None', marker=None,c = 'steelblue', label='pref_low')
    plt.errorbar(x,mean_bin_null_all[0],std_bin_null_all[0], linestyle='None', marker=None ,c = 'navy', label='null_low')
    plt.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    plt.xticks(np.linspace(0,n_tm_bins-1,5),np.linspace(0,0.8,5) )
    plt.ylim(-0.5,0.7)
    plt.legend()
    plt.subplot(224)
    plt.errorbar(x,mean_bin_pref_all[-1],std_bin_pref_all[-1], linestyle='None', marker=None ,c = 'tomato', label='pref_high')
    plt.errorbar(x,mean_bin_null_all[-1],std_bin_null_all[-1], linestyle='None', marker=None ,c = 'orange', label='null_high')
    plt.hlines(0,-1,len(mean_bin_pref_all[0]), alpha=0.5)
    plt.xticks(np.linspace(0,n_tm_bins-1,5),np.linspace(0,0.8,5) )
    plt.ylim(-0.5,0.7)
    plt.legend()

    plt.savefig('Output/DSI_TC/Spikes_2/error_corr_bin_low_high_cmaps')
    plt.close()

def moreSpikesCross():

    cross_cor_inhib_all_pref = np.load('./work/DSI_CrossCorr_PSTH_Inhib_pref_short.npy')
    cross_cor_inhib_all_pref_null = np.load('./work/DSI_CrossCorr_PSTH_Inhib_pref_null_short.npy')  


    w_size = np.shape(cross_cor_inhib_all_pref)[2]
    t_steps= np.linspace(-100,100,w_size,dtype='int32')
    
    temp = np.load('./work/DSI_RF_tempE1_I1.npy')
    n_pyrCells, n_Inhib = np.shape(temp)    


    dsi_kim = np.load('./work/dsi_kim_Cells_TC.npy')
    dsi_kim = dsi_kim[0]

    n_bins = 4
    dsi_bins = np.linspace(0,1,n_bins+1)

    n_tm_bins = 7
    tm_bound = np.max(np.abs(temp))
    #tm_bins = np.linspace(tm_bound,-tm_bound,n_tm_bins+1)
    tm_bins = np.linspace(tm_bound,0,n_tm_bins+1) 
    #print(tm_bins)

    for b in range(n_bins):
        idx_bin = np.where((dsi_kim >= dsi_bins[b]) & (dsi_kim < dsi_bins[b+1]))[0]
        print('idx_bin: ',idx_bin)
        corr_pref = cross_cor_inhib_all_pref[idx_bin]
        corr_pref_null = cross_cor_inhib_all_pref_null[idx_bin]
        deltaT_pref = t_steps[np.argmax(corr_pref,axis=-1)]
        deltaT_pref_null = t_steps[np.argmax(corr_pref_null,axis=-1)]

        deltaT_pref_sort = np.copy(deltaT_pref)
        deltaT_pref_null_sort = np.copy(deltaT_pref_null)
        for cc in range(len(idx_bin)):
            sort_arg_temp = np.argsort(temp[idx_bin[cc]]*-1)
            deltaT_pref_sort[cc] = deltaT_pref_sort[cc, sort_arg_temp]
            deltaT_pref_null_sort[cc] = deltaT_pref_null_sort[cc, sort_arg_temp]

        x = np.linspace(0,n_Inhib-1,n_Inhib,dtype='int32')

        plt.figure()
        plt.plot(x, np.mean(deltaT_pref_sort,axis=0),'o-',color='steelblue', label='pref')
        plt.fill_between(x, np.mean(deltaT_pref_sort,axis=0)-np.std(deltaT_pref_sort,axis=0,ddof=1), np.mean(deltaT_pref_sort,axis=0)+np.std(deltaT_pref_sort,axis=0,ddof=1) ,color='steelblue', alpha=0.5  )
        plt.plot(x, np.mean(deltaT_pref_null_sort,axis=0),'o-',color='tomato', label='null')
        plt.fill_between(x, np.mean(deltaT_pref_null_sort,axis=0)-np.std(deltaT_pref_null_sort,axis=0,ddof=1), np.mean(deltaT_pref_null_sort,axis=0)+np.std(deltaT_pref_null_sort,axis=0,ddof=1) ,color='tomato', alpha=0.5  )
        plt.legend()
        #plt.hlines(0,0,80)
        #plt.ylim(-0.7,0.7)
        plt.savefig('Output/DSI_TC/Spikes_2/mean_deltaT_bin_%i'%b)
        plt.close()

        ## get the values for all the inihbitory in a certain tm-range, for all pyr cells
        bin_deltaT_pref_bins = []
        bin_deltaT_null_bins = []
        for t in range(n_tm_bins):
            bin_deltaT_pref = []
            bin_deltaT_null = []
            for c in range(len(idx_bin)):
                idx_tm = np.where((temp[idx_bin[c]] < tm_bins[t]) & (temp[idx_bin[c]]>= tm_bins[t+1]))[0]
                bin_deltaT_pref = np.append(bin_deltaT_pref,np.asarray(deltaT_pref[c,idx_tm]))
                bin_deltaT_null = np.append(bin_deltaT_null,np.asarray(deltaT_pref_null[c,idx_tm]))
            bin_deltaT_pref_bins.append(np.asarray(bin_deltaT_pref))   
            bin_deltaT_null_bins.append(np.asarray(bin_deltaT_null))

        x = np.linspace(0,len(bin_deltaT_pref_bins)-1,len(bin_deltaT_pref_bins),dtype='int32')
        """
        #print(x)
        #print(len(bin_deltaT_pref_bins))
        fig, ax = plt.subplots()
        ax.violinplot(bin_deltaT_pref_bins, x, showmeans=True,showextrema =False)
        ax.violinplot(bin_deltaT_null_bins, x+0.25, showmeans=True,showextrema =False)
        #ax.hlines(0,-1,len(bin_deltaT_pref_bins), alpha=0.5)
        #plt.ylim(-1,1)
        plt.savefig('Output/DSI_TC/Spikes_2/violin_deltaT_bin_%i'%b)
        plt.close()
        """
        mean_bin_pref =np.zeros(len(bin_deltaT_pref_bins)) 
        std_bin_pref = np.zeros(len(bin_deltaT_pref_bins)) 
        leng_bin_pref= np.zeros(len(bin_deltaT_pref_bins)) 

        mean_bin_null = np.zeros(len(bin_deltaT_null_bins)) 
        std_bin_null  = np.zeros(len(bin_deltaT_null_bins)) 
        leng_bin_null = np.zeros(len(bin_deltaT_null_bins)) 


        for bb in range(len(bin_deltaT_pref_bins)):
            mean_bin_pref[bb] = np.mean(bin_deltaT_pref_bins[bb])
            std_bin_pref[bb] = np.std(bin_deltaT_pref_bins[bb],ddof=1)
            leng_bin_pref[bb] = len(bin_deltaT_pref_bins[bb])

            mean_bin_null[bb] = np.mean(bin_deltaT_null_bins[bb])
            std_bin_null[bb] = np.std(bin_deltaT_null_bins[bb],ddof=1)
            leng_bin_null[bb] = len(bin_deltaT_null_bins[bb])

        #print(leng_bin_null)

        x = np.linspace(0,len(bin_deltaT_pref_bins)-1,len(bin_deltaT_pref_bins),dtype='int32')
        plt.figure()
        plt.scatter(x,mean_bin_pref,c = 'steelblue', s = leng_bin_pref)
        plt.scatter(x,mean_bin_null,c = 'tomato', s = leng_bin_null)
        #ax.hlines(0,-1,len(mean_bin_pref), alpha=0.5)
        #plt.ylim(-1,1)
        plt.savefig('Output/DSI_TC/Spikes_2/scatter_deltaT_bin_%i'%b)
        plt.close()

def analyzeCrossCorr():

    cor_inhib_all_pref = np.load('./work/DSI_corr_Spike_Inhib_pref_short.npy')
    cor_inhib_all_pref_null = np.load('./work/DSI_corr_Spike_Inhib_pref_null_short.npy')

    crossCorr_inhib_all_pref = np.load('./work/DSI_crosscorr_Spike_Inhib_pref_short.npy')
    crossCorr_inhib_all_pref_null = np.load('./work/DSI_crosscorr_Spike_Inhib_pref_null_short.npy')


    print(np.shape(crossCorr_inhib_all_pref))
    
    w_size = np.shape(crossCorr_inhib_all_pref)[2]
    t_steps= np.linspace(-100,100,w_size,dtype='int32')
    
    temp = np.load('./work/DSI_RF_tempE1_I1.npy')
    n_pyrCells, n_Inhib = np.shape(temp)    


    dsi_kim = np.load('./work/dsi_kim_Cells_TC.npy')
    dsi_kim = dsi_kim[0]

    n_bins = 5
    dsi_bins = np.linspace(0,1,n_bins+1)

    n_tm_bins = 7
    tm_bound = np.max(np.abs(temp))
    #tm_bins = np.linspace(tm_bound,-tm_bound,n_tm_bins+1)
    tm_bins = np.linspace(tm_bound,0,n_tm_bins+1) 
    #print(tm_bins)

    for b in range(n_bins):
        idx_bin = np.where((dsi_kim >= dsi_bins[b]) & (dsi_kim < dsi_bins[b+1]))[0]
        #print('idx_bin: ',idx_bin)
        corr_pref = crossCorr_inhib_all_pref[idx_bin]
        corr_pref_null = crossCorr_inhib_all_pref_null[idx_bin]
        deltaT_pref = t_steps[np.argmax(corr_pref,axis=-1)]
        deltaT_pref_null = t_steps[np.argmax(corr_pref_null,axis=-1)]

        deltaT_pref_sort = np.copy(deltaT_pref)
        deltaT_pref_null_sort = np.copy(deltaT_pref_null)
        for cc in range(len(idx_bin)):
            sort_arg_temp = np.argsort(temp[idx_bin[cc]])
            deltaT_pref_sort[cc] = deltaT_pref_sort[cc, sort_arg_temp]
            deltaT_pref_null_sort[cc] = deltaT_pref_null_sort[cc, sort_arg_temp]

        x = np.linspace(0,n_Inhib-1,n_Inhib,dtype='int32')
        plt.figure()
        plt.plot(x, np.mean(deltaT_pref_sort,axis=0),'o-',color='steelblue', label='pref')
        plt.fill_between(x, np.mean(deltaT_pref_sort,axis=0)-np.std(deltaT_pref_sort,axis=0,ddof=1), np.mean(deltaT_pref_sort,axis=0)+np.std(deltaT_pref_sort,axis=0,ddof=1) ,color='steelblue', alpha=0.5  )
        plt.plot(x, np.mean(deltaT_pref_null_sort,axis=0),'o-',color='tomato', label='null')
        plt.fill_between(x, np.mean(deltaT_pref_null_sort,axis=0)-np.std(deltaT_pref_null_sort,axis=0,ddof=1), np.mean(deltaT_pref_null_sort,axis=0)+np.std(deltaT_pref_null_sort,axis=0,ddof=1) ,color='tomato', alpha=0.5  )
        plt.legend()
        #plt.hlines(0,0,80)
        #plt.ylim(-0.7,0.7)
        plt.savefig('Output/DSI_TC/Spikes_2/mean_deltaT_bin_%i'%b)
        plt.close()
        


#-----------------------------------------------------------------------------
if __name__=="__main__":

    main()
    moreDSI()
    spikeDSI()
    #moreSpikesCorr()
    #analyzeCrossCorr()
    #moreSpikesCross()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from scipy.optimize import leastsq

def createDirs():
    if not os.path.exists('Output/speed/'):
        os.mkdir('Output/speed/')

    if not os.path.exists('Output/speed/Ex1'):
        os.mkdir('Output/speed/Ex1/')

    if not os.path.exists('Output/speed/Ex1/imShow'):
        os.mkdir('Output/speed/Ex1/imShow')

    if not os.path.exists('Output/speed/Ex1/contur'):
        os.mkdir('Output/speed/Ex1/contur')

    if not os.path.exists('Output/speed/Ex1/SpeedCurve'):
        os.mkdir('Output/speed/Ex1/SpeedCurve')


def plotSingleCell(spkC,lvl_speed, lvl_spFrq, path):
    # use the response to see, if a cell is speed selective
    n_cells = np.shape(spkC)[3]

    for i in range(n_cells):        
        cellA = spkC[:,:,:,i]
        pref_dir = np.where(cellA == np.max(cellA))[2]
        pref_dir = pref_dir[0]
        cont = cellA[:,:,pref_dir]


        plt.figure()
        plt.contourf(lvl_spFrq,np.linspace(0,len(lvl_speed)-1,len(lvl_speed)),cont ,cmap='gray_r', levels=4)
        plt.ylabel('speed [Hz]')
        plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),lvl_speed)
        plt.xlabel('spat. freq [cycles/Img]')
        plt.xticks(lvl_spFrq)#(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)))
        plt.colorbar()
        plt.savefig('Output/speed/'+path+'/contur/contour_%i'%(i),dpi=300,bbox_inches='tight')
        plt.close()


        plt.figure()
        plt.imshow( np.flip(cont,axis=0) ,cmap='gray_r')
        plt.ylabel('speed [Hz]')
        plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),np.flip(lvl_speed,axis=0))
        plt.xlabel('spat. freq [cycles/Img]')
        plt.xticks(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)), lvl_spFrq)
        plt.colorbar()
        plt.savefig('Output/speed/'+path+'/imShow/imShow_%i'%(i),dpi=300,bbox_inches='tight')
        plt.close()


        idx_Freq = np.where(np.max(cont))
        idx_Freq = idx_Freq[0]
        plt.figure()
        plt.plot(cont[:,idx_Freq],'-o')
        plt.ylabel('Spike Count')
        plt.xticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),lvl_speed)
        plt.savefig('Output/speed/'+path+'/SpeedCurve/cell_%i'%(i))
        plt.close()

def createGaussianFitt(spk_tf_sp, lvl_speed, lvl_spFrq, fitting_parameters):
    A = np.max(spk_tf_sp) # peak response
    len_tf = len(lvl_speed) 
    len_sf = len(lvl_spFrq)
    spk_tf_sp = np.reshape(spk_tf_sp,(len_tf,len_sf))
    sf_0 = np.argmax(np.mean(spk_tf_sp,axis=0)) # max sf, averraged over all tf's
    tf_0 = np.argmax(np.mean(spk_tf_sp,axis=1)) # max tf, averraged over all sf's
    spk_tf_sp = spk_tf_sp.flatten()

    sf_0 = lvl_spFrq[sf_0]
    tf_0 = lvl_spFrq[tf_0]


    sigma_sf = fitting_parameters[0]
    sigma_tf = fitting_parameters[1]
    zeta = fitting_parameters[2]
    xi = fitting_parameters[3]

    gauss = np.zeros((len_sf,len_tf))
    for x in range(len_sf):
        for y in range(len_tf):
            sf = lvl_spFrq[x] # actual spatial frequency
            tf = lvl_speed[y] # actual temporal frequency (speed)

            log2_tfp_sf = xi*(np.log2(sf) - np.log2(sf_0))+np.log2(tf_0)
            exp_1 = np.exp(- ((np.log2(sf) - np.log2(sf_0))**2)/(2* sigma_sf**2) )
            exp_2_1 = - (np.log(tf) - np.log2(log2_tfp_sf))**2 
            exp_2_2 = 2*(sigma_tf + zeta*(np.log2(tf) - np.log2(log2_tfp_sf) ) )**2
            gauss[x,y] = A*exp_1*np.exp(exp_2_1 / exp_2_2) - np.exp(-1/(zeta**2))

    gauss = gauss.T

    return(gauss.flatten())

def modifyParameters():

    fitting_parameters = np.zeros(4)
    fitting_parameters[0] = np.random.rand()+0.0001#  sigma_sf
    fitting_parameters[1] = np.random.rand()+0.0001# sigma_tf
    fitting_parameters[2] = np.random.rand()+0.0001# zeta
    fitting_parameters[3] = np.random.rand()+0.0001# xi

    return(fitting_parameters)

def fittSpatioTemporal(spk_tf_sp, lvl_speed, lvl_spFrq):
    ####
    # Fit spatio-temporal tuning with a two-dimensional Gauss as described in Priebe et al. (2006)
    ###
    rounds = 100
    n_cells, h,w = np.shape(spk_tf_sp)
    bestFittsImages = np.zeros((n_cells, w, h))
    bestFittsParameter=np.zeros((n_cells, 4))    
    bestFitt_error = np.zeros(n_cells)



    for i in range(n_cells):

        tf_sp = spk_tf_sp[i]
        tf_sp = tf_sp.flatten()
        # start fitting by choose the starting values randomly
        fitting_parameters = modifyParameters()
        gauss_fitt = createGaussianFitt(tf_sp, lvl_speed, lvl_spFrq,fitting_parameters)

        actRound = 0
        minError = 1.0

        sys.stdout.write('.')
        sys.stdout.flush()

        # perform the fitting algorithm a few hundred times with different starting parameters
        while(actRound < rounds):
            initParams = np.copy(fitting_parameters)
            initGauss = np.copy(gauss_fitt)
            initMSE = np.copy(np.sum((initGauss - tf_sp)**2)/(np.sum(tf_sp**2)))

            ## fitt one time
            errfunc = lambda fitting_parameters,tf_sp: (createGaussianFitt(tf_sp, lvl_speed, lvl_spFrq,fitting_parameters) - tf_sp)**2 
            # quadratic Distance to the target
            params, success = leastsq(errfunc,fitting_parameters, args=(tf_sp),maxfev=2000,epsfcn=0.05)

            # use the fitted params as new params 
            fitting_parameters=params
            # create a new gauss with new parameters
            gauss_fitt = createGaussianFitt(tf_sp, lvl_speed, lvl_spFrq,fitting_parameters)
            # calc the actual error
            actError = np.sum((initGauss - tf_sp)**2)/(np.sum(tf_sp**2))
      
            if(actError > initMSE):
               actError = initMSE
               gauss_fitt = initParams
               fitting_parameters=initParams
            if(minError > actError):
                minError = actError
                bestFittsImages[i,:,:] = np.reshape(gauss_fitt,(w,h) )
                bestFittsParameter[i] = fitting_parameters #params
                bestFitt_error[i] = minError
               
            fitting_parameters = modifyParameters() #choose new for the next fitting round
            actRound+=1

    np.save('./work/speed_fitt_Images',bestFittsImages)
    np.save('./work/speed_fitt_Parameter',bestFittsParameter)
    np.save('./work/speed_fitt_Error',bestFitt_error)


def main():
    createDirs()
    spkC_E1_all = np.load('./work/directGrating_Speed_SpikeCount_E1.npy')

    sim_param = pickle.load( open( './work/directGrating_Speed_parameter.p', "rb" ) )
    lvl_speed = sim_param['speed']
    lvl_spFrq = sim_param['spatFreq']
    n_steps = sim_param['n_degree']#15#20    
    lvl_speed = np.around(lvl_speed,2)
    print(np.shape(spkC_E1_all))
    print(sim_param)

    spkC_E1_all = spkC_E1_all[0]# -> only one Amplitude level at the moment
    spkC_E1_meanoR = np.mean(spkC_E1_all, axis=3)# mean over the stimuli repetitions
    
    plotSingleCell(spkC_E1_meanoR, lvl_speed, lvl_spFrq, 'Ex1')

    n_cells = np.shape(spkC_E1_meanoR)[3]
    mean_spkC = np.zeros((len(lvl_speed), len(lvl_spFrq)))
    mean_spkC_mean = np.zeros((len(lvl_speed), len(lvl_spFrq)))
    nCells_Speed_Freq = np.zeros((len(lvl_speed), len(lvl_spFrq)))
    
    spk_tf_sp = np.zeros((n_cells,len(lvl_speed), len(lvl_spFrq)))

    for i in range(n_cells):
        cellA = spkC_E1_meanoR[:,:,:,i]
        pref_dir = np.where(cellA == np.max(cellA))[2]
        mean_spkC += cellA[:,:,pref_dir[0]]
        mean_spkC_mean+= cellA[:,:,pref_dir[0]]/np.max(cellA[:,:,pref_dir[0]])
        pref_sp, pref_fr = np.where(cellA[:,:,pref_dir[0]] == np.max(cellA[:,:,pref_dir[0]]))
        nCells_Speed_Freq[pref_sp, pref_fr] +=1
        spk_tf_sp[i] = cellA[:,:,pref_dir[0]]

    print(np.shape(spk_tf_sp))

    ## create a boxplot for different spat. Freq (see Priebe et al. (2006))
    plt.figure()
    plt.boxplot(spk_tf_sp[:,:,0]) 
    plt.savefig('Output/speed/Ex1/box',dpi=300,bbox_inches='tight')


    mean_spkC /= n_cells
    mean_spkC_mean /= n_cells

    plt.figure(figsize=(6,8))
    plt.contourf(lvl_spFrq,np.linspace(0,len(lvl_speed)-1,len(lvl_speed)),mean_spkC,cmap='gray_r')#, levels=4)
    plt.ylabel('speed [Hz]')
    plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),lvl_speed)
    plt.xlabel('spat. freq [cycles/Img]')
    plt.xticks(lvl_spFrq)#(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)))
    plt.colorbar(label='Mean Firing rate [Hz]')
    plt.savefig('Output/speed/Ex1/contour',dpi=300,bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(8,6))
    plt.imshow( np.flip(mean_spkC,axis=0) ,cmap='gray_r')
    plt.ylabel('speed [Hz]')
    plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),np.flip(lvl_speed,axis=0))
    plt.xlabel('spat. freq [cycles/Img]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)), lvl_spFrq)
    plt.colorbar(label='Mean Firing rate [Hz]')
    plt.savefig('Output/speed/Ex1/imShow',dpi=300,bbox_inches='tight')
    plt.close()



    plt.figure(figsize=(6,8))
    plt.contourf(lvl_spFrq,np.linspace(0,len(lvl_speed)-1,len(lvl_speed)),mean_spkC_mean,cmap='gray_r')#, levels=4)
    plt.ylabel('speed [Hz]')
    plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),lvl_speed)
    plt.xlabel('spat. freq [cycles/Img]')
    plt.xticks(lvl_spFrq)#(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)))
    plt.colorbar(label='Normalized Firing rate [Hz]')
    plt.savefig('Output/speed/Ex1/contour_normResponse',dpi=300,bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(8,6))
    plt.imshow( np.flip(mean_spkC_mean,axis=0) ,cmap='gray_r')
    plt.ylabel('speed [Hz]')
    plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),np.flip(lvl_speed,axis=0))
    plt.xlabel('spat. freq [cycles/Img]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)), lvl_spFrq)
    plt.colorbar(label='Normalized Firing rate [Hz]')
    plt.savefig('Output/speed/Ex1/imShow_normResponse',dpi=300,bbox_inches='tight')
    plt.close()



    plt.figure(figsize=(8,6))
    plt.imshow( np.flip(nCells_Speed_Freq,axis=0) ,cmap='gray_r')
    plt.ylabel('speed [Hz]')
    plt.yticks(np.linspace(0, len(lvl_speed)-1, len(lvl_speed)),np.flip(lvl_speed,axis=0))
    plt.xlabel('spat. freq [cycles/Img]')
    plt.xticks(np.linspace(0,len(lvl_spFrq)-1, len(lvl_spFrq)), lvl_spFrq)
    plt.colorbar(label='Number of cells')
    plt.savefig('Output/speed/Ex1/inCells_Speed_Freq',dpi=300,bbox_inches='tight')
    plt.close()


    ####
    # Fit spatio-temporal tuning with a two-dimensional Gauss as described in Priebe et al. (2006)
    ###
    #fittSpatioTemporal(spk_tf_sp, lvl_speed, lvl_spFrq)


if __name__=="__main__":

    main()

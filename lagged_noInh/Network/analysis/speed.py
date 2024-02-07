import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


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
    for i in range(n_cells):
        cellA = spkC_E1_meanoR[:,:,:,i]
        pref_dir = np.where(cellA == np.max(cellA))[2]
        mean_spkC += cellA[:,:,pref_dir[0]]
        mean_spkC_mean+= cellA[:,:,pref_dir[0]]/np.max(cellA[:,:,pref_dir[0]])
        pref_sp, pref_fr = np.where(cellA[:,:,pref_dir[0]] == np.max(cellA[:,:,pref_dir[0]]))
        nCells_Speed_Freq[pref_sp, pref_fr] +=1

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

if __name__=="__main__":

    main()

import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm, trange

def plotFFT(strf_list):


    n_cells, n_steps, x_dim = np.shape(strf_list)
    for c in tqdm(range(n_cells),ncols=80):
        for i in range(n_steps):
            plt.figure()
            plt.subplot(211)
            data = strf_list[c,i,:]
            plt.plot(data ,'-o')
            plt.ylim(-15,15)
            plt.subplot(212)
            fft = np.fft.fft(data)/len(data)
            fft = fft[range(int(len(data)/2))]
            plt.plot(fft.real,'-o')
            plt.ylim(-2,2)
            plt.savefig('./Output/STRF/FFT/FFT_cell'+str(c)+'/STRF_FFT_'+str(i))
            plt.close()


def startAnalye2d():

    if not os.path.exists('Output/STRF/FFT_2D'):
            os.mkdir('Output/STRF/FFT_2D')
    strf_list = np.load('./work/STRF_list.npy')

    n_cells = len(strf_list)
    mse_list = np.zeros(n_cells)
    ampl_list = np.zeros(n_cells) # difference index for tf amplituds
    amplNorm_list = np.zeros(n_cells)

    for i in tqdm(range(n_cells),ncols=80):
        #strf_list[i] = strf_list[i]/np.max(np.abs(strf_list[i]))
        #strf_list[i] -= np.mean(strf_list[i])

        c1 = strf_list[i]
        if not os.path.exists('Output/STRF/FFT_2D/cell_%i'%i):
                os.mkdir('Output/STRF/FFT_2D/cell_%i'%i)

        

        fft_img = np.fft.fft2(c1)/len(c1.flatten())
        fft_img = np.fft.fftshift(fft_img)
        ampl_spec = np.sqrt( (fft_img.real)**2 + (fft_img.imag)**2 )
        #print(np.shape(ampl_spec))

        t_dim,x_dim = np.shape(ampl_spec)
        x_c = int(x_dim/2)
        t_c = int(t_dim/2)

        """
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(ampl_spec[t_c-20:t_c+20, x_c-20:x_c+20],aspect='auto')
        plt.subplot(1,2,2)
        plt.imshow(ampl_spec,aspect='auto')
        plt.savefig('Output/STRF/FFT_2D/cell_%i/Ampl_Spec'%i)
        plt.close()

        plt.figure(figsize=(8,10))
        plt.imshow(ampl_spec[t_c-20:t_c+20, x_c-20:x_c+20],aspect='auto')
        plt.yticks(np.linspace(2,38,4),np.linspace(-18,18,4))
        plt.xticks(np.linspace(2,38,4),np.linspace(-18,18,4))
        plt.axvline(20,color='gray')
        plt.axhline(20, color='gray')
        plt.ylabel('Temporal frequency', fontsize='12')
        plt.xlabel('Spatial frequency', fontsize='12')
        plt.savefig('Output/STRF/FFT_2D/cell_%i/Ampl_Spec_zoom'%i,bbox_inches='tight',dpi=300 )
        plt.close()

        plt.figure(figsize=(4,7))
        plt.subplot(2,1,1)
        plt.imshow(c1[50:], cmap=plt.get_cmap('RdBu',7),aspect='auto', vmin=-np.max(np.abs(c1)), vmax= np.max(np.abs(c1)))
        plt.xlabel('x [px]',fontsize='12')
        plt.ylabel('t [ms]',fontsize='12')
        plt.subplot(2,1,2)
        plt.imshow(ampl_spec[t_c-15:t_c+15+1, x_c-15:x_c+15+1],aspect='auto')
        plt.yticks(np.linspace(0,30,5),np.linspace(-15,15,5))
        plt.xticks(np.linspace(0,30,5),np.linspace(-15,15,5))
        plt.axvline(15,color='gray')
        plt.axhline(15, color='gray')
        plt.ylabel('Temporal frequency [Hz]', fontsize='12')
        plt.xlabel('Spatial frequency [Cycles/Img]', fontsize='12')
        plt.savefig('Output/STRF/FFT_2D/cell_%i/Ampl_Spec_zoom_2'%i,bbox_inches='tight',dpi=300 )
        plt.close()
        """
        # take only one half (two parts of TF and one SF)
        half_q = ampl_spec[:,:x_c]

        t_dim,x_dim = np.shape(half_q)
        x_c = int(x_dim/2)
        t_c = int(t_dim/2)

        """
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(half_q[t_c-20:t_c+20, x_c-20:x_c+20],aspect='auto')
        plt.subplot(1,2,2)
        plt.imshow(half_q,aspect='auto')
        plt.savefig('Output/STRF/FFT_2D/cell_%i/Ampl_Spec_half'%i)
        plt.close()
        """

        #print(np.where(half_q == np.max(np.abs(half_q)) ))
        idx_t,idx_x = np.where(half_q == np.max(np.abs(half_q)))

        t_c = int(len(half_q)/2)


        z = 5
        z_x = 18

        q1 = half_q[t_c-z:t_c+1,z_x:]
        q2 = half_q[t_c+1:t_c+z+2,z_x:]

        ampl_list[i] = np.abs(np.max(q1) - np.max(q2))/(np.max(q1) + np.max(q2)) # see Ohazawa et al. (1996)
        amplNorm_list[i] = np.abs(np.max(q1) - np.max(q2))/(np.max([np.max(q1),np.max(q2)]))
        #mse_list[i] = np.sqrt(np.mean((half_q[t_c-z:t_c+1,idx_x[0]] - np.flip(half_q[t_c:t_c+z+1,idx_x[0]]))**2))
        mse_list[i] = np.sqrt(np.mean((q1 - np.flip(q2,axis=0))**2))
    
        """
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(q1, vmin = 0, vmax = np.max(half_q))
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.imshow(np.flip(q2,axis=0), vmin = 0, vmax = np.max(half_q))
        plt.colorbar()
        plt.savefig('Output/STRF/FFT_2D/cell_%i/spec_half_half'%i)
        plt.close()

        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(half_q[:,idx_x[0]],label='max complete')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(half_q[:t_c+1,idx_x[0]],label='max complete_first half')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(half_q[t_c:,idx_x[0]],label='max complete_second half')
        plt.legend()
        plt.savefig('Output/STRF/FFT_2D/cell_%i/Ampl_Spec_half_max'%i)
        plt.close()

        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(half_q[t_c-z:t_c+z,idx_x[0]],label='max complete')
        plt.legend()
        plt.subplot(4,1,2)
        plt.plot(half_q[t_c-z:t_c+1,idx_x[0]],label='max complete_first half')
        plt.legend()
        plt.subplot(4,1,3)
        plt.plot(half_q[t_c:t_c+z+1,idx_x[0]],label='max complete_second half')
        plt.legend()
        plt.subplot(4,1,4)
        plt.plot(half_q[t_c-z:t_c+1,idx_x[0]],label='max complete_first half')
        plt.plot(np.flip(half_q[t_c:t_c+z+1,idx_x[0]]),label='max complete_second half/flipped')
        plt.legend()
        plt.savefig('./Output/STRF/FFT_2D/cell_%i/Ampl_Spec_half_zoom'%i)
        plt.close()
        """

    
    np.save('./work/rmseFFT2D',mse_list)
    np.save('./work/ampDiffFFT2D',ampl_list)
    np.save('./work/ampDiffFFT2D_Norm', amplNorm_list)

    plt.figure()
    plt.hist(mse_list, range=(0,0.3),bins=9)
    plt.ylabel('number of cells')
    plt.xlabel('RMSE')
    plt.xlim(0,0.3)
    plt.ylim(0,120)
    plt.savefig('./Output/STRF/FFT_2D/MSE_hist',bbox_inches='tight',dpi=300)

    plt.figure()
    plt.hist(amplNorm_list,bins=9, range=(0,0.7))
    plt.ylabel('number of cells')
    plt.xlabel('TFD')
    plt.xlim(0,0.7)
    plt.ylim(0,120)
    plt.savefig('./Output/STRF/FFT_2D/AMPDIFFnorm_hist',bbox_inches='tight',dpi=300)


    plt.figure()
    plt.hist(ampl_list,bins=9, range=(0,0.45))
    plt.ylabel('number of cells')
    plt.xlabel('TFD')
    plt.xlim(0,0.45)
    plt.ylim(0,140)
    plt.savefig('./Output/STRF/FFT_2D/AMPDIFF_hist',bbox_inches='tight',dpi=300)

    n_bins=10
    bins = np.linspace(0,0.3,n_bins)
    print(bins)
    for b in tqdm(range(1,len(bins)),ncols=80):
        if not os.path.exists('Output/STRF/FFT_2D/bin_%i'%b):
                os.mkdir('Output/STRF/FFT_2D/bin_%i'%b)
        #print(bins[b-1], bins[b])
        idx =np.where( (mse_list>=bins[b-1]) & (mse_list<bins[b]))[0]

        for i in idx:
            plt.figure()
            plt.imshow(strf_list[i,100:], cmap=plt.get_cmap('RdBu',7),aspect='auto', vmin=-np.max(np.abs(strf_list[i])), vmax= np.max(np.abs(strf_list[i])) )
            plt.xlabel('x',fontsize='12')
            plt.ylabel('t',fontsize='12')
            plt.savefig('./Output/STRF/FFT_2D/bin_%i/cell_%i'%(b,i),bbox_inches='tight')
            plt.close()


    n_bins=10
    bins = np.linspace(0,0.7,n_bins)
    print(bins)
    for b in tqdm(range(1,len(bins)),ncols=80):
        if not os.path.exists('Output/STRF/FFT_2D/norm_bin_%i'%b):
                os.mkdir('Output/STRF/FFT_2D/norm_bin_%i'%b)
        #print(bins[b-1], bins[b])
        idx =np.where( (amplNorm_list>=bins[b-1]) & (amplNorm_list<bins[b]))[0]

        for i in idx:
            plt.figure()
            plt.imshow(strf_list[i,100:], cmap=plt.get_cmap('RdBu',7),aspect='auto', vmin=-np.max(np.abs(strf_list[i])), vmax= np.max(np.abs(strf_list[i])) )
            plt.xlabel('x',fontsize='12')
            plt.ylabel('t',fontsize='12')
            plt.savefig('./Output/STRF/FFT_2D/norm_bin_%i/cell_%i'%(b,i),bbox_inches='tight')
            plt.close()

    return()

def startAnalyze():
    

    if not os.path.exists('Output/STRF/FFT'):
            os.mkdir('Output/STRF/FFT')


    strf_list = np.load('./work/STRF_list.npy')
    strf_list = strf_list[:,100:,]
    print(np.shape(strf_list))

    n_cells, n_steps, x_dim = np.shape(strf_list)


    for i in range(n_cells):
        if not os.path.exists('Output/STRF/FFT/FFT_cell'+str(i)+'/'):
            os.mkdir('Output/STRF/FFT/FFT_cell'+str(i)+'/')


    #print(np.fft.fft(strf_list[6,27,:]).real)
    #print(np.fft.fftfreq(n=len(strf_list[6,27,:])))
    
    fft_all = np.zeros((n_cells,n_steps,8))
    fft_div = np.zeros((n_cells,n_steps-1))
    max_amp = np.zeros((n_cells,n_steps))
    for c in tqdm(range(n_cells),ncols=80):
        for t in range(n_steps):
            data = strf_list[c,t,:]
            fft = np.fft.fft(data)/len(data)
            fft = fft[range(int(len(data)/2))]
            fft = fft.real
            fft = fft[:8]
            fft_all[c,t,:] = fft
            max_amp[c,t] = fft[np.argmax((fft))]
            if t > 0:
                fft_div[c,t-1] = np.mean(np.abs(fft_all[c,t,:] - fft_all[c,t-1,:]))
 
    fft_freqMaxAmp = np.zeros((n_cells,n_steps))    
    for c in range(n_cells):
        idx_max = np.where(fft_all[c] == np.max(fft_all[c]))
        fft_freqMaxAmp[c] = fft_all[c,:,idx_max[1]]
        

    plt.figure(figsize=(9,9))
    for c in range(9):
        plt.subplot(3,3,c+1)
        plt.plot(max_amp[c],'o-')
        plt.xlabel('Timestep')
        plt.ylabel('max Ampl.')
    plt.savefig('Output/STRF/FFT/maxAmpl')
    plt.close()

    plt.figure(figsize=(9,9))
    for c in range(9):
        plt.subplot(3,3,c+1)
        plt.plot(fft_freqMaxAmp[c],'o-')
        plt.xlabel('Timestep')
        plt.ylabel('max FreqAmpl.')
    plt.savefig('Output/STRF/FFT/maxFreqAmpl')
    plt.close()


    plt.figure(figsize=(9,9))
    for c in range(9):
        plt.subplot(3,3,c+1)
        plt.plot(fft_div[c],'o-')
        plt.xlabel('Timestep')
        plt.ylabel('diff')
    plt.savefig('Output/STRF/FFT/diff')
    plt.close()

    plt.figure(figsize=(12,12))
    for c in range(9):
        plt.subplot(3,3,c+1)
        plt.imshow(fft_all[c],aspect='auto')
    plt.savefig('Output/STRF/FFT/FFT_allC')
    plt.close()

    #plotFFT(strf_list)

#------------------------------------------------------------------------------
if __name__=="__main__":
    #startAnalyze()
    startAnalye2d()

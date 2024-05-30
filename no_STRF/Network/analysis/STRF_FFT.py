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

        c1 = strf_list[i]
        fft_img = np.fft.fft2(c1)/len(c1.flatten())
        fft_img = np.fft.fftshift(fft_img)
        ampl_spec = np.sqrt( (fft_img.real)**2 + (fft_img.imag)**2 )
        #print(np.shape(ampl_spec))

        t_dim,x_dim = np.shape(ampl_spec)
        x_c = int(x_dim/2)
        t_c = int(t_dim/2)


        # take only one half (two parts of TF and one SF)
        half_q = ampl_spec[:,:x_c]

        t_dim,x_dim = np.shape(half_q)
        x_c = int(x_dim/2)
        t_c = int(t_dim/2)



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

#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalye2d()

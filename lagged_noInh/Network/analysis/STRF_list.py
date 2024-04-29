import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


def calcStatistics():

    strf_list = np.load('./work/STRF_list.npy')

    n_cells, n_timepoints, n_pixels = np.shape(strf_list)

    ## get the max amplitude time point
    max_t = np.zeros(n_cells)
    duration = np.zeros(n_cells)
    for c in range(n_cells):
        roll=False
        idx_max = np.where(np.abs(strf_list[c]) == np.max(np.abs(strf_list[c])))
        if len(idx_max[0]) > 1:
            max_t[c] = n_timepoints - idx_max[0][0]
            curve = strf_list[c,:, idx_max[1][0]]
        else:
            max_t[c] = n_timepoints - idx_max[0]
            curve = strf_list[c,:, idx_max[1]]
            curve = curve[0]

        #curve = curve[2:]
        delta = [np.abs(curve[i+2] - curve[i]) for i in range(len(curve)-2)]
        end_t = np.where(delta > (np.mean(delta)+(0.5*np.mean(delta))) )[0]
        #duration[c] = n_timepoints- end_t[0]

        ## see DeAngelis1993
        curve = np.flip(curve)
        ## if the response curve is at t=0 not at 0, roll by a fixed value?
        roll_t = 50
        if np.abs(curve[0]) > 1:
            curve = np.roll(curve,roll_t)
            roll=True
        ## get Hilber Transformation
        h_curve = np.abs(hilbert(curve))
        ## calculate the envelop curve
        e_t = np.sqrt(curve**2 + h_curve**2)

        ## get the peak envelop value
        if roll==True:
            peak_et = np.max(e_t[roll_t+10:])
        else:
            peak_et = np.max(e_t)
        peak_idx = np.where(e_t == peak_et)[0]

        
        ## calcualte the width by looking, where is the envelop curve <1/e*peak value
        thresh = peak_et/2.718271
        ## look where the curve is above the threshold
        idx_et = np.where(e_t > thresh)[0]
        duration[c] = idx_et[-1] - idx_et[0]

        if c == 3:
            plt.figure()
            plt.plot(curve, label='STRF curve')
            plt.plot(h_curve, label='Hilbert curve')
            plt.plot(e_t, label='Envelope curve')
            plt.legend()
            plt.savefig('./Output/STRF/STRF_E1/curve_%i.png'%(c),bbox_inches='tight',dpi=300)

    mean_t = np.mean(max_t)

    plt.figure()
    plt.hist(max_t,15,range =(0,180) , label='Mean: %.2f'%(np.mean(max_t)))
    plt.xlabel(r'$T_{peak}$ [ms]', fontsize=15)
    plt.ylabel('# of cells', fontsize=15)
    plt.xlim(0,180)
    plt.vlines(mean_t,0,80,'black')
    plt.legend()
    plt.savefig('./Output/STRF/STRF_E1/peak_time_hist.png',bbox_inches='tight',dpi=300)
    plt.close()

    mean_d = np.mean(duration)
    plt.figure()
    plt.hist(duration,15,range =(0,500), label='Mean: %.2f'%(np.mean(duration)))
    plt.xlabel('duration time [ms]', fontsize=15)
    plt.ylabel('# of cells', fontsize=15)
    plt.xlim(0,500)
    plt.vlines(mean_d,0,170,'black')
    plt.legend()    
    plt.savefig('./Output/STRF/STRF_E1/duration_time_hist.png',bbox_inches='tight',dpi=300)
    plt.close()

if __name__=="__main__":
    calcStatistics()

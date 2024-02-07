import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def startAnalysis():
    frLGN = np.load('./work/frLGNSpkes.npy')
    frExc = np.load('./work/frSingleSpkes.npy')
    frExc = frExc.item()
    frLGN = frLGN.item()
    n_Cels = len(frExc)
    n_LGNC = len(frLGN)

    
    isi =[]
    isiList=[]
    cv = np.zeros(n_Cels)
    for cell_i in range(n_Cels):
        isiC = np.diff(frExc[cell_i])
        cv[cell_i] = np.std(isiC,ddof=1)/np.mean(isiC) # coefficent variation of the ISI after Destexhe et al. 2001
        isi.append(isiC)
        isiList = np.concatenate((isiList,isiC))
      
    isi_LGN =[]
    isiList_LGN=[]
    cv_LGN = np.zeros(n_LGNC)
    for cell_i in range(n_LGNC):
        isiC = np.diff(frLGN[cell_i])
        cv_LGN[cell_i] = np.std(isiC,ddof=1)/np.mean(isiC) # coefficent variation of the ISI after Destexhe et al. 2001
        isi_LGN.append(isiC)
        isiList_LGN = np.concatenate((isiList_LGN,isiC))
      
    bins=30

    plt.figure()
    plt.hist(isiList,bins)
    plt.xlabel('ISI [ms]')
    plt.ylabel('# spike paris')
    plt.savefig('Output/ISI.jpg',dpi=300)

    plt.figure()
    plt.hist(isiList_LGN,bins)
    plt.xlabel('ISI [ms]')
    plt.ylabel('# spike paris')
    plt.savefig('Output/ISI_LGN.jpg',dpi=300)

    plt.figure()
    plt.hist(isiList,bins,log=True)
    plt.xlabel('ISI [ms]')
    plt.ylabel('# spike paris')
    plt.savefig('Output/ISI_logy.jpg',dpi=300)

    plt.figure()
    plt.hist(isiList,bins=bins,log=True)
    plt.xlabel('ISI [ms]')
    plt.ylabel('# spike paris')
    plt.gca().set_xscale("log")
    plt.savefig('Output/ISI_logX.jpg',dpi=300)

    plt.figure()
    plt.hist(isiList,bins=np.logspace(0.1,3.3,bins))
    plt.xlabel('ISI [ms]')
    plt.ylabel('# spike paris')
    plt.gca().set_xscale("log")
    plt.savefig('Output/ISI_logX_logBin.jpg',dpi=300)

    plt.figure()
    plt.hist(isiList,bins=np.logspace(0.1,3.3,bins),log=True)
    plt.xlabel('ISI [ms]')
    plt.ylabel('# spike paris')
    plt.gca().set_xscale("log")
    plt.savefig('Output/ISI_logXY.jpg',dpi=300)

    plt.figure()
    plt.hist(cv,30)
    plt.ylabel('# cells')
    plt.xlabel('CV per cell')
    plt.savefig('Output/ISI_CV_hist.jpg',dpi=300)

    plt.figure()
    plt.hist(cv_LGN,30)
    plt.ylabel('# cells')
    plt.xlabel('CV per cell')
    plt.savefig('Output/ISI_CV_LGN_hist.jpg',dpi=300)

#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalysis()

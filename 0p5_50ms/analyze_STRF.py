import os

def createFolders():
    if not os.path.exists('Output'):
        os.mkdir('Output')
    if not os.path.exists('work'):
        os.mkdir('work')

def analyzeSTRF(lagged_LGN = True):

    print('Python script to analyze the STRFs')

    ## create folder for Output figures and work data
    createFolders()

    ## create some lagged LGN cells, if necessary
    if lagged_LGN:
        if not os.path.isfile('./work/LGN_OFF_delay.npy'):
           os.system('python Network/net/createLaggedLGN.py')

    ## show some random bars 
    os.system('python Network/net/STRF_bar.py')

    ## create the STRF Plots
    os.system('python Network/analysis/STRF.py')

    ## create a gif for the STRFs
    os.system('python Network/analysis/createSTRF_movie.py')

    ## FFT analysis
    os.system('python Network/analysis/STRF_FFT.py')


if __name__ == "__main__":
    analyzeSTRF()

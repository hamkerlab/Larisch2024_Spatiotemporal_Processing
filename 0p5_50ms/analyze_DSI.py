import os

def createFolders():
    if not os.path.exists('Output'):
        os.mkdir('Output')
    if not os.path.exists('work'):
        os.mkdir('work')

def analyzeDSI(lagged_LGN = True):

    print('Python script to analyze the direction selectivity')

   ## create folder for Output figures and work data
    createFolders()

    ## create some lagged LGN cells, if necessary
    if lagged_LGN:
        if not os.path.isfile('./work/LGN_OFF_delay.npy'):
           os.system('python Network/net/createLaggedLGN.py')

    ## show some static sinus gratings
    print('Start with showing static sinus gratings')
    print('----------------------------------------')
    #os.system('python Network/net/tuningCurves_sinus.py')

    ## show some moving sinus gratings
    print('Start with showing moving sinus gratings')
    print('----------------------------------------')
    #os.system('python Network/net/direct_SinusGrating.py')

    os.system('python Network/analysis/direct_SinusGrating_TC.py')

    ## further STRF analyses, which are necessary for the analyses of the direction selecitivty
    print('Some more STRF analysis which are needed for the analyses of the direction selectivity')
    print('--------------------------------------------------------------------------------------')
    os.system('python Network/analysis/STRF_more.py')
    os.system('python Network/analysis/STRF_FFT.py')

    ## evaluate direction selectivity
    print('Start with the analysis of the direction selectivity')
    print('----------------------------------------------------')
    os.system('python Network/analysis/DSI_currents.py')
    os.system('python Network/analysis/spike_to_current.py')

if __name__ == "__main__":
    analyzeDSI()

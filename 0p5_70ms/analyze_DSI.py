import os
def analyzeDSI():

    print('Python script to analyze the direction selectivity')

   ## create folder for Output figures and work data
    createFolders()

    ## create some lagged LGN cells, if necessary
    if lagged_LGN:
        if not os.path.isfile('./work/LGN_OFF_delay.npy'):
           os.system('python Network/net/createLaggedLGN.py')

    ## show some sinus gratings
    os.system('python Network/net/direct_SinusGrating.py')




if __name__ == "__main__":
    analyzeDSI()

from ANNarchy import *
from net_EB_LGN_DoG import *
import numpy as np

def main():
    perc_cells = 0.5 # 50 percent of LGN cells are lagged
    delay_cells = 50#50 # 50ms is the lagg

    print('Create delay matrix for lagged LGN cells')
    

    compile()

    ### lagged LGN cells over syn-Delays
    # add some extra delays to implement "lagged" LGN-Cells -> additional delay depends on t_delay !
    n_LGNCells = int(n_LGN/2)
    lagged_cells = np.random.choice(n_LGNCells,int((n_LGNCells)*perc_cells),replace=False ) # chose random some "lagged" LGN cells
    d_old_ON = np.asarray(projInput_LGN_ON.delay) # get the old delays
    d_old_ON[lagged_cells] = d_old_ON[lagged_cells]+delay_cells # add a delay of n ms
    np.save('./work/LGN_ON_delay',d_old_ON, allow_pickle=False)
    

    lagged_cells = np.random.choice(n_LGNCells,int((n_LGNCells)*perc_cells),replace=False ) # chose random some "lagged" LGN cells
    d_old_OFF = np.asarray(projInput_LGN_OFF.delay) # get the old delays
    d_old_OFF[lagged_cells] = d_old_OFF[lagged_cells]+delay_cells # add a delay of n ms
    np.save('./work/LGN_OFF_delay',d_old_OFF, allow_pickle=False)

#------------------------------------------------------------------------------
if __name__=="__main__":

    main()

# Larisch2024_Spatiotemporal_Processing

Repository containing the relevant python scripts for the Larisch, R. and Hamker (2024) publication.


#### Dependencies

* Python >v3.8
* Numpy >= v1.11.0
* Matplotlib >= v1.5.1
* ANNarchy >= v4.7.5 (actual master release can be found [here](https://bitbucket.org/annarchy/annarchy/downloads/?tab=branches) and an installation guide is provided in the [documentation](https://annarchy.readthedocs.io/en/stable/intro/Installation.html) )

## Model variants

The different directories contain the evaluation scripts to the different model variants.
They differ as follow

|    Directory            | with RGC surround delay | $\%$ lagged cells | $ms$ of lag | inhibition | notes
|-------------------------|-------------------------|-------------------|-------------|------------|:-------------------------------------------------------:|
| *0p3_50ms*              |           Yes           |         30        |     50      |    Yes     |                                                         |
| *0p5_30ms*              |           Yes           |         50        |     30      |    Yes     |                                                         |
| *0p5_50ms*              |           Yes           |         50        |     50      |    Yes     |                                                         |
| *0p5_70ms*              |           Yes           |         50        |     70      |    Yes     |                                                         |   
| *0p7_50ms*              |           Yes           |         70        |     50      |    Yes     |                                                         |
| *lagged_noInh*          |           Yes           |         50        |     50      |    No      |                                                         |       
| *nolagged_Inh*          |           Yes           |         0         |     -       |    Yes     |                                                         |       
| *nolagged_noInh*        |           Yes           |         0         |     -       |    No      |                                                         |
| *no_STRF*               |           No            |         -         |     -       |    Yes     |                                                         |
| *shuffle_FBInhib*       |           Yes           |         50        |     50      |    Yes     | Shuffle all weights, related to feedback inhibition     |
| *shuffle_FWInhib*       |           Yes           |         50        |     50      |    Yes     | Shuffle all weights, related to feedforward inhibition  |
| *shuffleInhib_complete* |           Yes           |         50        |     50      |    Yes     | Shuffle all weights, related to inhibition              | 

## Script order to obtain results

Different python - scripts are provided to perform the different analyzes shown in the manuscript for the single model variants.
Before starting the evaluations, get sure to copy all the required weight matrices to the ```Input_network``` directory of each model variant.

As the evaluation is build around the *0p5_50ms* model variant (which can be seen as the "default model"), not all model variants contain the same evaluation script.
Due to this, to get the complete evaluation, please follow the current order:

1. Run the ``` python analyze_STRF.py ``` script in each of the model variant directories to measure the spatiotemporal receptive fields (STRFs) (PLEASE NOTE: Calculating the STRFs can lead to heavy memory load).
2. After all scripts are run successful, run the ``` python analyze_DSI.py ``` script in the *0p5_50ms* directory and in the *no_STRF* directory
3. Copy the ``` dsi_kim_Cells_TC.npy``` file to the *basis* directory and rename it to ```dsi_kim_Cells_TC_0p5_50ms.npy```
4. Run in all other directories the ``` python analyze_DSI.py ``` script


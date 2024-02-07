
# Larisch2024_Spatiotemporal_Processing

Repository containing the relevant python scripts for the Larisch, R. and Hamker (2024) publication.


#### Dependencies

* Python >v3.8
* Numpy >= v1.11.0
* Matplotlib >= v1.5.1
* ANNarchy >= v4.7.5 (actual master release can be found [here](https://bitbucket.org/annarchy/annarchy/downloads/?tab=branches) and an installation guide is provided in the [documentation](https://annarchy.readthedocs.io/en/stable/intro/Installation.html) )



# Script order to obtain results

Different python - scripts are provided to perform the different analyzes shown in the manuscript for the single model variants.
Before starting the evaluations, get sure to copy all the required weight matrices to the ```Input_network``` directory.

## Spatiotemporal receptive fields

To measure the spatiotemporal receptive fields (STRFs), type:

```
python analyze_STRF.py
```

## Direction selectivity

To measure the direction selectivity and related evaluations, type:

```
python analyze_DSI.py
```

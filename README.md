
# nested_ulens_parallax

modelling simple and parallax microlensing event lightcurves with nested sampling

## Summary

This code was used to conduct the modelling of microlensing events from the VISTA Variables in the Via Lactea (VVV) survey, described in ([Kaczmarek et al. (2022)](https://arxiv.org/abs/2205.07922)). It uses the nested sampling implementation from `dynesty` ([Speagle 2020](https://arxiv.org/abs/1904.02180)) to automatically characterise the degenerate, multimodal solutions in the parallax microlensing model.
<!---
add link to the paper, when available. Add license
--->
It is freely available for use. If you use this code, please cite ([Kaczmarek et al. (2022)](https://arxiv.org/abs/2205.07922)) and [Speagle (2020)](https://arxiv.org/abs/1904.02180).

## Data

The folder `data/` includes all data necessary for the modelling of the 21 best strong parallax signal candidates selected in Section 4 of the paper. It comprises of complete VVV photometry (`lightcurve_data.csv`) and coordinates and parameters of an initial simple lightcurve fit from [Husseiniova et al. (2021)](https://arxiv.org/abs/2106.15617) (`pspl_event_parameters.csv`) for all of them, coded by source ID.

In addition to those files, we are including a mock datafile containing RST-like astrometry and photometry as `0324_mockdata_RST.csv`.

## Photometry

The folders `parallax/` and `simple/` contain codes used for fitting parallax and simple (linear motion) microlensing lightcurve models respectively with nested sampling. Each of them is composed of:
- a `config.yaml` file containing settings for priors of model parameters and for the `dynesty` sampler
- a `utils.py` file containing various functions used in modelling, saving and plotting results
- a `fit_..._model.py` file fitting the model for a specified source ID, given as the argument `source_id`

For a list of source IDs stored in `sourceid_list.txt`, one can start the `run_analysis.sh` script for a simple test run of fitting both models to all events.

For each source ID and each model, the output will consist of a dynesty .pkl file containing the samples, a cornerplot for fitted parameters, and a plot showing original datapoints (blue) and an ensemble of lightcurves generated with 100 randomly chosen samples (grey). The output is stored in the `parallax/results/` and `simple/results/` folders and coded by source ID.

## Astrometry

The astrometry/ folder contains the code used in Section 5 of the paper. It includes simultaneous modelling of photometric (`mags`) and 2D astrometric (`racs`, `decs`) data, and introduces additional astrometric parameters to fit for.

This part is structured similarly to the previous two. The main difference is that coordinates, reference time and datafile path are fixed for the single mock event we have simulated, so there is no `source_id` argument. Instead, `n_cores` (number of cores to be used in parallel) is taken as an argument, and parallelisation is included in the body of the code.

Contrary to the photometry part, where we introduced parallelisation by running the `fit_parallax_model.py`/`fit_simple_model.py` contents in separate processes for each event, here computational challenges make it necessary to use a number of cores for modelling a single event.

## Dependencies

Below we are listing the dependencies of the code. The version number only indicates the version used in our setup - we have tested the code works well on them, but earlier or later versions could work equally well.

- `astropy==5.0.1`
- `corner==2.2.1`
- `dynesty==1.1`
- `matplotlib==3.5.1`
- `numba==0.53.1`
- `numpy==1.22.2`
- `pandas==1.4.1`
- `PyYAML==6.0`

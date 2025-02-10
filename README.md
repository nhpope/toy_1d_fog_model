This project is a simple 1d numerical model designed to simulate fog formed through cloud base lowering.
The purpose is to create a simple tool allowing experimentation to attempt to recreate the bimodal DSD that is often observed in fog

There are three python files
1. model_1d_namelist.py contains variables that are meant to be modified by the user for different runs
2. model_1d_functions.py contains functions for the model
3. model_1d_run.py is the main python script to run the model

The model uses bin microphysics with collision coalescence code translated from FORTRAN code originally written by Andreas Bott at the University of Bonn
Radiation is not effected by microphysics and is applied as an exponential decay specified in the namelist
Turbulent transport is parameterized with k theory
Droplet settling is differentially applied between the droplet size bins
Condensation is applied in proportion to the surface area of each microphysics bin. There is no system to mimic aerosol activation after initialization (a potential weakness)
Evaporation is applied differentially across bins and allows for movement to the next smaller bin

The model produces a netcdf file as output

As of February 10th, 2025 at 2:20 PM PST, the parameters in the namelist have been tested for stability.

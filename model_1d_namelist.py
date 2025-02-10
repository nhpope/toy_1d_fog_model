# List of input variables for the model
# These are the variables that are intended to be fiddled with
# Variables in the functions or the run files can be modified if necessary, but they are intended to be fixed

# code written for python 3.11; needs math, numpy, netcdf4, scipy, and copy packages
# collision code was originally written in FORTRAN by Andreas Bott at the University of Bonn

# Model Parameters
top_height = 240 # height of the top of the model in meters
num_levs = 240 # number of model levels
dt = 4.
substeps = 100 # number of flux timesteps per collision timestep
t_max = 7200 # simulation end time in seconds
out_interval = 60 # interval in seconds for data output

# Initial Temperature and Moisture Profile Parameters
S_Temp = 288 # Temperature at the surface in Kelvin
cbh = 120 # Initial height of the cloud base in meters

# Radiation Parameters
rad_total = 50 # Radiative cooling rate in W/m^2
rad_decay = 10 # Radiative cooling profile decay parameter in meters

# Microphysics parameters
shape = 8. # shape parameter of the initial gamma PDF DSD
n_param = 40. # DNC in /cm^3 when LWC is 0.1 g/m^3
nbins = 96 # Number of droplet size bins
dsd_min = .6 # minimum radius of the smallest droplet size bin
dsd_max = 100. # maximum radius of the largest droplet size bin
gmin = 1e-12

# Turbulence
k = 0.4
lam = 40.

# output file name
outfile = 'model_1d_output_2' # automatically ads .nc extension
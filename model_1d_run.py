# import packages
import numpy as np
import netCDF4 as nc

from scipy.optimize import fsolve
from scipy.stats import gamma

# import functions and variables from other model files
from model_1d_namelist import *
from model_1d_functions import *

# Initialize model grid
num_levs = int(num_levs)
mid_h = np.linspace(0.5 * top_height / num_levs, top_height - (0.5 * top_height / num_levs), num_levs)
z_fluxes = np.linspace(top_height / num_levs, top_height - (top_height / num_levs), num_levs - 1)
dz = mid_h[1] - mid_h[0]
dtt = dt / substeps

# Initialize temperature, moisture, and radiation profiles
T_init = np.zeros(np.shape(mid_h))
qv_init = np.zeros(np.shape(mid_h))
ql_init = np.zeros(np.shape(mid_h))
Theta_init = np.zeros(np.shape(mid_h))
rad_prof = np.zeros(np.shape(mid_h))

# Calculate dew point and vapor pressure
Td = S_Temp - 9.8 * (cbh/1000)
Tdc = Td - 273.15
Tdc1 = Tdc+1
es = 6.11 * 10 ** (7.5 * Tdc / (237.7 + Tdc))
es1 = 6.11 * 10 ** (7.5 * Tdc1 / (237.7 + Tdc1))

# Calculate initial mixing ratio in g/kg
qv_sfc = 621.97 * es / (1000 - es)
qv1 = 621.97 * es1 / (1000 - es1)

Theta_288 = np.linspace(S_Temp-9.8*0.0005, S_Temp-(9.8*(top_height - 0.5)/1000), top_height)

# Assign initial values of T, qv, and ql at all model levels
for i in range(np.shape(mid_h)[0]):
    [T,qv,ql, Theta] = Inits(S_Temp, cbh, qv_sfc, Td, Theta_288, mid_h[i])
    T_init[i] = T
    qv_init[i] = qv
    ql_init[i] = ql
    Theta_init[i] = Theta
    rad_prof[i] = radiation(rad_decay, rad_total, top_height, mid_h[i]) # radiation profile remains constant throughout the simulation

# set low values of CWC to 0 for simplicity
ql_init[ql_init < 1e-4] = 0

# initialize cloud water

# calculate cloud water concentration
dnc_init = np.zeros(np.shape(ql_init))
dnc_init[ql_init > 0] = 10 ** (((np.log10(ql_init[ql_init > 0]) + 1) / 2) + np.log10(n_param))

scal = nbins / np.log2((dsd_max ** 3) / (dsd_min ** 3))

dlnr = np.log(2) / (3.0 * scal)
ax = 2 ** (1.0 / scal)
    
# Mass and radius grid
emin = (4/3) * np.pi * (dsd_min * 1e-4)**3 * 0.001 * 1e6
# emin = 1e-9
r = np.zeros(nbins + int(scal + 1))
rb = np.zeros(nbins + int(scal + 2))
e = np.zeros(nbins + int(scal + 1))
eb = np.zeros(nbins + int(scal + 2))
rb[0] = dsd_min
eb[0] = emin
e[0] = emin * 0.5 * (ax + 1.0)
r[0] = 1000.0 * np.exp(np.log(3.0 * e[0] / (4.0 * np.pi)) / 3.0)
eb[1] = emin * ax
rb[1] = 1000.0 * np.exp(np.log(3.0 * eb[1] / (4.0 * np.pi)) / 3.0)
for i in range(1, nbins + int(scal + 1)):
    e[i] = ax * e[i - 1]
    r[i] = 1000.0 * np.exp(np.log(3.0 * e[i] / (4.0 * np.pi)) / 3.0)
    eb[i+1] = ax * eb[i]
    rb[i+1] = 1000.0 * np.exp(np.log(3.0 * eb[i+1] / (4.0 * np.pi)) / 3.0)

e_widths = np.diff(eb)
r_widths = np.diff(rb)

# initialize cloud water grid
dNdR_init = np.zeros((np.shape(ql_init)[0], int(nbins)))

for i in range(np.shape(dNdR_init)[0]):
    if dnc_init[i] > 0.:
        dNdR_init[i,:] = scale_param_gamma(r[:nbins], r_widths[:nbins], dnc_init[i], ql_init[i], shape, rho_water=1e6)

# calculate collision and settling parameters
ec = calculate_efficiency(r)
c, ima = courant(e, dlnr)
winf = fallg(r) # note that this gives fall speed in cm/s
ck = trkern(r, winf)

# Convert to liquid water content for each bin
LWC_init = dNdR_init* dnc_init[:,np.newaxis] * r_widths[np.newaxis,:nbins] * e[np.newaxis,:nbins] * 1e3

# Set initial values of variables for model run
T_m = T_init
Theta_m = Theta_init # have not written the code to save this in the netcdf file
qv_m = qv_init
ql_m = LWC_init
dep_m = 0 # have not written the code to save this in the netcdf file

# calculate number of timesteps and model outputs
numsteps = int(t_max / dt)
num_outs = int(t_max / out_interval) + 1

# initialize output arrays
T_out = np.zeros((np.shape(T_m)[0], num_outs))
Theta_out = np.zeros((np.shape(T_m)[0], num_outs))
qv_out = np.zeros((np.shape(T_m)[0], num_outs))
dep_out = np.zeros(num_outs)
ql_out = np.zeros((np.shape(LWC_init)[0], np.shape(LWC_init)[1], num_outs))
# print(scal)

# model loop
for i in range(numsteps + 1):

    # check if it is an output step
    if i%int(out_interval/dt) == 0:
        T_out[:,int(i/int(out_interval/dt))] = T_m
        Theta_out[:,int(i/int(out_interval/dt))] = Theta_m
        qv_out[:,int(i/int(out_interval/dt))] = qv_m
        ql_out[:,:,int(i/int(out_interval/dt))] = ql_m
        dep_out[int(i/int(out_interval/dt))] = dep_m
        
        # print the number of seconds that have elapsed in the model
        print(i * dt)
    
    # last step is just for saving to outputs
    if i < numsteps:
        
        # transport loop running on the substep
        for j in range(substeps):
            T_m, Theta_m, qv_m, ql_m, dep = model_subtimestep(T_m, Theta_m, qv_m, ql_m, winf, dtt, dz, Theta_288, S_Temp, z_fluxes, lam, k, rad_prof, nbins, r, rb, r_widths, e)
            ql_m[ql_m < 1e-12] = 0.
            dep_m += np.nansum(dep)

            # stability check
            if np.isnan(ql_m).any() or np.sum(ql_m[-1,:]) < 0.05:
                print(i, j)
                break
        
        # stability check
        if np.isnan(ql_m).any() or np.sum(ql_m[-1,:]) < 0.05:
            break
        
        # collision step
        ql_m, dep = collision_tstep(ql_m, ima, e, r, ck, dt, c, gmin, dlnr, r_widths, nbins)
        dep_m += dep
        # print(i, T_m[-1], np.sum(ql_m[-1,:]), qv_m[-1])

    
# write output to netcdf file (netcdf4 I think)
newfile = outfile + '.nc'

ncfile = nc.Dataset(newfile,mode='w')

nt = np.shape(ql_out)[2]
nr = np.shape(ql_out)[1]
nz = np.shape(ql_out)[0]

t_dim = ncfile.createDimension('time', nt)
r_dim = ncfile.createDimension('r_bin', nr)
z_dim = ncfile.createDimension('height', nz)

v_t = ncfile.createVariable('time', np.float32, ('time',))
v_t.units = 'seconds'
v_t.long_name = 'Time in Seconds'
v_t[:] = np.arange(0, nt * out_interval, out_interval)

v_r = ncfile.createVariable('r_bin', np.float32, ('r_bin',))
v_r.units = 'microns'
v_r.long_name = 'droplet bin geometric mean radii'
v_r[:] = r[:nr]

v_z = ncfile.createVariable('height', np.float32, ('height',))
v_z.units = 'meters'
v_z.long_name = 'model level height'
v_z[:] = mid_h

v_Temp = ncfile.createVariable('temp', np.float32, ('height', 'time'))
v_Temp.units = 'K'
v_Temp.long_name = 'Temperature'
v_Temp[:] = T_out

v_qv = ncfile.createVariable('qv', np.float32, ('height', 'time'))
v_qv.units = 'g/kg'
v_qv.long_name = 'Water vapor mixing ration'
v_qv[:] = qv_out

v_ql = ncfile.createVariable('ql', np.float32, ('height', 'r_bin', 'time'))
v_ql.units = 'g/kg'
v_ql.long_name = 'bin liquid water content'
v_ql[:] = ql_out

ncfile.title='1D_Model_Output'
ncfile.subtitle="Output of the 1-D Model"

ncfile.close()

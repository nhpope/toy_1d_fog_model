# import packages
import math
import numpy as np
import copy

from scipy.optimize import fsolve
from scipy.stats import gamma

def Inits(S_Temp, CBH, qv_sfc, Td, Theta_288, h):
    """ Creates initial values of T, qv, and ql at height h
    """
    if h<=CBH:
        T = S_Temp - 9.8 * (h/1000)
        Theta = 288
        qv = qv_sfc
        ql = 0
    else:
        T = Td - 6.5 * ((h-CBH)/1000)
        Tc = T - 273.15
        esh = 6.11 * 10 ** (7.5 * Tc / (237.7 + Tc))
        qv = 621.97 * esh / (1000 - esh)
        ql = qv_sfc - qv
        Theta = 288 + T - (S_Temp - 9.8 * (h/1000))
    return [T, qv, ql, Theta]

def radiation(rad_decay, rad_total, top_height, h):
    """
    Calculates radiative cooling per cubic meter given set of parameters and a given height
    """
    return (rad_total / rad_decay) * math.exp((h-top_height)/rad_decay)

def gamma_pdf(x, k, theta):
    return gamma.pdf(x, k, scale=theta)
    
def solve_for_scale_param(x, dx, N_total, mass_total, k, rho_water=1e6):
    """
    Solve for the scale parameter (theta) of the Gamma distribution that fits the 
    given number and mass concentrations.
    
    Parameters:
    - x: array of midpoints of droplet size bins (in um)
    - dx: array of bin widths (in um)
    - N_total: total number concentration (#/cm^3)
    - mass_total: total mass concentration (g/m^3)
    - k: shape parameter of the Gamma distribution
    - rho_water: density of water in g/m^3 (default is 1e6 g/m^3)
    
    Returns:
    - theta: scale parameter of the Gamma distribution
    """
    # Convert diameters from um to cm
    x_cm = x * 1e-4
    dx_cm = dx * 1e-4
    
    # Function to compute number and mass concentration given theta
    def objective(theta):
        # Compute the Gamma PDF for the midpoints of the bins
        gamma_pdf = gamma.pdf(x_cm, a=k, scale=theta)
        
        # Normalize the Gamma PDF so that the total number concentration matches N_total
        pdf_integral = np.sum(gamma_pdf * dx_cm)
        gamma_pdf_normalized = gamma_pdf / pdf_integral
        
        # Calculate number concentration per bin (in #/cm^3)
        N_bin = N_total * gamma_pdf_normalized * dx_cm
        
        # Calculate mass concentration per bin (in g/m^3)
        mass_bin = N_bin * (4 * np.pi / 3) * x_cm**3 * rho_water
        
        # Calculate total mass concentration
        mass_total_calc = np.sum(mass_bin)
        
        # Return the difference between calculated and provided mass concentrations
        return mass_total_calc - mass_total

    # Use fsolve to find the scale parameter (theta) that satisfies the objective function
    initial_guess = np.mean(x_cm) / k  # A reasonable starting guess for theta
    theta_solution = fsolve(objective, initial_guess)[0]
    
    return theta_solution

def scale_param_gamma(x, dx, N_total, mass_total, k, rho_water=1e6):
    """
    Returns a droplet size distribution defined by its droplet number concentration, CWC, and shape parameter
    """
    return gamma_pdf(x * 1e-4, k, solve_for_scale_param(x, dx, N_total, mass_total, k, rho_water=1e6)) * 1e-4


# Function for Courant calculations
def courant(e, dlnr):
    """
    Calculates courant numbers for collisions code
    Inputs: droplet mass grid (e), logarithmic spacing of droplet mass grid
    Outputs: courant numbers
    """
    n = np.shape(e)[0]
    c = np.zeros((n, n))
    ima = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            x0 = e[i] + e[j]
            for k in range(j, n):
                if e[k] >= x0 and e[k - 1] < x0:
                    if c[i, j] < 1.0 - 1.e-8:
                        kk = k - 1
                        c[i, j] = np.log(x0 / e[k - 1]) / (3.0 * dlnr)
                    else:
                        c[i, j] = 0.0
                        kk = k
                    ima[i, j] = min(n - 1, kk)
                    break
            c[j, i] = c[i, j]
            ima[j, i] = ima[i, j]
    
    return c, ima

# Function to initialize kernel
def trkern(r, winf):
    """
    Uses a Hall kernel
    inputs: r - radius grid
    outputs: ck - kernel
    """
    n = np.shape(r)[0]
    rr = r * 1e-4
    ck = np.zeros((n,n))
    ec = calculate_efficiency(r)
    for j in range(n):
        for i in range(j):
            ck[j, i] = np.pi * (rr[j] + rr[i]) ** 2 * ec[j, i] * abs(winf[j] - winf[i])
            ck[i, j] = ck[j, i]
                
    return ck

# Function for calculating terminal velocity in centimeters per second
def fallg(r):
    """
    Calculates terminal velocities in cm/s based on droplet size
    Inputs: droplet radii
    Outputs: terminal velocities
    """
    n = np.shape(r)[0]
    rr = np.zeros(n)
    winf = np.zeros(n)
    eta = 1.818e-4
    xlamb = 6.62e-6
    rhow = 1.0
    rhoa = 1.225e-3
    grav = 980.665
    cunh = 1.257 * xlamb
    stok = 2.0 * grav * (rhow - rhoa) / (9.0 * eta)
    stb = 32.0 * rhoa * (rhow - rhoa) * grav / (3.0 * eta * eta)
    
    for j in range(n):
        rr[j] = r[j] * 1.e-4
    
    for j in range(n):
        if rr[j] <= 1.e-3:
            winf[j] = stok * (rr[j] ** 2 + cunh * rr[j])
        elif 1.e-3 < rr[j] <= 5.35e-2:
            x = np.log(stb * rr[j] ** 3)
            y = np.polynomial.Polynomial([-0.318657e1, 0.992696, -0.153193e-2, -0.987059e-3, -0.578878e-3, 0.855176e-4, -0.327815e-5])(x)
            xrey = (1.0 + cunh / rr[j]) * np.exp(y)
            winf[j] = xrey * eta / (2.0 * rhoa * rr[j])
        else:
            bond = grav * (rhow - rhoa) * rr[j] ** 2 / 76.1
            if rr[j] > 0.35:
                bond = grav * (rhow - rhoa) * 0.35 ** 2 / 76.1
            x = np.log(16.0 * bond * 76.1 / 3.0)
            y = np.polynomial.Polynomial([-0.500015e1, 0.523778e1, -0.204914e1, 0.475294, -0.542819e-1, 0.238449e-2])(x)
            xrey = 76.1 * np.exp(y)
            winf[j] = xrey * eta / (2.0 * rhoa * rr[j])
            if rr[j] > 0.35:
                winf[j] = xrey * eta / (2.0 * rhoa * 0.35)
                
    return winf

# Collision efficiency calculation
def calculate_efficiency(r):
    """
    calculates collision efficiency based on radius grid
    """
    n = np.shape(r)[0]
    ec = np.zeros((n,n))
    # Data arrays
    # For collision efficiency of the Hall kernel
    r0 = np.array([6., 8., 10., 15., 20., 25., 30., 40., 50., 60., 70., 100., 150., 200., 300.])
    rat = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

    ecoll = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,
                      0.003,0.003,0.003,0.004,0.005,0.005,0.005,0.010,0.100,0.050,0.200,0.500,0.770,0.870,0.970,
                      0.007,0.007,0.007,0.008,0.009,0.010,0.010,0.070,0.400,0.430,0.580,0.790,0.930,0.960,1.000,
                      0.009,0.009,0.009,0.012,0.015,0.010,0.020,0.280,0.600,0.640,0.750,0.910,0.970,0.980,1.000,
                      0.014,0.014,0.014,0.015,0.016,0.030,0.060,0.500,0.700,0.770,0.840,0.950,0.970,1.000,1.000,
                      0.017,0.017,0.017,0.020,0.022,0.060,0.100,0.620,0.780,0.840,0.880,0.950,1.000,1.000,1.000,
                      0.030,0.030,0.024,0.022,0.032,0.062,0.200,0.680,0.830,0.870,0.900,0.950,1.000,1.000,1.000,
                      0.025,0.025,0.025,0.036,0.043,0.130,0.270,0.740,0.860,0.890,0.920,1.000,1.000,1.000,1.000,
                      0.027,0.027,0.027,0.040,0.052,0.200,0.400,0.780,0.880,0.900,0.940,1.000,1.000,1.000,1.000,
                      0.030,0.030,0.030,0.047,0.064,0.250,0.500,0.800,0.900,0.910,0.950,1.000,1.000,1.000,1.000,
                      0.040,0.040,0.033,0.037,0.068,0.240,0.550,0.800,0.900,0.910,0.950,1.000,1.000,1.000,1.000,
                      0.035,0.035,0.035,0.055,0.079,0.290,0.580,0.800,0.900,0.910,0.950,1.000,1.000,1.000,1.000,
                      0.037,0.037,0.037,0.062,0.082,0.290,0.590,0.780,0.900,0.910,0.950,1.000,1.000,1.000,1.000,
                      0.037,0.037,0.037,0.060,0.080,0.290,0.580,0.770,0.890,0.910,0.950,1.000,1.000,1.000,1.000,
                      0.037,0.037,0.037,0.041,0.075,0.250,0.540,0.760,0.880,0.920,0.950,1.000,1.000,1.000,1.000,
                      0.037,0.037,0.037,0.052,0.067,0.250,0.510,0.770,0.880,0.930,0.970,1.000,1.000,1.000,1.000,
                      0.037,0.037,0.037,0.047,0.057,0.250,0.490,0.770,0.890,0.950,1.000,1.000,1.000,1.000,1.000,
                      0.036,0.036,0.036,0.042,0.048,0.230,0.470,0.780,0.920,1.000,1.020,1.020,1.020,1.020,1.020,
                      0.040,0.040,0.035,0.033,0.040,0.112,0.450,0.790,1.010,1.030,1.040,1.040,1.040,1.040,1.040,
                      0.033,0.033,0.033,0.033,0.033,0.119,0.470,0.950,1.300,1.700,2.300,2.300,2.300,2.300,2.300,
                      0.027,0.027,0.027,0.027,0.027,0.125,0.520,1.400,2.300,3.000,4.000,4.000,4.000,4.000,
                      4.000]).reshape(15, 21)
    for j in range(n):
        for i in range(j+1):
            ir = 1
            if r[j] >= r0[14]:
                ir = 16
            elif r[j] <= r0[0]:
                ir = 1
            else:
                for k in range(1, 15):
                    if r[j] <= r0[k] and r[j] >= r0[k-1]:
                        ir = k + 1

            rq = r[i] / r[j]
            iq = 1
            for kk in range(1, 21):
                if rat[kk-1] < rq <= rat[kk]:
                    iq = kk

            if ir < 16:
                if ir >= 2:
                    p = (r[j] - r0[ir-2]) / (r0[ir-1] - r0[ir-2])
                    q = (rq - rat[iq-1]) / (rat[iq] - rat[iq-1])
                    ec[j, i] = (
                        (1 - p) * (1 - q) * ecoll[ir-2, iq-1] +
                        p * (1 - q) * ecoll[ir-1, iq-1] +
                        q * (1 - p) * ecoll[ir-2, iq] +
                        p * q * ecoll[ir-1, iq]
                    )
                else:
                    q = (rq - rat[iq-1]) / (rat[iq] - rat[iq-1])
                    ec[j, i] = (1 - q) * ecoll[0, iq-1] + q * ecoll[0, iq]
            else:
                q = (rq - rat[iq-1]) / (rat[iq] - rat[iq-1])
                ek = (1 - q) * ecoll[14, iq-1] + q * ecoll[14, iq]
                ec[j, i] = min(ek, 1.0)

            ec[i, j] = ec[j, i]

            if ec[i, j] < 1e-20:
                print("uh oh")
                raise ValueError("Collision efficiency too small, stopping execution")
                
    return ec

def theta_v_func(T, qv, ql, Theta_288):
    """ Calculates virtual potential temperature profile given profiles of temperature and moisture
    """
    Theta = 288 + T - Theta_288
    Theta_v = np.multiply(Theta, (1 + 0.00061 * qv - 0.001 * ql))
    return Theta_v

def Richardson_func(Theta_v, z_e):
    """ Calculates the Richardson number at every grid level
    """
    z2 = np.power(z_e,2)
    
    # Assume a minimum du/dz of 0.01 /s
    z2[z2>10000] = 10000
    Thetav = (Theta_v[0:-1] + Theta_v[1:]) / 2
    Partial_Theta_z = Theta_v[1:] - Theta_v[0:-1]
    Ri = np.multiply(np.multiply(9.8 / Thetav, z2),Partial_Theta_z)
    Ri[np.absolute(Ri)<1e-3] = 0
    return Ri

def K_func(lam,k,Rich,z_e):
    """ Calculates K's for mixing using Richardson numbers
    """
    dudz = np.divide(1,z_e)
    
    # Minimum du/dz for K's
    dudz[dudz < 0.01] = 0.01
    lamstuff = lam / (1 + lam / (k * z_e))
    Ri1 = np.multiply(Rich,1)
    Ri2 = np.multiply(Rich,1)
    Ri1[ Ri1 > 0 ] = 0
    Ri1[ Ri1 != 0 ] *= -16
    Ri1[ Ri1 != 0 ] += 1
    Ri1[ Ri1 != 0 ] **= 0.2
    Ri2[ Ri2 < 0 ] = 0
    Ri2[ Ri2 != 0 ] *= -5
    Ri2[ Ri2 != 0 ] += 1
    Ri2[Ri2 < 0 ] = 50
    Ri2[ Ri2 != 0 ] **= 2
    Rit = np.add(Ri1,Ri2)
    Rit[Rit == 0] = 1
    Rit[Rit == 2500 ] = 0
    # print(Rit)
    lamsquare = np.multiply(lamstuff, lamstuff)
    lamdudz = np.multiply(lamsquare,dudz)
    Ka = np.multiply(lamdudz,Rit)
    return Ka

# initially we will assume zero surface flux and zero entrainment
def flux_func(prof, K, dz):
    """ Calculates fluxes through each flux level
    """
    diffs = np.diff(prof)
    fluxes = (-1 / dz) * diffs * K
    return fluxes

def flux_func_m(prof, K, dz):
    """
    version of flux_func for microphysics grid
    """
    diffs = np.diff(prof, axis = 0)
    fluxes = (-1/dz) * diffs * K[:, np.newaxis]
    return fluxes


# dz = 1
def turb_transfer_func(fluxes, dt, dz):
    """ Calculates change in value for each timestep due to turbulent flux divergence
    """
    
    if fluxes.ndim == 1:
        t_flux = [0.]
        b_flux = [0.]
        tb_fluxes = np.concatenate((b_flux, fluxes, t_flux),axis=None)
    else:
        t_flux = np.zeros((1, np.shape(fluxes)[1]))
        b_flux = np.zeros((1, np.shape(fluxes)[1]))
        tb_fluxes = np.concatenate((b_flux, fluxes, t_flux),axis=0)
        
    flux_dif = (tb_fluxes[1:] - tb_fluxes[0:-1]) / dz
    flux_tstep_change = - dt * flux_dif
    return flux_tstep_change

def condensation_step_a(T_in,qv_in):
    """ Calculates amount of water condensed in g/kg of dry air
    """
    TC = T_in-273.15
    e_es = 6.11 * 10 ** (7.5 * TC / (237.7 + TC))
    qv_s = 621.97 * e_es / (1000 - e_es)
    delta_qv = (qv_in - qv_s) * 0.25
    delta_qv[delta_qv < 0] = 0
    return delta_qv

def condensation_step_b(delta_ql, LWC_in, r, nbins):
    """ Calculates amount of water condensed onto each bin
    """
    sfc = LWC_in / r[np.newaxis,:nbins]
    sfc_sum = np.sum(sfc, axis = 1)
    sfc_sum[sfc_sum < 1e-6] = 1e-6
    norm_sfc= sfc / sfc_sum[:,np.newaxis]
    norm_sfc[norm_sfc < 1e-6] = 0.
    bin_delta_ql = norm_sfc * delta_ql[:,np.newaxis]
    bin_delta_ql[bin_delta_ql < 1e-8] = 0.
    bin_delta_ql[np.isnan(bin_delta_ql)] = 0.
    return bin_delta_ql


def evaporation_step(T_in, qv_in, LWC_in, r, rb, r_widths, dt, nbins):
    """ Calculates evaporated water and redistributes LWC between bins
    """
    dd = np.exp(8.488 - 2281.86 / T_in)
    tc = T_in-273.15
    e_es = 6.11 * 10 ** (7.5 * tc / (237.7 + tc))
    qv_s = 621.97 * e_es / (1000 - e_es)
    delta_qv = qv_s - qv_in
    delta_qv[delta_qv < 10**(-10)] = 10**(-10)
    delta_D = dd * delta_qv
    bin_time = 0.1 * r[np.newaxis,:nbins] **2 / delta_D[:,np.newaxis]
    # print(np.min(bin_time, axis = 0))
    edge_time = 0.1 * rb[np.newaxis,1:nbins+1] **2 / delta_D[:,np.newaxis]
    bin_time[bin_time < dt] = dt
    edge_time[edge_time < dt] = dt
    ql_loss = (dt / bin_time) * LWC_in
    # qloss = np.sum(ql_loss, axis = 1)

    new_edge = rb[np.newaxis,1:nbins+1] * ((1 - (dt / edge_time)) ** (1/3))
    prop_change = (rb[np.newaxis, 1:nbins+1] - new_edge) / r_widths[np.newaxis, 1:nbins+1]
    prop_change[prop_change > 1.] = 1.
    # print(np.mean(prop_change, axis = 0))
    change1 = np.zeros(np.shape(prop_change))
    change2 = np.zeros(np.shape(prop_change))
    change1[:,:-1] = (LWC_in[:,1:] - ql_loss[:,1:]) * prop_change[:,:-1]
    change2[:,1:] = -change1[:,:-1]
    delta_ql = -ql_loss + change1 + change2
    # print(np.sum(-ql_loss, axis = 0))
    delta_ql[np.abs(delta_ql) < 1e-12] = 0.
    delta_ql = np.minimum(delta_ql,LWC_in)
    delta_ql[np.isnan(delta_ql)] = 0.
    delta_ql[delta_qv < 10**(-8), :] = 0.
    # dql = np.sum
    return delta_ql

def settling_step(winf, bin_LWC, dt, dz,nbins):
    """
    calculates droplet settling flux
    """
    flux = (winf[np.newaxis,:nbins] * dt / (dz * 100.)) * bin_LWC
    # print((np.sum(flux[-1,:]) / dt) / np.sum(bin_LWC[-1,:]))
    return flux[:,:nbins]

def model_subtimestep(T_i, Theta_i, qv_i, ql_i, winf, dtt, dz, Theta_288, S_Temp, z_fluxes, lam, k, rad_prof, nbins, r, rb, r_widths, e):
    """ 
    Runs the transport, condensation, and evaporation portions of the model for each substep
    """
    # Radiation
    T_rad = radiation_step(rad_prof, dtt)

    T_ir = T_i + T_rad

    # Calculate K
    ql = np.sum(ql_i, axis = 1)
    Theta_v_i = theta_v_func(T_i, qv_i, ql, Theta_288)
    Ri_i = Richardson_func(Theta_v_i, z_fluxes)
    k_i = K_func(lam,k,Ri_i,z_fluxes)
    k_i[k_i > 1.] = 1.
    
    # Calculate Fluxes
    Theta_df = flux_func(Theta_i, k_i, dz)
    qv_df = flux_func(qv_i, k_i, dz)
    ql_df = flux_func_m(ql_i, k_i, dz)
    
    # Calculate turbulent fluxes profiles
    Theta_turb = turb_transfer_func(Theta_df,dtt, dz) 
    T_turb = Theta_turb
    qv_turb = turb_transfer_func(qv_df,dtt, dz)
    ql_turb = turb_transfer_func(ql_df,dtt, dz)
    
    

    # Settling
    sflux = settling_step(winf, ql_i, dtt, dz, nbins)
    dep_i = sflux[0,:]
    ql_sflux = np.zeros(np.shape(sflux))
    ql_sflux[:-1,:] = np.diff(sflux, axis = 0)
    ql_sflux[-1,:] = -sflux[-1,:]
    
    # Evaporation
    ql_l = evaporation_step(T_ir, qv_i, ql_i, r, rb, r_widths, dtt, nbins)
    ql_evap = np.sum(ql_l, axis = 1)
    qv_evap = -ql_evap
    T_evap = 2.487 * ql_evap
    
    # Condensation
    d_ql = condensation_step_a(T_ir,qv_i)
    b_dql = condensation_step_b(d_ql, ql_i, r, nbins)
    qv_c = -d_ql
    T_cond = 2.487 * d_ql

    T_o = T_ir + T_turb+ T_evap + T_cond
    qv_o = qv_i + qv_turb + qv_evap + qv_c
    ql_o = ql_i + ql_l + b_dql + ql_turb + ql_sflux
    # print(T_i[-1], T_turb[-1], qv_evap[-1] + qv_c[-1])
    # print(np.sum(ql_i[-1,:]), np.sum(ql_l[-1,:]), np.sum(b_dql[-1,:]), np.sum(ql_turb[-1,:]), np.sum(ql_sflux[-1,:]))
    
    
    Theta_o = S_Temp + T_o - Theta_288
    
    return T_o, Theta_o, qv_o, ql_o, dep_i

def coad(g, ima, e, r, ck, dt, c, gmin, dlnr):
    """
    calculates new DSD from input DSD
    """
    g[g < 0] = 0.
    ck = ck * dt * dlnr * np.log(2)
    n = np.shape(g)[0]
    gg = g.copy()
    # Lower and upper integration limits
    for i in range(n - 1):
        i0 = i
        if (g[i] / r[i]) > gmin:
            break

    for i in range(n - 2, -1, -1):
        i1 = i
        if (g[i] / r[i]) > gmin:
            break
    
     
    for i in range(i0, i1+1):
        for j in range(i, i1+1):
            k = ima[i,j]
            kp = k + 1
            g[kp] = max(g[kp], 0.)
            x0 = ck[i, j] * g[i] * g[j]
            x0 = min(x0, g[i] * e[j])
            if j != k:
                x0 = min(x0, g[j] * e[i])
            gsi = x0 / e[j]
            gsj = x0 / e[i]
            gsk = gsi + gsj
            g[i] -= gsi
            g[j] -= gsj
            gk = g[k] + gsk
            if (gk / r[k]) > gmin:
                if gk < 0. or gk == 0.:
                    print("oh no gk", gg)
                if g[kp] < 0.:
                    print("oh no g[kp]", gg)
                x1 = np.log(g[kp] / gk + 1.0e-20)
                flux = gsk / x1 * (np.exp(0.5 * x1) - np.exp(x1 * (0.5 - c[i, j])))
                flux = min(flux, gk)
                g[k] = gk - flux
                g[kp] += flux
                
    return g

def collision_tstep(LWC_in, ima, e, r, ck, dt, c, gmin, dlnr, r_widths, nbins):
    """
    Runs the collision step of the model
    """
    rain = 0
    LWC_out = copy.deepcopy(LWC_in)
    LWC_in[LWC_in < 0] = 0.
    for i in range(np.shape(LWC_in)[0]):
        if np.sum(LWC_in[i,:]) > 1e-4:
            lnr = np.log(r)
            dMdlnR = np.zeros(np.shape(r))
            dMdlnR[:nbins] = (LWC_in[i,:] / r_widths[:nbins]) / r[:nbins]
            dMdlnR[dMdlnR < 0.] = 0.
            factor  = np.trapz(dMdlnR, lnr) * 1000 / (np.nansum(LWC_in[i,:]))
            dMdlnR = dMdlnR / factor
            dMdlnRo = coad(dMdlnR, ima, e, r, ck, dt, c, gmin, dlnr) 
            LWC_io = dMdlnRo * r * r_widths
            LWC_io2 = LWC_io * np.nansum(LWC_in[i,:]) / np.nansum(LWC_io)
            LWC_out[i,:] = LWC_io2[:nbins]
            rain += np.sum(LWC_io2[nbins:])

    return LWC_out, rain

def radiation_step(rad_prof, dt):
    """ 
    Applies radiative cooling
    """
    deltaT = -1 * rad_prof * dt / (1.27 * 10**3)
    # print(deltaT)
    outprof = deltaT
    return outprof
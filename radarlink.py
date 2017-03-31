from math import sqrt, cos, pi, exp, factorial
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt

# Planck constant times light speed
CONST_HC = 6.626176e-34 * 2.99792458e8
R_Earth = 6378000.0 # meters

# Utility function
def multiply_values(D):
    result = 1
    for value in D.values():
        result *= value
    return result

# Transmission values of elements in the transmit optics. These are all multiplied
# to get the total transmission eta_t.
def get_eta_t():
    transmit_optics = {
        "TM1" : 1.00000,
        "TM2" : 0.99995,
        "TM3" : 0.99995,
        "BEx1" : 0.99900,
        "TM4" : 0.99995,
        "TM5" : 0.99000,
        "CW1" : 0.99900,
        "CM1" : 0.99000,
        "CM2" : 0.99000,
        "CM3" : 0.99000,
        "CM4" : 0.99000,
        "CM5" : 0.99000,
        "BEx2" : 0.99900,
        "CW2" : 0.99900
    }
    return multiply_values(transmit_optics)

# Transmission values of elements in the receive optics. These are all multiplied
# to get the total transmission eta_r.
# The bandpass filter value is 0.9 for the night filter and 0.6 for the daytime filter.
def get_eta_r(night=True):
    receive_optics = {
        "telW" : 0.999,
        "primM" : 0.960,
        "secM" : 0.960,
        "detBoxW" : 0.999,
        "coll" : 0.999,
        "DichM1" : 0.990,
        "bandpass" : 0.900 if night else 0.600,
        "NDfilter" : 0.999
    }
    return multiply_values(receive_optics)


# Atmospheric transmission
# These is done with slightly too clever functional tricks.

# Atmospheric transmission function
def atmospheric_transmittance(T0, delta, theta_z):
    theta_z *= pi/180
    return T0 - delta / cos(theta_z)

# Atmospheric values
Ta_values = {
    "green" : ({
        "very clear" : 0.95,
        "standard" : 0.84,
        "clear" : 0.72,
        "light haze" : 0.59
    }, 0.15),
    "IR" : ({
        "very clear" : 1.04,
        "standard" : 0.93,
        "clear" : 0.82,
        "light haze" : 0.54
    }, 0.07)
}

# Atmospheric transmission function maker
def T_atmos_func(wavelength, condition):
    T0 = Ta_values[wavelength][0][condition]
    dT = Ta_values[wavelength][1]
    def f(theta):
        return atmospheric_transmittance(T0, dT, theta)
    return f


# Cirrus function
def cirrus_transmittance(t, theta_z):
    theta_z *= pi/180
    return exp(-0.14 * (t / cos(theta_z))**2)


# Compute slant range from station height, target altitude and zenith angle of target.
# Assumes spherical Earth.
def slant_range(h_station, h_target, zenith_angle):
    r_station = R_Earth + h_station
    cosz = cos(zenith_angle * pi/180)
    t = sqrt(r_station**2 * cosz**2 + 2*R_Earth*(h_target-h_station) + h_target**2 - h_station**2)
    return -r_station * cosz + t


# Compute transmitter gain G_t
#  theta_t = far-field divergence (arcsec)
#  theta = seeing (arcsec)
def gain(theta_t, theta):
    return 8/(0.0000048481*theta_t)**2 * exp(-2*(theta/theta_t)**2)

# Radar link equation implemented
def n_electrons(eta_q, pulse_energy, wavelength, eta_t, transmit_gain, cross_section,
                R, aperture, eta_r, T_atmos, T_cirrus):
    N = eta_q * pulse_energy * wavelength / CONST_HC
    N *= eta_t * transmit_gain * cross_section / (4*pi*R**2)**2
    N *= aperture * eta_r
    N *= T_atmos**2 * T_cirrus**2
    return N

# Probability of detecting at least the threshold number of photoelectrons
# in a pulse, if n is the the expected number of electrons.
def pulse_probability(n, threshold):
    S = 0
    for m in range(threshold):
        S += n**m / factorial(m)
    return 1 - exp(-n) * S

# Binomial distribution probability density function.
def binomial(f, m, p):
    return comb(f, m, exact=False) * p**m * (1-p)**(f-m)

# Probability of detecting at least k pulses per second, given
#   f = laser fire rate (pulses per second)
#   p = probability of detecting a single pulse
def detection_probability(f, k, p):
    return 1 - sum(binomial(f, m, p) for m in range(k))



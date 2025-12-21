import matplotlib.pyplot as plt
import numpy as np
from spectratestingmodified_chirag_edit import (
    calculate_wavenumber_components,
    calculate_thermal_velocity,
    calculate_cyclotron_frequency,
    calculate_average_gyroradius,
    calculate_debye_length,
    calculate_alpha,
    calculate_sound_speed,
    calculate_omega_values,
    calculate_collisional_term,
    calculate_modified_distribution,
    calculate_electric_susceptibility,
    calcSpectra
)

# Define constants and parameters
EPSILON_0 = 8.854187817e-12  # Permittivity of free space [F/m]
KB = 1.380649e-23           # Boltzmann constant [J/K]
E = 1.602e-19               # Elementary charge [C]

user_values = {
    "nu_i": 1.0e-7,          # Ion collision frequency [Hz]
    "nu_e": 1.0e-7,          # Electron collision frequency [Hz]
    "ni": 2.0e11,            # Ion density [m^-3]
    "ne": 2.0e11,            # Electron density [m^-3]
    "mi": 2.65686e-26,       # Ion mass (O+) [kg]
    "me": 9.11e-31,          # Electron mass [kg]
    "B": 3.6e-5,             # Magnetic field [T]
    "theta": 60,             # Scattering angle [degrees]
    "Te": 500,               # Electron temperature [K]
    "Ti": 500,               # Ion temperature [K]
    "frequency": 430e6,      # Radar frequency [Hz]
    "epsilon_0": EPSILON_0,  # Permittivity of free space [F/m]
    "kB": KB,                # Boltzmann constant [J/K]
    "e": E,                  # Elementary charge [C]
    "n_terms": 2001          # Number of frequency points for calculation
}

# Perform calculations
c_light = 3e8  # Speed of light in m/s
lambda_wavelength = c_light / user_values["frequency"]
k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(lambda_wavelength, user_values["theta"])
vth_e = calculate_thermal_velocity(KB, user_values["Te"], user_values["me"])
Oc_e = calculate_cyclotron_frequency(1, E, user_values["B"], user_values["me"])
rho_e = calculate_average_gyroradius(vth_e, Oc_e)
lambda_De = calculate_debye_length(user_values["Te"], user_values["ne"], EPSILON_0, KB, E)
alpha_e = calculate_alpha(k_total, lambda_De)
c = calculate_sound_speed(KB, user_values["Te"], user_values["Ti"], user_values["mi"])
omega_values = calculate_omega_values(k_total, c)
U_e = calculate_collisional_term(user_values["nu_e"], k_parallel, vth_e, k_perpendicular, rho_e, user_values["n_terms"], omega_values, Oc_e)
M_e = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_e, user_values["n_terms"], rho_e, Oc_e, user_values["nu_e"], U_e)
chi_e = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_e, user_values["n_terms"], rho_e, Oc_e, user_values["nu_e"], alpha_e, U_e, user_values["Te"], user_values["Ti"])

# Multi-ion species calculations
ion_species = [
    {"name": "O+",   "fraction": 0.488, "density": 2.03e5, "mass": 2.65686e-26},
    {"name": "N+",   "fraction": 0.032, "density": 1.33e4, "mass": 2.32587e-26},
    {"name": "H+",   "fraction": 0.456, "density": 1.89e5, "mass": 1.67262e-27},
    {"name": "HE+",  "fraction": 0.024, "density": 9.96e3, "mass": 6.64648e-27},
    {"name": "O2+",  "fraction": 0.0,   "density": 0.0,    "mass": 5.31372e-26},
    {"name": "NO+",  "fraction": 0.0,   "density": 0.0,    "mass": 2.4828e-26}
]

M_i_total = 0
chi_i_total = 0
for ion in ion_species:
    if ion["fraction"] > 0:
        mi = ion["mass"]
        frac = ion["fraction"]
        vth_i = calculate_thermal_velocity(KB, user_values["Ti"], mi)
        Oc_i = calculate_cyclotron_frequency(1, E, user_values["B"], mi)
        rho_i = calculate_average_gyroradius(vth_i, Oc_i)
        lambda_Di = calculate_debye_length(user_values["Ti"], user_values["ni"], EPSILON_0, KB, E)
        alpha_i = calculate_alpha(k_total, lambda_Di)
        U_i = calculate_collisional_term(user_values["nu_i"], k_parallel, vth_i, k_perpendicular, rho_i, user_values["n_terms"], omega_values, Oc_i)
        M_i = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_i, user_values["n_terms"], rho_i, Oc_i, user_values["nu_i"], U_i)
        chi_i = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_i, user_values["n_terms"], rho_i, Oc_i, user_values["nu_i"], alpha_i, U_i, user_values["Ti"], user_values["Ti"])
        M_i_total += frac * M_i
        chi_i_total += frac * chi_i

# Calculate spectra
spectra = calcSpectra(M_i_total, M_e, chi_i_total, chi_e)

# Convert omega_values to frequency in MHz
frequency_MHz = omega_values / (2 * np.pi * 1e6)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frequency_MHz, spectra, label='Spectra')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Spectra')
plt.title('ISR Spectra')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
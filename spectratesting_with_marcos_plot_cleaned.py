import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import ive, dawsn
import scipy.special as sp
import numexpr as ne

sys.path.append(os.path.join(os.path.dirname(__file__), 'EEVersion'))
import plasma_parameters as plasma_param
import physical_constants as phys_cons
from isr_spectrum import isrSpectrum

def calculate_wavenumber_components(lambda_wavelength, theta_degrees):
    """
    Parameters:
    - lambda_wavelength (float): wavelength of the radar wave [m].
    - theta_degrees (float): angle [degrees].
    
    Returns:
    - k (float): Total wavenumber of the radar wave [rad/m].
    - k_parallel (float): Parallel component of the wavenumber [rad/m].
    - k_perpendicular (float): Perpendicular component of the wavenumber [rad/m].
    """
    theta = np.radians(theta_degrees) #Convert theta value from degrees to radians
    k = 2 * np.pi / lambda_wavelength #Calculate total wavenumber 
    k_parallel = k * np.cos(theta) #Break down k into the portion that is in the same direction of propogation
    k_perpendicular = k * np.sin(theta) #Break down k into the portion that is orthogonal to the parallel direction 
                                        #(the portion that is perpendicular to the direction of propogation)
    return k, k_parallel, k_perpendicular #Return the total wavenumber and it's components

def calculate_sound_speed(kB, Te, Ti, mi):
    """
    Parameters:
    - kB (float): Boltzmann constant [J/K].
    - Te (float): Electron temperature [Kelvin].
    - Ti (float): Ion temperature [Kelvin].
    - mi (float): Mass of the ion [kg].

    Returns:
    - c (float): Sound speed of the plasma [m/s].
    """
    c = np.sqrt((5 / 3) * (kB * (Ti + Te) / mi)) #calculate the sound speed
    return c

def calculate_omega_values(k, c):
    """
    Parameters:
    - k (float): Wavenumber of the radar wave [rad/m]. 
    - c (float): Sound speed of the plasma [m/s]. 
    Returns:
    - an array of doppler shifted frequency points [Hz]
     """
    # Define a uniform range for 100,000 points
    total_points = 50000
    start_freq_mhz = -6.0  # Start frequency in MHz
    end_freq_mhz = 6.0    # End frequency in MHz

    # Generate a uniform frequency array
    omega_hz = np.linspace(start_freq_mhz * 1e6, end_freq_mhz * 1e6, total_points)
    omega_rad = omega_hz * 2 * np.pi  # Convert Hz to rad/s

    return omega_rad

def calculate_thermal_velocity(kB, T, m):
    """
    Parameters:
    - kB (float): Boltzmann constant [J/K].
    - T (float): Temperature [Kelvin].
    - m (float): Mass [kg].

    Returns:
    - vth (float): Thermal velocity [m/s].
    """
    vth = np.sqrt( (2 * kB * T ) / m) #calculate the thermal velocity
    return vth

def calculate_cyclotron_frequency(Z, e, B, m):
    """
    Parameters:
    - Z (int): Charge state of the particle.
    - e (float): Elementary electric charge [C].
    - B (float): Magnetic field strength [T].
    - m (float): Mass of the particle [kg]

    Returns:
    - Oc(float): Cyclotron frequency [rad/s].
    """
    Oc = (Z * e * B) / m #calculate the cyclotron frequency
    return Oc

def calculate_average_gyroradius(vth, Oc):
    """
    Parameters:
    - vth (float): Thermal velocity [m/s].
    - Oc (float): Cyclotron frequency [rad/s].

    Returns:
    - rho (float): Average gyroradius [m].
    """
    rho = vth / (np.sqrt(2) * Oc) #calculate the gyroradius
    return rho

def calculate_collisional_term(nu, k_par, vth, k_perp, rho, n, omega, Oc):
    """
    Parameters:
    - nu (float): Collisional Frequency [Hz].
    - k_par (float): Parallel component of the wavenumber [rad/m].
    - vth (float): Thermal velocity [m/s].
    - k_perp (float): Perpendicular component of the wavenumber [rad/m].
    - rho (float): Average gyroradius [m].
    - n (int): constant for summation.
    - omega (float): An array of doppler shifted frequency points [Hz].
    - Oc (float): Cyclotron frequency [rad/s].

   Returns;
    - U (complex float): Collisional Term.
    """
    U = np.zeros_like(omega) + 1j * 0.0
    k_rho_sq = (k_perp ** 2) * (rho ** 2)
    for i in range(-n, n + 1):
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        exp_term = ne.evaluate('exp(-yn**2)')
        U += sp.ive(i, k_rho_sq) * (2 * sp.dawsn(yn) + 1j * np.sqrt(np.pi) * exp_term)
    U = U * 1j * nu / (k_par * vth)
    return U

def calculate_modified_distribution(omega, k_par, k_perp, vth, n, rho, Oc, nu, U):
    """
    Calculate the modified distribution using the updated formula with Dawson function.
    """
    M = np.zeros_like(omega) + 1j * 0.0
    k_rho_sq = (k_perp ** 2) * (rho ** 2)
    term_one = -(abs(U) ** 2) / (nu * (abs(1 + U) ** 2))
    term_two = 1 / (k_par * vth * (abs(1 + U) ** 2))
    for i in range(-n, n + 1):
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        exp_re = np.real(ne.evaluate('exp(-yn**2)'))
        imag_da = 2 * np.imag(dawsn(yn))
        bessel_term = ive(i, k_rho_sq)
        M += bessel_term * (np.sqrt(np.pi) * exp_re + imag_da)
    M_s = term_one + term_two * M
    return np.real(M_s)

def calculate_debye_length(Te, ne_, epsilon_0, kB, e):
    """
    Calculate the electron Debye length.

    Parameters:
    - Te (float): Electron temperature [Kelvin].
    - ne (float): Electron density [m^-3].
    - epsilon_0 (float): Vacuum permittivity  [F/m].
    - kB (float): Boltzmann constant [J/K].
    - e (float): Elementary charge [C].

    Returns:
    - Electron Debye length (float) [m].
    """
    lambda_De = np.sqrt(epsilon_0 * kB * Te / (ne_ * e**2)) #calculate the debye length
    return lambda_De

def calculate_alpha(k, lambda_De):
    """
    Parameters:
    - k (float): Wavenumber of the radar wave [rad/m].
    - lambda_De (float): Electron Debye length [m].

    Returns:
    - alpha (float): the incoherent scattering parameter.
    """
    alpha = 1 / (k * lambda_De) #calculate alpha
    return alpha

def calculate_electric_susceptibility(omega, k_par, k_perp, vth, n, rho, Oc, nu, alpha, U, Te, Ts):
    """
    Parameters:
    - omega (float): An array of doppler shifted frequency points [Hz].
    - k_par (float): Parallel component of the wavenumber [rad/m].
    - k_perp (float): Perpendicular component of the wavenumber [rad/m].
    - vth (float): Thermal velocity [m/s].
    - n (int): constant for summation.
    - rho (float): Average gyroradius [m].
    - Oc (float): Cyclotron frequency [rad/s].
    - nu (float): Collision frequency (range of 0.01 to 10) [Hz].
    - alpha (float): Incoherent scattering parameter.
    - U (float): Collisional term.
    - Te (float): Temperature of the electron [Kelvin].
    - Ts (float): Temperature of the species [Kelvin].
    Returns:
    - an array of values, representing the real and imaginary portion of the electric susceptibility.
    """
    chi = np.zeros_like(omega) + 1j * 0.0
    k_rho_sq = (k_perp ** 2) * (rho ** 2)
    term_one = (omega - 1j * nu) / (k_par * vth)
    term_four = (alpha ** 2) / (1 + U)
    term_five = Te / Ts
    for i in range(-n, n + 1):
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        term_two = 2 * sp.dawsn(yn)
        exp_term = ne.evaluate('exp(-yn**2)')
        term_three = 1j * np.sqrt(np.pi) * exp_term
        chi += sp.ive(i, k_rho_sq) * (1 - term_one * (term_two + term_three))
    return term_four * term_five * chi

def calcSpectra(M_i, M_e, chi_i, chi_e):
    """
    Parameters:
    - M_i (float): An array representing the modified distribution for ions .
    - M_e (float): An array representing the modified distribution for electrons.
    - chi_i (complex float): An array representing the electric susceptibility for ions.
    - chi_e  (complex float): An array representing the electric susceptibility for electrons.
    Returns:
    - Spectrum (complex float): An array representing the incoherent scatter spectrum
    """
    epsilon = 1 + chi_e + chi_i #calculate the epsilon constant
    
    spectrum = 2*np.abs(1-chi_e/epsilon)**2*M_e+2*np.abs(chi_e/epsilon)**2*M_i #calculate the spectrum using ion and electron components
    return spectrum

def plot_combined_spectra(omega_values, chirag_spectra_list, marcos_spectra, Te_values, alpha_chirag=None, alpha_marco=None):
    """
    Plots both Chirag's and Marcos' spectra on the same plot.

    Parameters:
    - omega_values: Array of angular frequency values.
    - chirag_spectra_list: List of spectra from Chirag's calculations.
    - marcos_spectra: Tuple containing Marcos' frequency and spectra.
    - Te_values: List of electron temperatures corresponding to Chirag's spectra.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Chirag's spectra
    for i, spectra in enumerate(chirag_spectra_list):
        ax.plot(omega_values / (2 * np.pi * 1e6), spectra, label="NJIT Spectra", color='blue', linewidth=4)

    # Plot Marcos' spectra
    marcos_freq, marcos_spectrum = marcos_spectra
    ax.plot(marcos_freq, marcos_spectrum, label="Marco Spectra", color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Frequency (MHz)', fontsize=14)
    ax.set_ylabel('Spectra', fontsize=14)
    ax.set_title('Spectra Combined', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlim(-6, 6)
    # Annotate alpha (collective parameter) values on the plot
    if (alpha_chirag is not None) or (alpha_marco is not None):
        lines = []
        if alpha_chirag is not None:
            lines.append(f"Chirag α = {alpha_chirag:.3e}")
        if alpha_marco is not None:
            lines.append(f"Marco α = {alpha_marco:.3e}")
        ax.text(
            0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.8')
        )

    plt.tight_layout()
    plt.show()

# Ensure the script only runs when executed directly, not when imported
if __name__ == "__main__":
    # Sample Data
    nu_i = 1364.30161374848 # Ion collision frequency in Hz
    nu_e = 3.32424637  # Electron collision frequency in Hz
    ni =  2e11  # Ion and electron densities in m^-3
    ne_ = 2e11
    mi = 2.65686e-26  # Ion mass (atomic oxygen) in kga
    m_e = 9.11e-31  # Electron mass [kg]
    B = 3.60e-5  # Magnetic field strength in Tesla
    theta = 89  # Scattering angle in degrees
    Te_values = 500 # Electron temperatures in Kelvin
    Ti = 500

    epsilon_0 = 8.854187817e-12  # Vacuum permittivity [F/m]
    kB = 1.380649e-23  # Boltzmann constant [J/K]
    e = 1.602e-19  # Elementary charge [C]

    # Define your ion species here
    ion_species = [
        {"name": "O+",   "fraction": 1.0, "density": 2e11, "mass": 2.65686e-26},
        {"name": "N+",   "fraction": 0.0, "density": 1.33e4, "mass": 2.32587e-26},
        {"name": "H+",   "fraction": 0.0, "density": 1.89e5, "mass": 1.67262e-27},
        {"name": "HE+",  "fraction": 0.0, "density": 9.96e3, "mass": 6.64648e-27},
        {"name": "O2+",  "fraction": 0.0,   "density": 0.0,    "mass": 5.31372e-26},
        {"name": "NO+",  "fraction": 0.0,   "density": 0.0,    "mass": 2.4828e-26}
    ]

    n_terms = 2001

    k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(0.341, theta)
    
    vth_e = calculate_thermal_velocity(kB, Te_values, m_e)
    Oc_e = calculate_cyclotron_frequency(1, e, B, m_e)
    rho_e = calculate_average_gyroradius(vth_e, Oc_e)
    lambda_De = calculate_debye_length(Te_values, ne_, epsilon_0, kB, e)
    alpha_e = calculate_alpha(k_total, lambda_De)
    alpha_chirag = alpha_e  # Chirag alpha (collective parameter) used in this formulation
    c = calculate_sound_speed(kB, Te_values, Ti, sum(ion["fraction"] * ion["mass"] for ion in ion_species))
    omega_values = calculate_omega_values(k_total, c)

    # --- GET MARCO'S COLLISION FREQUENCIES ---
    plasma_tmp = plasma_param.Plasma(ne_, B, Te_values, Ti, 16, 1)

    nu_e_marco = plasma_tmp.collision_frequency(0, 2, kB=k_total, aspdeg=1)[0]
    nu_i_marco = plasma_tmp.collision_frequency([1], 2, kB=k_total, aspdeg=1)[0]

    print("Marco electron collision freq:", nu_e_marco)
    print("Marco ion collision freq:", nu_i_marco)

    # Overwrite Chirag's nu with Marco's values (after testing)
    nu_e = nu_e_marco
    nu_i = nu_i_marco

    U_e = calculate_collisional_term(nu_e, k_parallel, vth_e, k_perpendicular, rho_e, n_terms, omega_values, Oc_e)
    M_e = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, U_e)
    chi_e = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, alpha_e, U_e, Te_values, Te_values)

# --- Multi-ion calculation ---
    M_i_total = 0
    chi_i_total = 0
    for ion in ion_species:
        if ion["fraction"] > 0:
            mi = ion["mass"]
            ni = ion["density"]
            frac = ion["fraction"]
            vth_i = calculate_thermal_velocity(kB, Ti, mi)
            Oc_i = calculate_cyclotron_frequency(1, e, B, mi)
            rho_i = calculate_average_gyroradius(vth_i, Oc_i)
            lambda_Di = calculate_debye_length(Ti, ni, epsilon_0, kB, e)
            alpha_i = calculate_alpha(k_total, lambda_Di)
            U_i = calculate_collisional_term(nu_i, k_parallel, vth_i, k_perpendicular, rho_i, n_terms, omega_values, Oc_i)
            M_i = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, U_i)
            chi_i = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, alpha_i, U_i, Ti, Ti)
            M_i_total += frac * M_i
            chi_i_total += frac * chi_i
                
    spectra = calcSpectra(M_i_total, M_e, chi_i_total, chi_e)

    def run_marco_spectrum():
        Ne = 2E11
        Bm = 3.60e-5
        Te = 500
        Ti = 500
        ion_mass = 16
        ion_comp = 1

        plasma = plasma_param.Plasma(Ne, Bm, Te, Ti, ion_mass, ion_comp)

        # Print Marco's ion density
        # print("Marco's Ion Density (Ni):", plasma.Ni)

        N = int(100E3)
        fs = 12E6
        frad = 440E6
        lambdaB = phys_cons.c / frad / 2
        aspdeg = 1
        modgy = 1
        modnu = 2

        isr_spc = isrSpectrum(N, fs, lambdaB, aspdeg, plasma, modgy=modgy, modnu=modnu)

        # Marco alpha (collective parameter) using Bragg wavenumber kB = 2*pi/lambdaB
        # Debye length (electron) based on plasma Ne and Te
        eps0 = 8.854187817e-12
        kB_const = 1.380649e-23
        e_charge = 1.602e-19
        lambda_De = np.sqrt(eps0 * kB_const * Te / (Ne * e_charge**2))
        k_bragg = 2 * np.pi / lambdaB
        alpha_marco = 1.0 / (k_bragg * lambda_De)

        return (isr_spc.f / 1E3), isr_spc.spc, alpha_marco

    # Generate Marcos' spectra
    marcos_freq, marcos_spectrum, alpha_marco = run_marco_spectrum()
    marcos_freq_mhz = marcos_freq / 1000  # Convert kHz to MHz

    mhz_values = [-5, -3, -1, 0, 1, 3, 5]  # Frequencies in MHz

    for mhz in mhz_values:
        # Convert MHz to Hz and rad/s for Chirag's spectra
        omega_hz = mhz * 1e6
        omega_rad = omega_hz * 2 * np.pi

        # Interpolate Chirag's and Marco's spectra
        chirag_value = np.interp(omega_rad, omega_values, spectra)
        marcos_value = np.interp(mhz, marcos_freq_mhz, marcos_spectrum)

        # Print the results
        print(f"Frequency: {mhz} MHz")
        print(f"  Chirag's Spectra: {chirag_value}")
        print(f"  Marco's Spectra: {marcos_value}")

    # Print alpha values alongside the plotted spectra
    print(f"Chirag alpha (as used in this script): {alpha_chirag:.6e}")
    print(f"Marco alpha (Bragg k): {alpha_marco:.6e}")

    # Plot combined spectra with adjusted frequency scale
    plot_combined_spectra(
        omega_values,
        [spectra],
        (marcos_freq_mhz, marcos_spectrum),
        [Te_values],
        alpha_chirag=alpha_chirag,
        alpha_marco=alpha_marco,
    )
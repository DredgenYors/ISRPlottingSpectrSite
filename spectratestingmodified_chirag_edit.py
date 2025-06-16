import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ive, dawsn
import scipy.special as sp

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
    k = 2 * 2 * np.pi / lambda_wavelength #Calculate total wavenumber 
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
    # Define MHz regions
    regions = [
        # (start_MHz, end_MHz, num_points)
        (-6.0, -4.6, 100),    # far left
        (-4.5, -4.0, 1000),    # plasma line (left)
        (-3.9, -0.9, 100),    # between plasma and gyro (left)
        (-0.8, -0.3, 1000),    # gyro line (left)
        (-0.2, -0.1, 100),    # between gyro and ion (left)
        (-0.1, 0.1, 1001),    # ion line (center)
        (0.1, 0.2, 100),      # between ion and gyro (right)
        (0.3, 0.8, 1000),      # gyro line (right)
        (0.9, 3.9, 100),      # between gyro and plasma (right)
        (4.0, 4.5, 1000),      # plasma line (right)
        (4.6, 6.0, 100),      # far right
    ]

    omega_hz = []
    for start, end, num in regions:
        # endpoint=False except for the last region to avoid duplicate points
        endpoint = False if end != regions[-1][1] else True
        omega_hz.append(np.linspace(start * 1e6, end * 1e6, num, endpoint=endpoint))
    omega_hz = np.concatenate(omega_hz)
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
    U = np.zeros_like(omega) + 1j * 0.0 #create an array of complex 0s the same size as the omega values array
    k_rho_sq = (k_perp ** 2) * (rho ** 2) #calculate the k_rho_sq constant
    
    for i in range(-n, n + 1): #iterate from -n to n + 1
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth) #calculate the yn constant, using the current value of i 
                                                        #omega is adjusted by subtracting the current cyclotron frequency and the collisional damping
                                                        #omega is normalized by dividing by the parallel wavenumber and thermal velocity
        
        U += sp.ive(i, k_rho_sq) * (2 * sp.dawsn(yn) + 1j * np.sqrt(np.pi) * np.exp(-yn ** 2)) #accumulate the components
    U = U * 1j * nu / (k_par * vth) #scale U by the collisional damping factor  
    return U

def calculate_modified_distribution(omega, k_par, k_perp, vth, n, rho, Oc, nu, U):
    """
    Calculate the modified distribution using the updated formula with Dawson function.
    """
    M = np.zeros_like(omega) + 1j * 0.0
    k_rho_sq = (k_perp ** 2) * (rho ** 2)
    
    # Term involving |U|^2
    term_one = -(abs(U) ** 2) / (nu * (abs(1 + U) ** 2))
    
    # Prefactor involving k_parallel and vth
    term_two = 1 / (k_par * vth * (abs(1 + U) ** 2))
    
    # Summation term
    for i in range(-n, n + 1):
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        
        # Real part of the exponential term
        exp_re = np.real(np.exp(-yn**2))
        
        # Imaginary part (Dawson function term)
        imag_da = 2*np.imag(dawsn(yn))
        
        # Bessel function and exponential term
        bessel_term = ive(i, k_rho_sq)
        # exp_term = np.exp(-k_rho_sq) # not included, called already in ive 
        
        # Combine all terms in the summation
        M += bessel_term * (np.sqrt(np.pi) * exp_re + imag_da)
    
    # Combine all terms into M_s
    M_s = term_one + term_two * M
    return np.real(M_s)

def calculate_debye_length(Te, ne, epsilon_0, kB, e):
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
    lambda_De = np.sqrt(epsilon_0 * kB * Te / (ne * e**2)) #calculate the debye length
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
    chi = np.zeros_like(omega) + 1j * 0.0 #create an array of complex 0s, the same size as the omega values array
    k_rho_sq = (k_perp ** 2) * (rho ** 2) #calculate the k_rho_sq constant
    term_one = (omega - 1j * nu) / (k_par * vth) #calculate the first term
    term_four = (alpha ** 2) / (1 + U) #calculate the fourth term
    term_five = Te / Ts #calculate the fifth term
    for i in range(-n, n + 1): #iterate from -n to n + 1
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth) #calculate the yn constant, using the current value of i 
                                                        #omega is adjusted by subtracting the current cyclotron frequency and the collisional damping
                                                        #omega is normalized by dividing by the parallel wavenumber and thermal velocity 
        
        term_two = 2 * sp.dawsn(yn) #calculate the second term using the current value of yn
        term_three = 1j * np.sqrt(np.pi) * np.exp(-yn ** 2) #calculate the second term using the current value of yn
        chi += sp.ive(i, k_rho_sq) * (1 - term_one * (term_two + term_three)) #accumulate the terms
    return term_four * term_five * chi #return the remaining terms and the accumulated term

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

# Ensure the script only runs when executed directly, not when imported
if __name__ == "__main__":
    # Sample Data
    nu_i = .0000001  # Ion collision frequency in Hz
    nu_e = 0.0000001  # Electron collision frequency in Hz
    ni = ne = 2e11  # Ion and electron densities in m^-3
    mi = 2.65686e-26  # Ion mass (atomic oxygen) in kg
    m_e = 9.11e-31  # Electron mass [kg]
    B = 3.6e-5  # Magnetic field strength in Tesla
    theta = 80  # Scattering angle in degrees
    Te_values = [500, 1500, 2500, 3500]  # Electron temperatures in Kelvin

    epsilon_0 = 8.854187817e-12  # Vacuum permittivity [F/m]
    kB = 1.380649e-23  # Boltzmann constant [J/K]
    e = 1.602e-19  # Elementary charge [C]

    # Define your ion species here
    ion_species = [
        {"name": "O+",   "fraction": 0.488, "density": 2.03e5, "mass": 2.65686e-26},
        {"name": "N+",   "fraction": 0.032, "density": 1.33e4, "mass": 2.32587e-26},
        {"name": "H+",   "fraction": 0.456, "density": 1.89e5, "mass": 1.67262e-27},
        {"name": "HE+",  "fraction": 0.024, "density": 9.96e3, "mass": 6.64648e-27},
        {"name": "O2+",  "fraction": 0.0,   "density": 0.0,    "mass": 5.31372e-26},
        {"name": "NO+",  "fraction": 0.0,   "density": 0.0,    "mass": 2.4828e-26}
    ]

    n_terms = 2000

    k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(0.69719, theta)
    
    spectra_list = []
    for T in Te_values:
        Te = Ti = T
        vth_i = calculate_thermal_velocity(kB, Ti, mi)
        vth_e = calculate_thermal_velocity(kB, Te, m_e)
        Oc_i = calculate_cyclotron_frequency(1, e, B, mi)
        Oc_e = calculate_cyclotron_frequency(1, e, B, m_e)
        rho_i = calculate_average_gyroradius(vth_i, Oc_i)
        rho_e = calculate_average_gyroradius(vth_e, Oc_e)
        lambda_De = calculate_debye_length(Te, ne, epsilon_0, kB, e)
        lambda_Di = calculate_debye_length(Te, ni, epsilon_0, kB, e)
        alpha_e = calculate_alpha(k_total, lambda_De)
        alpha_i = calculate_alpha(k_total, lambda_Di)
        c = calculate_sound_speed(kB, Te, Ti, mi)
        omega_values = calculate_omega_values(k_total, c)
        U_i = calculate_collisional_term(nu_i, k_parallel, vth_i, k_perpendicular, rho_i, n_terms, omega_values, Oc_i)
        U_e = calculate_collisional_term(nu_e, k_parallel, vth_e, k_perpendicular, rho_e, n_terms, omega_values, Oc_e)
        M_i = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, U_i)
        M_e = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, U_e)
        chi_i = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, alpha_i, U_i, Ti, Ti)
        chi_e = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, alpha_e, U_e, Te, Te)

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
                
        spectra = calcSpectra(M_i, M_e, chi_i, chi_e)
        spectra_list.append(spectra)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560)]
    labels = [f'Te = {T} K' for T in Te_values]
    
    for i, spectra in enumerate(spectra_list):
        ax.plot(omega_values / (2 * np.pi * 1e6), spectra, color=colors[i], label=labels[i])
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Spectra')
    ax.set_title('Backscatter Spectra')  # Updated title
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    # ax.set_ylim(1e-14, 1e-4)
    ax.set_xlim(-6, 6)
    plt.tight_layout()
    plt.show(block=True)

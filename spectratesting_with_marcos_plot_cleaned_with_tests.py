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

# -----------------------------
# Regime + metric helpers
# -----------------------------
def debye_length_e(Te_K, ne_m3, eps0=8.854187817e-12, kB=1.380649e-23, e=1.602176634e-19):
    """Electron Debye length (meters)."""
    Te_K = float(Te_K)
    ne_m3 = float(ne_m3)
    return np.sqrt(eps0 * kB * Te_K / (ne_m3 * e**2))

def alpha_from_k_lambdaD(k, lambdaD):
    """alpha = 1/(k*lambdaD)."""
    return 1.0 / (k * lambdaD)

def classify_collective_regime(alpha, k_lambdaD=None):
    """
    Practical 3-zone classifier using k*lambda_D (or alpha).
    - non-collective: k*lambdaD >= 3  (alpha <= 0.333...)
    - collective:     k*lambdaD <= 0.3 (alpha >= 3.333...)
    - transition:     otherwise
    These thresholds are heuristic but very useful for automated tests.
    """
    if k_lambdaD is None:
        k_lambdaD = 1.0 / alpha

    if k_lambdaD >= 3.0:
        return "NON-COLLECTIVE (kλD ≥ 3)"
    if k_lambdaD <= 0.3:
        return "COLLECTIVE (kλD ≤ 0.3)"
    return "TRANSITION (0.3 < kλD < 3)"

def normalize_spectrum(S):
    S = np.asarray(S, dtype=float)
    S = np.maximum(S, 0.0)
    area = np.trapz(S, dx=1.0)
    if area == 0 or not np.isfinite(area):
        return S
    return S / area

def compute_fwhm(x, y):
    """
    Robust-ish FWHM using linear interpolation.
    Returns np.nan if it can't be computed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    ymax = np.max(y)
    if ymax <= 0:
        return np.nan
    half = 0.5 * ymax
    above = y >= half
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    i1, i2 = idx[0], idx[-1]

    # interpolate left crossing
    if i1 == 0:
        x1 = x[0]
    else:
        xL0, xL1 = x[i1-1], x[i1]
        yL0, yL1 = y[i1-1], y[i1]
        x1 = xL0 + (half - yL0) * (xL1 - xL0) / (yL1 - yL0 + 1e-30)

    # interpolate right crossing
    if i2 == len(x)-1:
        x2 = x[-1]
    else:
        xR0, xR1 = x[i2], x[i2+1]
        yR0, yR1 = y[i2], y[i2+1]
        x2 = xR0 + (half - yR0) * (xR1 - xR0) / (yR1 - yR0 + 1e-30)

    return float(abs(x2 - x1))

def compute_second_moment_width(x, y):
    """
    sqrt(E[x^2]) around 0 (assumes centered).
    For symmetric spectra, this correlates with width.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ysum = np.trapz(y, x)
    if ysum <= 0:
        return np.nan
    m2 = np.trapz(y * (x**2), x) / ysum
    return float(np.sqrt(m2))

def compute_dip_metric(x, y, center_exclude_hz=500.0):
    """
    Measures whether there's a central dip vs two side maxima:
    D = (avg(side_peaks)-center)/avg(side_peaks)
    Uses the max on each side excluding a small band around 0.
    Returns 0 if it can't find two side peaks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    center_idx = np.argmin(np.abs(x))
    center_val = y[center_idx]

    left_mask = x < -center_exclude_hz
    right_mask = x > center_exclude_hz
    if not np.any(left_mask) or not np.any(right_mask):
        return 0.0

    left_peak = np.max(y[left_mask])
    right_peak = np.max(y[right_mask])
    side_avg = 0.5 * (left_peak + right_peak)
    if side_avg <= 0:
        return 0.0
    return float((side_avg - center_val) / side_avg)

# -----------------------------
# Refactored runners (single cases)
# -----------------------------

def _normalize_ion_species_for_ne(ion_species, ne_m3):
    """Ensure ion species densities track ne_m3.

    The original (older) comparison script effectively kept Chirag and Marco
    in sync by deriving ion density from the same high-level inputs.

    In this newer script, if you pass a fixed `ion_species` list that contains
    a hard-coded `density`, then sweeping or changing `ne_m3` only updates the
    electron Debye length (and a few other electron-side quantities) while the
    ion term stays pinned to the old density. That breaks the "match params as
    much as possible" intent and can make the plots look "different" even when
    you believe you're using the same Ne.

    Rules:
      - If a species has no 'density' (or it's None), set density = ne_m3 * fraction.
      - If densities are provided but their sum != ne_m3, rescale them to sum to ne_m3.
    """
    if ion_species is None:
        return None

    species = []
    sum_frac = 0.0
    sum_den = 0.0
    missing_density = False
    for spc in ion_species:
        frac = float(spc.get("fraction", 0.0))
        den = spc.get("density", None)
        sum_frac += frac
        if den is None:
            missing_density = True
        else:
            sum_den += float(den)
        species.append({
            "mass": float(spc["mass"]),
            "fraction": frac,
            "density": None if den is None else float(den),
        })

    # If any densities missing, fill from fractions.
    if missing_density:
        # If fractions don't sum to 1, normalize them for density assignment.
        frac_norm = sum_frac if sum_frac > 0 else 1.0
        for spc in species:
            spc["density"] = float(ne_m3) * (spc["fraction"] / frac_norm)
        return species

    # Otherwise, rescale provided densities to sum to ne_m3 (if needed).
    if sum_den <= 0:
        # degenerate: fall back to fraction-based
        frac_norm = sum_frac if sum_frac > 0 else 1.0
        for spc in species:
            spc["density"] = float(ne_m3) * (spc["fraction"] / frac_norm)
        return species

    scale = float(ne_m3) / float(sum_den)
    if not np.isfinite(scale) or scale <= 0:
        return species

    # Only rescale if it's meaningfully different.
    if abs(scale - 1.0) > 1e-6:
        for spc in species:
            spc["density"] *= scale

    return species
def run_chirag_case(
    ne_m3=2e11,
    B_T=3.60e-5,
    Te_K=500,
    Ti_K=500,
    theta_deg=89,
    lambda_r_m=0.341,
    nu_e_hz=3.32424637,
    nu_i_hz=1364.30161374848,
    ion_species=None,
    n_terms=5,
    match_nu_to_marco=False,
    marco_ion_mass_amu=16,
    marco_ion_comp=1,
    marco_aspdeg=1,
    electron_only=False,
):
    """
    Returns: omega(rad/s), spectrum (arb), alpha_internal, alpha_backscatter, lambda_De
    """
    epsilon_0 = 8.854187817e-12
    kB = 1.380649e-23
    e = 1.602e-19
    m_e = 9.11e-31

    if ion_species is None:
        # Default: pure O+ (density tracks ne_m3)
        ion_species = [{"mass": 2.65686e-26, "fraction": 1.0, "density": None}]

    # IMPORTANT: keep ion densities consistent with ne_m3 for fair comparisons.
    ion_species = _normalize_ion_species_for_ne(ion_species, ne_m3)

    k_total, k_par, k_perp = calculate_wavenumber_components(lambda_r_m, theta_deg)

    # OPTIONAL (but matches your *original* comparison script):
    # use Marco's collision-frequency routine so both implementations share nu_e / nu_i.
    if match_nu_to_marco:
        try:
            plasma_tmp = plasma_param.Plasma(ne_m3, B_T, Te_K, Ti_K, marco_ion_mass_amu, marco_ion_comp)
            nu_e_hz = float(plasma_tmp.collision_frequency(0, 2, kB=k_total, aspdeg=marco_aspdeg)[0])
            nu_i_hz = float(plasma_tmp.collision_frequency([1], 2, kB=k_total, aspdeg=marco_aspdeg)[0])
        except Exception as ex:
            print(f"[warn] match_nu_to_marco=True but collision_frequency failed: {ex}")

    lambda_De = calculate_debye_length(Te_K, ne_m3, epsilon_0, kB, e)
    alpha_internal = calculate_alpha(k_total, lambda_De)  # matches original Chirag formulation

    # standard ISR backscatter k for regime labeling
    k_back = 4 * np.pi / lambda_r_m
    alpha_back = alpha_from_k_lambdaD(k_back, lambda_De)

    # use sound speed from your existing helper
    c_s = calculate_sound_speed(kB, Te_K, Ti_K, sum(ion["fraction"] * ion["mass"] for ion in ion_species))
    omega = calculate_omega_values(k_total, c_s)

    vth_e = calculate_thermal_velocity(kB, Te_K, m_e)
    Oc_e = calculate_cyclotron_frequency(-1, e, B_T, m_e)
    rho_e = calculate_average_gyroradius(vth_e, Oc_e)

    U_e = calculate_collisional_term(nu_e_hz, k_par, vth_e, k_perp, rho_e, n_terms, omega, Oc_e)
    M_e = calculate_modified_distribution(omega, k_par, k_perp, vth_e, n_terms, rho_e, Oc_e, nu_e_hz, U_e)
    chi_e = calculate_electric_susceptibility(
        omega, k_par, k_perp, vth_e, n_terms, rho_e, Oc_e, nu_e_hz,
        alpha_internal, U_e, Te_K, Te_K
    )

    if electron_only:
        M_i_total = 0.0
        chi_i_total = 0.0
    else:
        M_i_total = 0
        chi_i_total = 0
        for ion in ion_species:
            if ion["fraction"] > 0:
                mi = ion["mass"]
                ni = ion["density"]  # density already matched/scaled to ne_m3 above
                frac = ion["fraction"]
                vth_i = calculate_thermal_velocity(kB, Ti_K, mi)
                Oc_i = calculate_cyclotron_frequency(1, e, B_T, mi)
                rho_i = calculate_average_gyroradius(vth_i, Oc_i)
                lambda_Di = calculate_debye_length(Ti_K, ni, epsilon_0, kB, e)
                alpha_i = calculate_alpha(k_total, lambda_Di)
                U_i = calculate_collisional_term(nu_i_hz, k_par, vth_i, k_perp, rho_i, n_terms, omega, Oc_i)
                M_i = calculate_modified_distribution(omega, k_par, k_perp, vth_i, n_terms, rho_i, Oc_i, nu_i_hz, U_i)
                chi_i = calculate_electric_susceptibility(
                    omega, k_par, k_perp, vth_i, n_terms, rho_i, Oc_i, nu_i_hz,
                    alpha_i, U_i, Ti_K, Ti_K
                )
                M_i_total += frac * M_i
                chi_i_total += frac * chi_i

    S = calcSpectra(M_i_total, M_e, chi_i_total, chi_e)
    return omega, S, float(alpha_internal), float(alpha_back), float(lambda_De)

def run_marco_case(
    Ne_m3=2e11,
    B_T=3.60e-5,
    Te_K=500,
    Ti_K=500,
    ion_mass_amu=16,
    ion_comp=1,
    N=int(100E3),
    fs_hz=12E6,
    frad_hz=440E6,
    aspdeg=1,
    modgy=1,
    modnu=2,
):
    """
    Returns: f_hz, spectrum, alpha_bragg, alpha_backscatter, lambda_De, lambdaB
    alpha_backscatter here uses k=4*pi/lambda_r where lambda_r = c/frad (radar wavelength).
    """
    plasma = plasma_param.Plasma(Ne_m3, B_T, Te_K, Ti_K, ion_mass_amu, ion_comp)

    lambdaB = phys_cons.c / frad_hz / 2
    isr_spc = isrSpectrum(N, fs_hz, lambdaB, aspdeg, plasma, modgy=modgy, modnu=modnu)

    # Marco alpha (Bragg) based on kB = 2*pi/lambdaB
    eps0 = 8.854187817e-12
    kB_const = 1.380649e-23
    e_const = 1.602176634e-19
    lambda_De = np.sqrt(eps0 * kB_const * Te_K / (Ne_m3 * e_const**2))

    k_bragg = 2 * np.pi / lambdaB
    alpha_bragg = alpha_from_k_lambdaD(k_bragg, lambda_De)

    # Standard ISR backscatter alpha
    lambda_r = phys_cons.c / frad_hz
    k_back = 4 * np.pi / lambda_r
    alpha_back = alpha_from_k_lambdaD(k_back, lambda_De)

    f_hz = isr_spc.f  # Marco's frequency axis (typically Hz)
    S = isr_spc.spc
    return f_hz, S, float(alpha_bragg), float(alpha_back), float(lambda_De), float(lambdaB)

# -----------------------------
# Tests A - F
# -----------------------------
def print_case_header(label, alpha_back, k_lambdaD):
    print("\n" + "="*70)
    print(label)
    print(f"alpha(backscatter) = {alpha_back:.6e}  |  kλD = {k_lambdaD:.6e}")
    print("Regime:", classify_collective_regime(alpha_back, k_lambdaD=k_lambdaD))
    print("="*70)

def metrics_summary(x_hz, S):
    x_hz = np.asarray(x_hz, dtype=float)
    S = np.asarray(S, dtype=float)
    # normalize by area in x-domain
    S_pos = np.maximum(S, 0.0)
    area = np.trapz(S_pos, x_hz)
    if area > 0:
        Sn = S_pos / area
    else:
        Sn = S_pos

    fwhm = compute_fwhm(x_hz, Sn)
    w2 = compute_second_moment_width(x_hz, Sn)
    dip = compute_dip_metric(x_hz, Sn, center_exclude_hz=max(200.0, 0.001*(np.max(np.abs(x_hz))+1)))
    return {"fwhm_hz": fwhm, "sigma_hz": w2, "dip_metric": dip}

def test_A_alpha_sweep_ne(ne_values, base_params, show_every=2):
    """
    Sweep Ne (log scale) and compute metrics vs alpha for both Chirag and Marco.
    """
    rows = []
    for i, ne in enumerate(ne_values):
        # Chirag
        omega, Sch, a_int, a_back, lambdaDe = run_chirag_case(
            ne_m3=ne,
            B_T=base_params["B_T"],
            Te_K=base_params["Te_K"],
            Ti_K=base_params["Ti_K"],
            theta_deg=base_params["theta_deg"],
            lambda_r_m=base_params["lambda_r_m"],
            nu_e_hz=base_params["nu_e_hz"],
            nu_i_hz=base_params["nu_i_hz"],
            ion_species=base_params["ion_species"],
            n_terms=base_params["n_terms"],
            match_nu_to_marco=base_params.get("match_nu_to_marco", False),
            marco_ion_mass_amu=base_params.get("ion_mass_amu", 16),
            marco_ion_comp=base_params.get("ion_comp", 1),
            marco_aspdeg=base_params.get("marco_aspdeg", 1),
        )
        f_ch = omega/(2*np.pi)  # Hz
        m_ch = metrics_summary(f_ch, Sch)
        k_lambdaD_ch = 1.0/a_back

        # Marco
        f_m, Sm, a_bragg_m, a_back_m, lambdaDe_m, lambdaB = run_marco_case(
            Ne_m3=ne,
            B_T=base_params["B_T"],
            Te_K=base_params["Te_K"],
            Ti_K=base_params["Ti_K"],
            ion_mass_amu=base_params["ion_mass_amu"],
            ion_comp=base_params["ion_comp"],
            N=base_params["marco_N"],
            fs_hz=base_params["marco_fs_hz"],
            frad_hz=base_params["marco_frad_hz"],
            aspdeg=base_params["marco_aspdeg"],
            modgy=base_params["marco_modgy"],
            modnu=base_params["marco_modnu"],
        )
        m_m = metrics_summary(f_m, Sm)
        k_lambdaD_m = 1.0/a_back_m

        rows.append((ne, a_back, m_ch["sigma_hz"], m_ch["dip_metric"], a_back_m, m_m["sigma_hz"], m_m["dip_metric"]))

        # Optional: show example plots for a subset
        if show_every and (i % show_every == 0 or i == len(ne_values)-1):
            # overlay quick plot
            plt.figure()
            plt.plot(f_ch/1e6, Sch, label=f"Chirag (Ne={ne:.2e})")
            plt.plot(f_m/1e6, Sm, label=f"Marco  (Ne={ne:.2e})")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Spectral Power (arb)")
            plt.title(f"Test A: Ne sweep | alpha_back Chirag={a_back:.2e}, Marco={a_back_m:.2e}")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Print summary
    print("\nTest A summary (Ne sweep)")
    print("Ne      | a_back(C)  sigma(C)[Hz] dip(C) | a_back(M)  sigma(M)[Hz] dip(M)")
    for r in rows:
        ne, aC, sC, dC, aM, sM, dM = r
        print(f"{ne: .2e} | {aC: .2e}  {sC: .2e}   {dC: .2f} | {aM: .2e}  {sM: .2e}   {dM: .2f}")

def test_B_sweep_Te(Te_values, base_params, show_every=2):
    rows=[]
    for i, Te in enumerate(Te_values):
        omega, Sch, a_int, a_back, lambdaDe = run_chirag_case(
            ne_m3=base_params["ne_m3"],
            B_T=base_params["B_T"],
            Te_K=Te,
            Ti_K=base_params["Ti_K"],
            theta_deg=base_params["theta_deg"],
            lambda_r_m=base_params["lambda_r_m"],
            nu_e_hz=base_params["nu_e_hz"],
            nu_i_hz=base_params["nu_i_hz"],
            ion_species=base_params["ion_species"],
            n_terms=base_params["n_terms"],
            match_nu_to_marco=base_params.get("match_nu_to_marco", False),
            marco_ion_mass_amu=base_params.get("ion_mass_amu", 16),
            marco_ion_comp=base_params.get("ion_comp", 1),
            marco_aspdeg=base_params.get("marco_aspdeg", 1),
        )
        f_ch=omega/(2*np.pi)
        m_ch=metrics_summary(f_ch, Sch)

        f_m, Sm, a_bragg_m, a_back_m, lambdaDe_m, lambdaB = run_marco_case(
            Ne_m3=base_params["ne_m3"],
            B_T=base_params["B_T"],
            Te_K=Te,
            Ti_K=base_params["Ti_K"],
            ion_mass_amu=base_params["ion_mass_amu"],
            ion_comp=base_params["ion_comp"],
            N=base_params["marco_N"],
            fs_hz=base_params["marco_fs_hz"],
            frad_hz=base_params["marco_frad_hz"],
            aspdeg=base_params["marco_aspdeg"],
            modgy=base_params["marco_modgy"],
            modnu=base_params["marco_modnu"],
        )
        m_m=metrics_summary(f_m, Sm)

        rows.append((Te, m_ch["sigma_hz"], m_m["sigma_hz"], a_back, a_back_m))

        if show_every and (i % show_every == 0 or i == len(Te_values)-1):
            plt.figure()
            plt.plot(f_ch/1e6, Sch, label=f"Chirag (Te={Te:.0f}K)")
            plt.plot(f_m/1e6, Sm, label=f"Marco  (Te={Te:.0f}K)")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Spectral Power (arb)")
            plt.title(f"Test B: Te sweep | sigma Chirag={m_ch['sigma_hz']:.2e} Hz, Marco={m_m['sigma_hz']:.2e} Hz")
            plt.legend()
            plt.grid(True)
            plt.show()

    print("\nTest B summary (Te sweep; sigma should roughly track sqrt(Te) in electron-Doppler dominated cases)")
    print("Te[K] | sigma(C)[Hz] sigma(M)[Hz] | a_back(C) a_back(M)")
    for Te, sC, sM, aC, aM in rows:
        print(f"{Te:6.0f} | {sC: .2e}   {sM: .2e} | {aC: .2e} {aM: .2e}")

def test_C_ion_sensitivity(base_params):
    """
    Vary ion mass and Ti; see how much spectrum changes.
    """
    cases = [
        ("O+ baseline", 16, base_params["Ti_K"]),
        ("H+ light ion", 1, base_params["Ti_K"]),
        ("Heavy ion", 32, base_params["Ti_K"]),
        ("Hot ions", 16, base_params["Ti_K"]*4),
    ]

    # Baseline spectra
    omega0, S0, _, a_back0, _ = run_chirag_case(
        ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=base_params["Ti_K"],
        theta_deg=base_params["theta_deg"], lambda_r_m=base_params["lambda_r_m"],
        nu_e_hz=base_params["nu_e_hz"], nu_i_hz=base_params["nu_i_hz"], ion_species=base_params["ion_species"],
        n_terms=base_params["n_terms"],
        match_nu_to_marco=base_params.get("match_nu_to_marco", False),
        marco_ion_mass_amu=base_params.get("ion_mass_amu", 16),
        marco_ion_comp=base_params.get("ion_comp", 1),
        marco_aspdeg=base_params.get("marco_aspdeg", 1),
    )
    f0 = omega0/(2*np.pi)
    base_ch = normalize_spectrum(np.maximum(S0,0))

    fM0, SM0, _, a_backM0, _, _ = run_marco_case(
        Ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=base_params["Ti_K"],
        ion_mass_amu=base_params["ion_mass_amu"], ion_comp=base_params["ion_comp"],
        N=base_params["marco_N"], fs_hz=base_params["marco_fs_hz"], frad_hz=base_params["marco_frad_hz"],
        aspdeg=base_params["marco_aspdeg"], modgy=base_params["marco_modgy"], modnu=base_params["marco_modnu"]
    )
    base_m = normalize_spectrum(np.maximum(SM0,0))

    def corr(a,b):
        a=a-np.mean(a); b=b-np.mean(b)
        den=np.sqrt(np.sum(a*a)*np.sum(b*b))+1e-30
        return float(np.sum(a*b)/den)

    print("\nTest C: ion sensitivity (correlation vs baseline; closer to 1 means less sensitivity)")
    for label, ion_mass, Ti in cases:
        # Chirag: approximate by changing ion mass in ion_species
        # density=None => automatically tied to ne_m3 (prevents accidental mismatches)
        ion_species = [{"mass": ion_mass*1.66053906660e-27, "fraction": 1.0, "density": None}]
        omega, S, _, a_back, _ = run_chirag_case(
            ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=Ti,
            theta_deg=base_params["theta_deg"], lambda_r_m=base_params["lambda_r_m"],
            nu_e_hz=base_params["nu_e_hz"], nu_i_hz=base_params["nu_i_hz"], ion_species=ion_species,
            n_terms=base_params["n_terms"],
            match_nu_to_marco=base_params.get("match_nu_to_marco", False),
            marco_ion_mass_amu=base_params.get("ion_mass_amu", 16),
            marco_ion_comp=base_params.get("ion_comp", 1),
            marco_aspdeg=base_params.get("marco_aspdeg", 1),
        )
        ch = normalize_spectrum(np.maximum(S,0))
        # resample to baseline grid if different
        ch_i = np.interp(np.arange(len(base_ch)), np.linspace(0,len(base_ch)-1,len(ch)), ch)
        corr_ch = corr(base_ch, ch_i)

        # Marco: change ion_mass_amu
        fM, SM, _, _, _, _ = run_marco_case(
            Ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=Ti,
            ion_mass_amu=ion_mass, ion_comp=base_params["ion_comp"],
            N=base_params["marco_N"], fs_hz=base_params["marco_fs_hz"], frad_hz=base_params["marco_frad_hz"],
            aspdeg=base_params["marco_aspdeg"], modgy=base_params["marco_modgy"], modnu=base_params["marco_modnu"]
        )
        m = normalize_spectrum(np.maximum(SM,0))
        m_i = np.interp(np.arange(len(base_m)), np.linspace(0,len(base_m)-1,len(m)), m)
        corr_m = corr(base_m, m_i)

        print(f"{label:12s} | Corr Chirag={corr_ch:.3f}  Corr Marco={corr_m:.3f}")

def test_D_windowing(base_params):
    """
    Plot the same spectra at different frequency windows.
    """
    omega, Sch, _, a_back, _ = run_chirag_case(
        ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=base_params["Ti_K"],
        theta_deg=base_params["theta_deg"], lambda_r_m=base_params["lambda_r_m"],
        nu_e_hz=base_params["nu_e_hz"], nu_i_hz=base_params["nu_i_hz"], ion_species=base_params["ion_species"],
        n_terms=base_params["n_terms"],
    )
    f_ch = omega/(2*np.pi)

    f_m, Sm, _, a_back_m, _, _ = run_marco_case(
        Ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=base_params["Ti_K"],
        ion_mass_amu=base_params["ion_mass_amu"], ion_comp=base_params["ion_comp"],
        N=base_params["marco_N"], fs_hz=base_params["marco_fs_hz"], frad_hz=base_params["marco_frad_hz"],
        aspdeg=base_params["marco_aspdeg"], modgy=base_params["marco_modgy"], modnu=base_params["marco_modnu"]
    )

    windows_mhz = [0.05, 0.2, 1.0, 6.0]
    for w in windows_mhz:
        plt.figure()
        plt.plot(f_ch/1e6, Sch, label="Chirag")
        plt.plot(f_m/1e6, Sm, label="Marco")
        plt.xlim(-w, w)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Spectral Power (arb)")
        plt.title(f"Test D: Window ±{w} MHz | alpha_back Chirag={a_back:.2e}, Marco={a_back_m:.2e}")
        plt.legend(); plt.grid(True)
        plt.show()

def test_E_electron_only(base_params):
    """
    Diagnostic hack: compare 'electron-only' approximations.
    Chirag: true electron-only switch (removes ion term).
    Marco: approximate by making ions very sparse (Ne fixed, but Ti/ion_mass changes don't fully remove ions in that model).
    This test is still useful to see if Marco can shed ion-line dominated structure.
    """
    omega_e, Sch_e, _, a_back, _ = run_chirag_case(
        ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=base_params["Ti_K"],
        theta_deg=base_params["theta_deg"], lambda_r_m=base_params["lambda_r_m"],
        nu_e_hz=base_params["nu_e_hz"], nu_i_hz=base_params["nu_i_hz"], ion_species=base_params["ion_species"],
        n_terms=base_params["n_terms"], electron_only=True
    )
    f_ch = omega_e/(2*np.pi)

    # Marco approximation: increase ion_mass and set Ti tiny to reduce ion dynamics visibility
    f_m, Sm, _, a_back_m, _, _ = run_marco_case(
        Ne_m3=base_params["ne_m3"], B_T=base_params["B_T"], Te_K=base_params["Te_K"], Ti_K=1.0,
        ion_mass_amu=200, ion_comp=base_params["ion_comp"],
        N=base_params["marco_N"], fs_hz=base_params["marco_fs_hz"], frad_hz=base_params["marco_frad_hz"],
        aspdeg=base_params["marco_aspdeg"], modgy=base_params["marco_modgy"], modnu=base_params["marco_modnu"]
    )

    plt.figure()
    plt.plot(f_ch/1e6, Sch_e, label="Chirag (electron-only)")
    plt.plot(f_m/1e6, Sm, label="Marco (ion suppressed approx)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Spectral Power (arb)")
    plt.title(f"Test E: electron-only diagnostic | alpha_back Chirag={a_back:.2e}, Marco={a_back_m:.2e}")
    plt.legend(); plt.grid(True)
    plt.show()

def test_F_regression_cases(base_params):
    """
    Three golden cases: collective, transition, non-collective stress.
    """
    cases = [
        ("Collective-ish (high Ne)",  base_params["ne_m3"]*50, base_params["Te_K"], base_params["Ti_K"]),
        ("Transition-ish",            base_params["ne_m3"],    base_params["Te_K"], base_params["Ti_K"]),
        ("Non-collective stress (low Ne)", base_params["ne_m3"]/200, base_params["Te_K"], base_params["Ti_K"]),
    ]

    for label, ne, Te, Ti in cases:
        omega, Sch, _, a_back, _ = run_chirag_case(
            ne_m3=ne, B_T=base_params["B_T"], Te_K=Te, Ti_K=Ti,
            theta_deg=base_params["theta_deg"], lambda_r_m=base_params["lambda_r_m"],
            nu_e_hz=base_params["nu_e_hz"], nu_i_hz=base_params["nu_i_hz"],
            ion_species=[{"mass": 16*1.66053906660e-27, "fraction": 1.0, "density": ne}],
            n_terms=base_params["n_terms"],
        )
        f_ch=omega/(2*np.pi)
        met_ch=metrics_summary(f_ch, Sch)

        f_m, Sm, _, a_back_m, _, _ = run_marco_case(
            Ne_m3=ne, B_T=base_params["B_T"], Te_K=Te, Ti_K=Ti,
            ion_mass_amu=base_params["ion_mass_amu"], ion_comp=base_params["ion_comp"],
            N=base_params["marco_N"], fs_hz=base_params["marco_fs_hz"], frad_hz=base_params["marco_frad_hz"],
            aspdeg=base_params["marco_aspdeg"], modgy=base_params["marco_modgy"], modnu=base_params["marco_modnu"],
        )
        met_m=metrics_summary(f_m, Sm)

        print_case_header(label, a_back, 1.0/a_back)
        print(f"Chirag metrics: sigma={met_ch['sigma_hz']:.2e} Hz, dip={met_ch['dip_metric']:.2f}, fwhm={met_ch['fwhm_hz']:.2e} Hz")
        print(f"Marco  metrics: sigma={met_m['sigma_hz']:.2e} Hz, dip={met_m['dip_metric']:.2f}, fwhm={met_m['fwhm_hz']:.2e} Hz")

        plt.figure()
        plt.plot(f_ch/1e6, Sch, label=f"Chirag (α_back={a_back:.2e})")
        plt.plot(f_m/1e6, Sm, label=f"Marco  (α_back={a_back_m:.2e})")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Spectral Power (arb)")
        plt.title(f"Test F: {label}")
        plt.legend(); plt.grid(True)
        plt.show()

# -----------------------------
# Main entry: choose test
# -----------------------------
def build_default_params():
    # Mirrors your original defaults
    return {
        "ne_m3": 2e11,
        "B_T": 3.60e-5,
        "Te_K": 500,
        "Ti_K": 500,
        "theta_deg": 89,
        "lambda_r_m": 0.341,
        "nu_e_hz": 3.32424637,
        "nu_i_hz": 1364.30161374848,
        # Matches the older comparison script (large n_terms for the summations).
        "n_terms": 2001,
        # If True, Chirag uses Marco's collision_frequency() outputs for nu_e and nu_i
        # so both implementations share the same collision-frequency model.
        "match_nu_to_marco": True,
        # IMPORTANT: set density=None so Chirag always re-matches ion density to the current ne_m3
        # (this is how the original comparison script behaved when you swept Ne).
        "ion_species": [{"mass": 2.65686e-26, "fraction": 1.0, "density": None}],  # O+ default
        # Marco defaults
        "ion_mass_amu": 16,
        "ion_comp": 1,
        "marco_N": int(100E3),
        "marco_fs_hz": 12E6,
        "marco_frad_hz": 440E6,
        "marco_aspdeg": 1,
        "marco_modgy": 1,
        "marco_modnu": 2,
    }

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compare Chirag vs Marco ISR spectra + collective/non-collective tests.")
    p.add_argument("--test", default="single", choices=["single", "A", "B", "C", "D", "E", "F"],
                   help="Which test to run: single (default), or A-F.")
    p.add_argument("--show_values", action="store_true", help="Print sample values at a few frequency indices (single mode).")
    args = p.parse_args()

    params = build_default_params()

    if args.test == "single":
        # --- Chirag ---
        omega, spectra, alpha_chirag_internal, alpha_chirag_back, lambdaDe = run_chirag_case(
            ne_m3=params["ne_m3"], B_T=params["B_T"], Te_K=params["Te_K"], Ti_K=params["Ti_K"],
            theta_deg=params["theta_deg"], lambda_r_m=params["lambda_r_m"],
            nu_e_hz=params["nu_e_hz"], nu_i_hz=params["nu_i_hz"],
	        ion_species=params["ion_species"], n_terms=params["n_terms"],
	        match_nu_to_marco=params.get("match_nu_to_marco", False),
	        marco_ion_mass_amu=params.get("ion_mass_amu", 16),
	        marco_ion_comp=params.get("ion_comp", 1),
	        marco_aspdeg=params.get("marco_aspdeg", 1),
        )

        # --- Marco ---
        f_m, Sm, alpha_marco_bragg, alpha_marco_back, lambdaDe_m, lambdaB = run_marco_case(
            Ne_m3=params["ne_m3"], B_T=params["B_T"], Te_K=params["Te_K"], Ti_K=params["Ti_K"],
            ion_mass_amu=params["ion_mass_amu"], ion_comp=params["ion_comp"],
            N=params["marco_N"], fs_hz=params["marco_fs_hz"], frad_hz=params["marco_frad_hz"],
            aspdeg=params["marco_aspdeg"], modgy=params["marco_modgy"], modnu=params["marco_modnu"]
        )

        f_ch = omega / (2*np.pi)  # Hz

        # Regime labels based on standard backscatter kλD
        regime_ch = classify_collective_regime(alpha_chirag_back)
        regime_m  = classify_collective_regime(alpha_marco_back)

        print("\nAlpha + regime summary")
        print(f"Chirag alpha_internal (k=2π/λ_r): {alpha_chirag_internal:.6e}")
        print(f"Chirag alpha_backscatter (k=4π/λ_r): {alpha_chirag_back:.6e}  -> {regime_ch}")
        print(f"Marco  alpha_bragg (k=2π/λ_B): {alpha_marco_bragg:.6e}")
        print(f"Marco  alpha_backscatter (k=4π/λ_radar): {alpha_marco_back:.6e}  -> {regime_m}")

        if args.show_values:
            # display a few sample points
            sample_idx = [0, len(f_ch)//2, -1]
            for i in sample_idx:
                print(f"Chirag f={f_ch[i]/1e6: .4f} MHz  S={spectra[i]}")
            sample_idx_m = [0, len(f_m)//2, -1]
            for i in sample_idx_m:
                print(f"Marco  f={f_m[i]/1e6: .4f} MHz  S={Sm[i]}")

        # Use your existing combined plotter (kept in file)
        plot_combined_spectra(
            omega,
            [spectra],
            (f_m/1e6, Sm),
            [params["Te_K"]],
            alpha_chirag=alpha_chirag_internal,
            alpha_marco=alpha_marco_bragg,
        )

        # Add a second print on the figure for regime text
        # (Your plot function already displays alphas; regime is printed in console.)

    elif args.test == "A":
        ne_vals = np.logspace(10.5, 13.0, 7)  # adjust range as needed
        test_A_alpha_sweep_ne(ne_vals, params, show_every=2)

    elif args.test == "B":
        Te_vals = [200, 500, 1000, 2000, 5000, 10000]
        test_B_sweep_Te(Te_vals, params, show_every=2)

    elif args.test == "C":
        test_C_ion_sensitivity(params)

    elif args.test == "D":
        test_D_windowing(params)

    elif args.test == "E":
        test_E_electron_only(params)

    elif args.test == "F":
        test_F_regression_cases(params)

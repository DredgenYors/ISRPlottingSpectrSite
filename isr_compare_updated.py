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
    theta = np.radians(theta_degrees)  # Convert theta value from degrees to radians
    k = 2 * np.pi / lambda_wavelength  # Calculate total wavenumber
    k_parallel = k * np.cos(theta)     # Parallel component
    k_perpendicular = k * np.sin(theta)  # Perpendicular component
    return k, k_parallel, k_perpendicular

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
    c = np.sqrt((5 / 3) * (kB * (Ti + Te) / mi))
    return c

def calculate_omega_values(k, c):
    """
    Parameters:
    - k (float): Wavenumber of the radar wave [rad/m].
    - c (float): Sound speed of the plasma [m/s].

    Returns:
    - omega_rad (np.ndarray): Doppler shifted angular frequency points [rad/s]
    """
    total_points = 50000
    start_freq_mhz = -6.0
    end_freq_mhz = 6.0
    omega_hz = np.linspace(start_freq_mhz * 1e6, end_freq_mhz * 1e6, total_points)
    omega_rad = omega_hz * 2 * np.pi
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
    vth = np.sqrt((2 * kB * T) / m)
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
    Oc = (Z * e * B) / m
    return Oc

def calculate_average_gyroradius(vth, Oc):
    """
    Parameters:
    - vth (float): Thermal velocity [m/s].
    - Oc (float): Cyclotron frequency [rad/s].

    Returns:
    - rho (float): Average gyroradius [m].
    """
    rho = vth / (np.sqrt(2) * Oc)
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
    - omega (np.ndarray): Doppler shifted angular frequency points [rad/s].
    - Oc (float): Cyclotron frequency [rad/s].

    Returns:
    - U (np.ndarray complex): Collisional Term.
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
    """
    lambda_De = np.sqrt(epsilon_0 * kB * Te / (ne_ * e**2))
    return lambda_De

def calculate_alpha(k, lambda_De):
    """
    alpha = 1 / (k * lambda_De)
    """
    alpha = 1 / (k * lambda_De)
    return alpha

def calculate_electric_susceptibility(omega, k_par, k_perp, vth, n, rho, Oc, nu, alpha, U, Te, Ts):
    """
    Returns complex electric susceptibility chi(omega).
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
    Compute ISR spectrum from M and chi terms.
    """
    epsilon = 1 + chi_e + chi_i
    spectrum = 2*np.abs(1-chi_e/epsilon)**2*M_e + 2*np.abs(chi_e/epsilon)**2*M_i
    return spectrum

def plot_combined_spectra(omega_values, chirag_spectra_list, marcos_spectra, Te_values, alpha_chirag=None, alpha_marco=None):
    """
    (Kept for compatibility, but in single-mode we now use an auto-window plot below.)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, spectra in enumerate(chirag_spectra_list):
        ax.plot(omega_values / (2 * np.pi * 1e6), spectra, label="NJIT Spectra", color='blue', linewidth=4)

    marcos_freq, marcos_spectrum = marcos_spectra
    ax.plot(marcos_freq, marcos_spectrum, label="Marco Spectra", color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Frequency (MHz)', fontsize=14)
    ax.set_ylabel('Spectra', fontsize=14)
    ax.set_title('Spectra Combined', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')

    # NOTE: This is the old fixed window:
    ax.set_xlim(-6, 6)

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

def classify_thomson_hard(alpha, k_lambdaD=None):
    """
    Hard-boundary classification at kλD = 1.
    """
    if k_lambdaD is None:
        k_lambdaD = 1.0 / alpha

    eps = 1e-12
    if k_lambdaD > 1.0 + eps:
        return "THOMSON / NON-COLLECTIVE (hard: kλD > 1)"
    if k_lambdaD < 1.0 - eps:
        return "COLLECTIVE (hard: kλD < 1)"
    return "BOUNDARY (kλD ≈ 1)"

# -----------------------------
# Refactored runners (single cases)
# -----------------------------
def _normalize_ion_species_for_ne(ion_species, ne_m3):
    """Ensure ion species densities track ne_m3."""
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

    if missing_density:
        frac_norm = sum_frac if sum_frac > 0 else 1.0
        for spc in species:
            spc["density"] = float(ne_m3) * (spc["fraction"] / frac_norm)
        return species

    if sum_den <= 0:
        frac_norm = sum_frac if sum_frac > 0 else 1.0
        for spc in species:
            spc["density"] = float(ne_m3) * (spc["fraction"] / frac_norm)
        return species

    scale = float(ne_m3) / float(sum_den)
    if not np.isfinite(scale) or scale <= 0:
        return species

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
        ion_species = [{"mass": 2.65686e-26, "fraction": 1.0, "density": None}]

    ion_species = _normalize_ion_species_for_ne(ion_species, ne_m3)

    k_total, k_par, k_perp = calculate_wavenumber_components(lambda_r_m, theta_deg)

    if match_nu_to_marco:
        try:
            plasma_tmp = plasma_param.Plasma(ne_m3, B_T, Te_K, Ti_K, marco_ion_mass_amu, marco_ion_comp)
            nu_e_hz = float(plasma_tmp.collision_frequency(0, 2, kB=k_total, aspdeg=marco_aspdeg)[0])
            nu_i_hz = float(plasma_tmp.collision_frequency([1], 2, kB=k_total, aspdeg=marco_aspdeg)[0])
        except Exception as ex:
            print(f"[warn] match_nu_to_marco=True but collision_frequency failed: {ex}")

    lambda_De = calculate_debye_length(Te_K, ne_m3, epsilon_0, kB, e)
    alpha_internal = calculate_alpha(k_total, lambda_De)

    k_back = 4 * np.pi / lambda_r_m
    alpha_back = alpha_from_k_lambdaD(k_back, lambda_De)

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
                ni = ion["density"]
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
    """
    plasma = plasma_param.Plasma(Ne_m3, B_T, Te_K, Ti_K, ion_mass_amu, ion_comp)

    lambdaB = phys_cons.c / frad_hz / 2
    isr_spc = isrSpectrum(N, fs_hz, lambdaB, aspdeg, plasma, modgy=modgy, modnu=modnu)

    eps0 = 8.854187817e-12
    kB_const = 1.380649e-23
    e_const = 1.602176634e-19
    lambda_De = np.sqrt(eps0 * kB_const * Te_K / (Ne_m3 * e_const**2))

    k_bragg = 2 * np.pi / lambdaB
    alpha_bragg = alpha_from_k_lambdaD(k_bragg, lambda_De)

    lambda_r = phys_cons.c / frad_hz
    k_back = 4 * np.pi / lambda_r
    alpha_back = alpha_from_k_lambdaD(k_back, lambda_De)

    f_hz = isr_spc.f
    S = isr_spc.spc
    return f_hz, S, float(alpha_bragg), float(alpha_back), float(lambda_De), float(lambdaB)

def build_default_params(profile="matched"):
    if profile == "sheet_original":
        return {
            "ne_m3": 2e11,
            "B_T": 3.60e-5,
            "Te_K": 500,
            "Ti_K": 500,
            "theta_deg": 0,
            "lambda_r_m": 0.341,
            "nu_e_hz": 3.32424637,
            "nu_i_hz": 1364.30161374848,
            "n_terms": 2001,
            "ion_species": [{"mass": 2.66e-26, "fraction": 1.0, "density": 2e11}],
            "ion_mass_amu": 16,
            "ion_comp": 1,
            "marco_N": int(100E3),
            "marco_fs_hz": 12E6,
            "marco_frad_hz": 440E6,
            "marco_aspdeg": 90,
            "marco_modgy": 1,
            "marco_modnu": 2,
            "match_nu_to_marco": False,
        }

    return {
        "ne_m3": 2e11,
        "B_T": 3.60e-5,
        "Te_K": 500,
        "Ti_K": 500,
        "theta_deg": 0,
        "lambda_r_m": 0.682,
        "nu_e_hz": 3.32424637,
        "nu_i_hz": 1364.30161374848,
        "n_terms": 2001,
        "ion_species": [{"mass": 2.65686e-26, "fraction": 1.0, "density": None}],
        "ion_mass_amu": 16,
        "ion_comp": 1,
        "marco_N": int(200E3),
        "marco_fs_hz": 200E6,
        "marco_frad_hz": 1e11,
        "marco_aspdeg": 90,
        "marco_modgy": 1,
        "marco_modnu": 2,
        "match_nu_to_marco": True,
    }

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compare Chirag vs Marco ISR spectra (single-mode auto window).")
    p.add_argument("--test", default="single", choices=["single"],
                   help="This file version implements single-mode changes only.")
    p.add_argument("--profile", default="matched", choices=["sheet_original", "matched"],
                   help="Parameter preset.")
    args = p.parse_args()

    params = build_default_params(profile=args.profile)

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

    # -----------------------------
    # Step (2): print axis + spectra sanity + samples
    # -----------------------------
    print("\nAlpha + regime summary")
    print(f"Chirag alpha_internal (k=2π/λ_r): {alpha_chirag_internal:.6e}")
    print(f"Chirag alpha_backscatter (k=4π/λ_r): {alpha_chirag_back:.6e}  -> {classify_thomson_hard(alpha_chirag_back)}")
    print(f"Marco  alpha_bragg (k=2π/λ_B): {alpha_marco_bragg:.6e}")
    print(f"Marco  alpha_backscatter (k=4π/λ_radar): {alpha_marco_back:.6e}  -> {classify_thomson_hard(alpha_marco_back)}")

    print("\n--- EE axis + spectra sanity ---")
    print(f"EE f range: {np.min(f_m)/1e6:.6f} to {np.max(f_m)/1e6:.6f} MHz")
    print(f"EE S range: min={np.min(Sm):.6e}, max={np.max(Sm):.6e}")

    print("\n--- Chirag axis + spectra sanity ---")
    print(f"Chirag f range: {np.min(f_ch)/1e6:.6f} to {np.max(f_ch)/1e6:.6f} MHz")
    print(f"Chirag S range: min={np.min(spectra):.6e}, max={np.max(spectra):.6e}")

    idxs_m = [0, len(f_m)//2, -1]
    print("\nEE samples (f_MHz, S):")
    for i in idxs_m:
        print(f"  {f_m[i]/1e6: .6f} MHz   {Sm[i]: .6e}")

    idxs_c = [0, len(f_ch)//2, -1]
    print("\nChirag samples (f_MHz, S):")
    for i in idxs_c:
        print(f"  {f_ch[i]/1e6: .6f} MHz   {spectra[i]: .6e}")

    # -----------------------------
    # Step (1): auto-set x-limits from EE data + plot
    # -----------------------------
    f_ch_mhz = f_ch / 1e6
    f_m_mhz = f_m / 1e6

    lo, hi = np.percentile(f_m_mhz, [0.5, 99.5])
    w = float(max(abs(lo), abs(hi))) if np.isfinite(lo) and np.isfinite(hi) else float(np.max(np.abs(f_m_mhz)))

    S_ch_plot = np.maximum(np.asarray(spectra, float), 1e-30)
    S_m_plot = np.maximum(np.asarray(Sm, float), 1e-30)

    plt.figure(figsize=(12, 8))
    plt.plot(f_ch_mhz, S_ch_plot, label="NJIT Spectra", linewidth=3)
    plt.plot(f_m_mhz, S_m_plot, label="Marco Spectra", linestyle="--", linewidth=2)

    plt.xlim(-w, w)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Spectral Power (arb)")
    plt.title(f"Single ({args.profile}): auto window ±{w:.3f} MHz from EE axis")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

    plt.text(
        0.02, 0.98,
        f"Chirag α_back = {alpha_chirag_back:.3e}\nMarco α_back  = {alpha_marco_back:.3e}",
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Import classic ISR spectrum code (Marco Milla) ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'EEApproach'))
import plasma_parameters as plasma_param
import physical_constants as phys_cons
from isr_spectrum import isrSpectrum

# --- Import ISRPlottingSpectrSite-main code (Chirag's) ---
from spectratestingmodified_chirag_edit import (
    calculate_wavenumber_components,
    calculate_thermal_velocity,
    calculate_cyclotron_frequency,
    calculate_average_gyroradius,
    calculate_debye_length,
    calculate_alpha,
    calculate_omega_values,
    calculate_collisional_term,
    calculate_modified_distribution,
    calculate_electric_susceptibility,
    calcSpectra
)

def run_marco_spectrum():
    Ne = 2E11
    Bm = 36e-6
    Te = 500
    Ti = 500
    ion_mass = 16
    ion_comp = 1

    plasma = plasma_param.Plasma(Ne, Bm, Te, Ti, ion_mass, ion_comp)
    N = int(100E3)
    fs = 12E6
    frad = 440E6
    lambdaB = phys_cons.c / frad / 2
    aspdeg = 30
    modgy = 1
    modnu = 2

    isr_spc = isrSpectrum(N, fs, lambdaB, aspdeg, plasma, modgy=modgy, modnu=modnu)
    return isr_spc.f / 1E3, isr_spc.spc

def run_chirag_spectrum():
    # Parameters (match as close as possible to Marco's)
    nu_i = 1e-7
    nu_e = 1e-7
    ni = 2e11
    ne_ = 2e11
    mi = 2.65686e-26  # O+ mass
    m_e = 9.11e-31
    B = 3.6e-5
    theta = 30
    Te = 500
    Ti = 500
    epsilon_0 = 8.854187817e-12
    kB = 1.380649e-23
    e = 1.602e-19
    n_terms = 2000

    # Wavelength for 430 MHz radar
    c_light = 3e8
    radar_freq = 430e6
    lambda_wavelength = c_light / radar_freq

    k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(lambda_wavelength, theta)
    vth_e = calculate_thermal_velocity(kB, Te, m_e)
    vth_i = calculate_thermal_velocity(kB, Ti, mi)
    Oc_e = calculate_cyclotron_frequency(1, e, B, m_e)
    Oc_i = calculate_cyclotron_frequency(1, e, B, mi)
    rho_e = calculate_average_gyroradius(vth_e, Oc_e)
    rho_i = calculate_average_gyroradius(vth_i, Oc_i)
    lambda_De = calculate_debye_length(Te, ne_, epsilon_0, kB, e)
    alpha = calculate_alpha(k_total, lambda_De)
    omega = calculate_omega_values(k_total, vth_i)

    U_i = calculate_collisional_term(nu_i, k_parallel, vth_i, k_perpendicular, rho_i, n_terms, omega, Oc_i)
    U_e = calculate_collisional_term(nu_e, k_parallel, vth_e, k_perpendicular, rho_e, 20, omega, Oc_e)
    M_i = calculate_modified_distribution(omega, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, U_i)
    M_e = calculate_modified_distribution(omega, k_parallel, k_perpendicular, vth_e, 20, rho_e, Oc_e, nu_e, U_e)
    chi_i = calculate_electric_susceptibility(omega, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, alpha, U_i, Te, Ti)
    chi_e = calculate_electric_susceptibility(omega, k_parallel, k_perpendicular, vth_e, 20, rho_e, Oc_e, nu_e, alpha, U_e, Te, Te)
    spectrum = calcSpectra(M_i, M_e, chi_i, chi_e)

    # Convert omega (rad/s) to frequency (kHz)
    freq_kHz = omega / (2 * np.pi * 1e3)
    return freq_kHz, spectrum

def align_spectra(f1, spc1, f2, spc2):
    # Interpolate both spectra onto a common frequency grid (intersection)
    f_min = max(np.min(f1), np.min(f2))
    f_max = min(np.max(f1), np.max(f2))
    num = min(len(f1), len(f2))
    f_common = np.linspace(f_min, f_max, num)
    spc1_interp = np.interp(f_common, f1, spc1)
    spc2_interp = np.interp(f_common, f2, spc2)
    return f_common, spc1_interp, spc2_interp

def feature_comparison(f, spc1, spc2):
    # Find peaks and their locations
    idx1 = np.argmax(spc1)
    idx2 = np.argmax(spc2)
    peak1 = spc1[idx1]
    peak2 = spc2[idx2]
    freq1 = f[idx1]
    freq2 = f[idx2]
    width1 = np.sum(spc1 > peak1/2) * (f[1] - f[0])
    width2 = np.sum(spc2 > peak2/2) * (f[1] - f[0])
    print("\n--- Feature Comparison ---")
    print(f"Marco: Peak = {peak1:.2e} at {freq1:.1f} kHz, FWHM ≈ {width1:.1f} kHz")
    print(f"Chirag: Peak = {peak2:.2e} at {freq2:.1f} kHz, FWHM ≈ {width2:.1f} kHz")

def tabular_comparison(f, spc1, spc2, points=[0, 500, 1000, 2000, 3000]):
    print("\n--- Tabular Comparison (selected frequencies) ---")
    print(f"{'Freq (kHz)':>10} | {'Marco (dB)':>12} | {'Chirag (dB)':>12} | {'Diff (dB)':>10}")
    for freq in points:
        idx = (np.abs(f - freq)).argmin()
        val1 = 10 * np.log10(spc1[idx])
        val2 = 10 * np.log10(spc2[idx])
        print(f"{f[idx]:10.1f} | {val1:12.2f} | {val2:12.2f} | {val1-val2:10.2f}")

def plot_comparison():
    f_marco, spc_marco = run_marco_spectrum()
    f_chirag, spc_chirag = run_chirag_spectrum()

    # ensure ascending frequency arrays and protect from zeros before plotting
    idx_m = np.argsort(f_marco); f_marco = f_marco[idx_m]; spc_marco = spc_marco[idx_m]
    idx_c = np.argsort(f_chirag); f_chirag = f_chirag[idx_c]; spc_chirag = spc_chirag[idx_c]
    floor = 1e-300
    spc_marco = np.maximum(spc_marco, floor)
    spc_chirag = np.maximum(spc_chirag, floor)

    # Overlayed spectra: plot the original (un-interpolated) outputs
    plt.figure(figsize=(12, 6))
    plt.plot(f_marco, 10 * np.log10(spc_marco), label="Marco's isrSpectrum ", alpha=0.8)
    plt.plot(f_chirag, 10 * np.log10(spc_chirag), label="Chirag's ISR Site ", alpha=0.8)
    plt.grid()
    plt.xlim([-6e3, 6e3])
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Spectra [dB]')
    plt.title('ISR Spectrum Comparison: EE Approach vs. ISR Site ')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Align spectra on a common frequency grid for difference/ratio/metrics
    f_common, spc_marco_interp, spc_chirag_interp = align_spectra(f_marco, spc_marco, f_chirag, spc_chirag)

    # Difference plot (on aligned grid)
    diff = 10 * np.log10(spc_marco_interp) - 10 * np.log10(spc_chirag_interp)
    plt.figure(figsize=(12, 4))
    plt.plot(f_common, diff, label='Difference (Marco - Chirag) [dB]')
    plt.grid()
    plt.xlim([-6e3, 6e3])
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Difference [dB]')
    plt.title('Spectra Difference (aligned grid)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Overlayed spectra: plot the original (raw linear units)
    plt.figure(figsize=(12, 6))
    plt.plot(f_marco, spc_marco, label="Marco's isrSpectrum (raw)", alpha=0.8)
    plt.plot(f_chirag, spc_chirag, label="Chirag's ISR Site (raw)", alpha=0.8)
    plt.grid()
    plt.xlim([-6e3, 6e3])
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Spectrum (linear units)')
    plt.title('ISR Spectrum Comparison: EE Approach vs. ISR Site (raw outputs)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Statistical metrics (on aligned diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    print("\n--- Statistical Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} dB")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} dB")
    print(f"Maximum Absolute Difference: {max_diff:.2f} dB")

    # Feature comparison (use aligned grid)
    feature_comparison(f_common, spc_marco_interp, spc_chirag_interp)

    # Tabular comparison at selected frequencies (use aligned grid)
    tabular_comparison(f_common, spc_marco_interp, spc_chirag_interp, points=[0, 500, 1000, 2000, 3000])

if __name__ == "__main__":
    plot_comparison()
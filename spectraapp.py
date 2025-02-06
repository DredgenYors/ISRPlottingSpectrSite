from flask import Flask, render_template, send_file
import matplotlib.pyplot as plt
import numpy as np
import io
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

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('spectrasite.html')

@app.route('/plot_page')
def plot_page():
    return render_template('plot_page.html')

@app.route('/plot')
def plot():
    # Define constants
    nu_i = .0000001  # Ion collision frequency in Hz
    nu_e = 0.0000001  # Electron collision frequency in Hz
    ni = ne = 2e11  # Ion and electron densities in m^-3
    mi = 2.65686e-26  # Ion mass (atomic oxygen) in kg
    m_e = 9.11e-31  # Electron mass [kg]
    B = 3.6e-5  # Magnetic field strength in Tesla
    theta = 60  # Scattering angle in degrees
    Te_values = [500, 1500, 2500, 3500]  # Electron temperatures in Kelvin
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity [F/m]
    kB = 1.380649e-23  # Boltzmann constant [J/K]
    e = 1.602e-19  # Elementary charge [C]
    n_terms = 2000

    # Calculate wavenumber components
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
        spectra = calcSpectra(M_i, M_e, chi_i, chi_e)
        spectra_list.append(spectra)

    # Plot spectra
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560)]
    labels = [f'Te = {T} K' for T in Te_values]
    for i, spectra in enumerate(spectra_list):
        ax.plot(omega_values / (2 * np.pi * 1e6), spectra, color=colors[i], label=labels[i])
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Spectra')
    ax.set_title('Full Spectra for Different Electron Temperatures (Te)')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_ylim(1e-14, 1e-4)
    ax.set_xlim(-6, 6)
    plt.tight_layout()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

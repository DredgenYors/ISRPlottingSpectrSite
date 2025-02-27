from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
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

# Default parameter values
user_values = {
    "nu_i": 1.0e-7,
    "nu_e": 1.0e-7,
    "ni": 2.0e11,
    "ne": 2.0e11,
    "mi": 2.65686e-26,
    "me": 9.11e-31,
    "B": 3.6e-5,
    "theta": 60,
    "Te": 500,
    "epsilon_0": 8.854187817e-12,
    "kB": 1.380649e-23,
    "e": 1.602e-19,
    "n_terms": 2000
}

@app.route('/')
def home():
    """ Render the main landing page: index.html """
    return render_template('index.html', user_values=user_values)

@app.route('/spectrasite')
def spectrasite():
    """ Redirect to the spectrasite page when 'Simulation' is clicked """
    return render_template('spectrasite.html')

@app.route('/plot_page')
def plot_page():
    """ Serve the existing plot page """
    return render_template('plot_page.html')

@app.route('/update_values', methods=['POST'])
def update_values():
    global user_values
    data = request.json
    for key in data:
        user_values[key] = float(data[key])
    return jsonify({"message": "Values updated"}), 200

@app.route('/plot')
def plot():
    nu_i = user_values["nu_i"]
    nu_e = user_values["nu_e"]
    ni = user_values["ni"]
    ne = user_values["ne"]
    mi = user_values["mi"]
    me = user_values["me"]
    B = user_values["B"]
    theta = user_values["theta"]
    Te = user_values["Te"]
    epsilon_0 = user_values["epsilon_0"]
    kB = user_values["kB"]
    e = user_values["e"]
    n_terms = int(user_values["n_terms"])
    
    k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(0.69719, theta)
    vth_i = calculate_thermal_velocity(kB, Te, mi)
    vth_e = calculate_thermal_velocity(kB, Te, me)
    Oc_i = calculate_cyclotron_frequency(1, e, B, mi)
    Oc_e = calculate_cyclotron_frequency(1, e, B, me)
    rho_i = calculate_average_gyroradius(vth_i, Oc_i)
    rho_e = calculate_average_gyroradius(vth_e, Oc_e)
    lambda_De = calculate_debye_length(Te, ne, epsilon_0, kB, e)
    alpha_e = calculate_alpha(k_total, lambda_De)
    c = calculate_sound_speed(kB, Te, Te, mi)
    omega_values = calculate_omega_values(k_total, c)
    
    U_i = calculate_collisional_term(nu_i, k_parallel, vth_i, k_perpendicular, rho_i, n_terms, omega_values, Oc_i)
    U_e = calculate_collisional_term(nu_e, k_parallel, vth_e, k_perpendicular, rho_e, n_terms, omega_values, Oc_e)
    M_i = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, U_i)
    M_e = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, U_e)
    chi_i = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, alpha_e, U_i, Te, Te)
    chi_e = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, alpha_e, U_e, Te, Te)
    
    spectra = calcSpectra(M_i, M_e, chi_i, chi_e)
    
    data = {
        "frequencies": (omega_values / (2 * np.pi * 1e6)).tolist(),
        "spectra": spectra.tolist()
    }
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)


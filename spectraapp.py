# ================================================================================
# IMPORTS AND DEPENDENCIES
# ================================================================================
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import matplotlib.pyplot as plt
import numpy as np
import io

# Import ISR calculation functions from the spectra module
from spectratestingmodified_chirag_edit import (
    calculate_wavenumber_components,    # Wave vector calculations
    calculate_thermal_velocity,         # Thermal velocity calculations
    calculate_cyclotron_frequency,      # Gyrofrequency calculations
    calculate_average_gyroradius,       # Gyroradius calculations
    calculate_debye_length,             # Debye length calculations
    calculate_alpha,                    # Alpha parameter calculations
    calculate_sound_speed,              # Ion acoustic speed calculations
    calculate_omega_values,             # Frequency array generation
    calculate_collisional_term,         # Collision frequency effects
    calculate_modified_distribution,    # Modified velocity distributions
    calculate_electric_susceptibility,  # Electric susceptibility calculations
    calcSpectra                         # Main spectra calculation function
)

# ================================================================================
# APPLICATION CONFIGURATION
# ================================================================================
app = Flask(__name__)

# ================================================================================
# PHYSICAL CONSTANTS
# ================================================================================
EPSILON_0 = 8.854187817e-12  # Permittivity of free space [F/m]
KB = 1.380649e-23           # Boltzmann constant [J/K]
E = 1.602e-19               # Elementary charge [C]

# ================================================================================
# DEFAULT PARAMETER VALUES
# ================================================================================
# Global default values used as fallback for all calculations
user_values = {
    # Collision frequencies
    "nu_i": 1.0e-7,          # Ion collision frequency [Hz]
    "nu_e": 1.0e-7,          # Electron collision frequency [Hz]
    
    # Plasma densities
    "ni": 2.0e11,            # Ion density [m^-3]
    "ne": 2.0e11,            # Electron density [m^-3]
    
    # Particle masses
    "mi": 2.65686e-26,       # Ion mass (O+) [kg]
    "me": 9.11e-31,          # Electron mass [kg]
    
    # Physical parameters
    "B": 3.6e-5,             # Magnetic field [T]
    "theta": 60,             # Scattering angle [degrees]
    "Te": 500,               # Electron temperature [K]
    "Ti": 500,               # Ion temperature [K]
    "frequency": 430e6,      # Radar frequency [Hz]
    
    # Physical constants
    "epsilon_0": EPSILON_0,  # Permittivity of free space [F/m]
    "kB": KB,                # Boltzmann constant [J/K]
    "e": E,                  # Elementary charge [C]
    
    # Numerical parameters
    "n_terms": 2001          # Number of frequency points for calculation
}

# ================================================================================
# PARAMETER PROFILES CONFIGURATION
# ================================================================================
# Pre-defined parameter sets for different ionospheric conditions
parameter_profiles = {
    "Plot 1": {  # Default/Standard conditions
        "nu_i": 1.0e-7, "nu_e": 1.0e-7, "ni": 2.0e11, "ne": 2.0e11,
        "mi": 2.65686e-26, "me": 9.11e-31, "B": 3.6e-5, "theta": 60,
        "Te": 500, "Ti": "500", "frequency": 430e6, 
        "epsilon_0": EPSILON_0, "kB": KB, "e": E, "n_terms": 2001
    },
    "Plot 2": {  # Enhanced conditions (higher collision frequencies, different temperatures)
        "nu_i": 5.0e-7, "nu_e": 5.0e-7, "ni": 1.0e11, "ne": 1.0e11,
        "mi": 4.0e-26, "me": 9.11e-31, "B": 4.0e-5, "theta": 45,
        "Te": 1500, "Ti": "500", "frequency": 430e6, 
        "epsilon_0": EPSILON_0, "kB": KB, "e": E, "n_terms": 2001
    },
    "Plot 3": {  # High collision, low density conditions
        "nu_i": 1.0e-6, "nu_e": 1.0e-6, "ni": 5.0e10, "ne": 5.0e10,
        "mi": 3.0e-26, "me": 9.11e-31, "B": 5.0e-5, "theta": 30,
        "Te": 2500, "Ti": "500", "frequency": 430e6, 
        "epsilon_0": EPSILON_0, "kB": KB, "e": E, "n_terms": 2001
    }
}

# ================================================================================
# ROUTE HANDLERS - PAGE RENDERING
# ================================================================================
@app.route('/')
def home():
    """
    Render the main landing page: index.html
    Serves as the entry point to the application.
    """
    return render_template('index.html', user_values=user_values)

@app.route('/spectrasite')
def spectrasite():
    """
    Redirect to the spectrasite page when 'Simulation' is clicked.
    Serves the main simulation interface page.
    """
    return render_template('spectrasite.html')

@app.route('/plot_page')
def plot_page():
    """
    Serve the existing plot page.
    This is the main interactive plotting interface.
    """
    return render_template('plot_page.html')

# ================================================================================
# API ENDPOINTS - PARAMETER MANAGEMENT
# ================================================================================
@app.route('/update_values', methods=['POST'])
def update_values():
    """
    Update global parameter values from client requests.
    Handles both individual parameters and ion species arrays.
    
    Expected JSON format:
    {
        "parameter_name": value,
        "ion_species": [array of ion species data]
    }
    """
    global user_values
    data = request.json
    
    # Process each parameter in the request
    for key in data:
        if key == "ion_species":
            # Handle ion species array separately
            user_values["ion_species"] = data["ion_species"]
        else:
            # Convert numeric parameters to float
            user_values[key] = float(data[key])
    
    return jsonify({"message": "Values updated"}), 200

# ================================================================================
# PROFILE MANAGEMENT ENDPOINTS
# ================================================================================
@app.route('/set_profile', methods=['POST'])
def set_profile():
    """
    Apply a predefined parameter profile to current values.
    Updates the global user_values with profile data.
    
    Expected JSON: {"profile": "Plot 1|Plot 2|Plot 3"}
    """
    global user_values
    data = request.json
    profile_name = data.get("profile")
    
    # Get profile data and handle errors
    profile_data, error = get_profile_data(profile_name)
    if error:
        return jsonify(error), 400
    
    # Update global values with profile data
    user_values.update(profile_data)
    return jsonify({"message": f"Profile {profile_name} applied"}), 200

@app.route('/get_profile', methods=['POST'])
def get_profile():
    """
    Retrieve and apply profile data for client-side parameter updates.
    Similar to set_profile but designed for frontend profile switching.
    
    Expected JSON: {"profile": "Plot 1|Plot 2|Plot 3"}
    """
    data = request.json
    profile_name = data.get("profile")
    
    # Get profile data and handle errors
    profile_data, error = get_profile_data(profile_name)
    if error:
        return jsonify(error), 400
    
    # Update global values and return success message
    user_values.update(profile_data)
    return jsonify({"message": f"Profile {profile_name} applied"}), 200

# ================================================================================
# CORE CALCULATION FUNCTIONS
# ================================================================================
def perform_calculations(user_values):
    """
    Helper function to perform the complete ISR spectra calculation chain.
    
    This function orchestrates all the physics calculations needed for ISR spectra:
    1. Extracts parameters from user_values
    2. Calculates wave vector components
    3. Computes thermal velocities and cyclotron frequencies
    4. Calculates plasma parameters (Debye length, alpha, etc.)
    5. Performs electron calculations
    6. Performs multi-ion species calculations
    7. Returns all necessary components for spectra generation
    
    Args:
        user_values (dict): Dictionary containing all plasma and radar parameters
        
    Returns:
        tuple: (omega_values, M_i_total, M_e, chi_i_total, chi_e)
            - omega_values: Frequency array
            - M_i_total: Total ion distribution function
            - M_e: Electron distribution function  
            - chi_i_total: Total ion susceptibility
            - chi_e: Electron susceptibility
    """
    
    # ============================================================================
    # PARAMETER EXTRACTION
    # ============================================================================
    # Extract collision frequencies
    nu_i = user_values["nu_i"]           # Ion collision frequency [Hz]
    nu_e = user_values["nu_e"]           # Electron collision frequency [Hz]
    
    # Extract plasma parameters
    ne = user_values["ne"]               # Electron density [m^-3]
    me = user_values["me"]               # Electron mass [kg]
    B = user_values["B"]                 # Magnetic field [T]
    theta = user_values["theta"]         # Scattering angle [degrees]
    Te = user_values["Te"]               # Electron temperature [K]
    Ti = user_values["Ti"]               # Ion temperature [K]
    frequency = user_values["frequency"] # Radar frequency [Hz]
    
    # Extract physical constants
    epsilon_0 = user_values["epsilon_0"] # Permittivity of free space [F/m]
    kB = user_values["kB"]               # Boltzmann constant [J/K]
    e = user_values["e"]                 # Elementary charge [C]
    n_terms = int(user_values["n_terms"]) # Number of frequency points

    # ============================================================================
    # ION SPECIES CONFIGURATION
    # ============================================================================
    # Get ion species data or use default multi-ion composition
    ion_species = user_values.get("ion_species", [
        {"name": "O+",   "fraction": 0.488, "density": 2.03e5, "mass": 2.65686e-26},  # Oxygen ions
        {"name": "N+",   "fraction": 0.032, "density": 1.33e4, "mass": 2.32587e-26},  # Nitrogen ions
        {"name": "H+",   "fraction": 0.456, "density": 1.89e5, "mass": 1.67262e-27},  # Hydrogen ions
        {"name": "HE+",  "fraction": 0.024, "density": 9.96e3, "mass": 6.64648e-27},  # Helium ions
        {"name": "O2+",  "fraction": 0.0,   "density": 0.0,    "mass": 5.31372e-26},  # Molecular oxygen ions
        {"name": "NO+",  "fraction": 0.0,   "density": 0.0,    "mass": 2.4828e-26}    # Nitric oxide ions
    ])

    # ============================================================================
    # FUNDAMENTAL CALCULATIONS
    # ============================================================================
    # Calculate radar wavelength from frequency
    c_light = 3e8  # Speed of light in m/s
    lambda_wavelength = c_light / frequency  # Wavelength in meters
    
    # Calculate wave vector components
    k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(lambda_wavelength, theta)
    
    # Calculate electron parameters
    vth_e = calculate_thermal_velocity(kB, Te, me)           # Electron thermal velocity
    Oc_e = calculate_cyclotron_frequency(1, e, B, me)       # Electron cyclotron frequency
    rho_e = calculate_average_gyroradius(vth_e, Oc_e)       # Electron gyroradius
    
    # Calculate plasma parameters
    lambda_De = calculate_debye_length(Te, ne, epsilon_0, kB, e)  # Debye length
    alpha_e = calculate_alpha(k_total, lambda_De)                 # Alpha parameter
    
    # Calculate ion acoustic parameters
    c = calculate_sound_speed(kB, Te, Ti, sum(ion["fraction"] * ion["mass"] for ion in ion_species))
    omega_values = calculate_omega_values(k_total, c)  # Frequency array
    
    # ============================================================================
    # ELECTRON CALCULATIONS
    # ============================================================================
    # Calculate electron collision and distribution terms
    U_e = calculate_collisional_term(nu_e, k_parallel, vth_e, k_perpendicular, rho_e, n_terms, omega_values, Oc_e)
    M_e = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, U_e)
    chi_e = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, alpha_e, U_e, Te, Ti)
    
    # ============================================================================
    # MULTI-ION SPECIES CALCULATIONS
    # ============================================================================
    # Initialize total ion contributions
    M_i_total = 0      # Total ion distribution function
    chi_i_total = 0    # Total ion susceptibility
    
    # Calculate contribution from each ion species
    for ion in ion_species:
        if ion["fraction"] > 0:  # Only process ions with non-zero fractions
            # Extract ion-specific parameters
            mi = ion["mass"]        # Ion mass [kg]
            frac = ion["fraction"]  # Ion fraction (relative abundance)
            
            # Calculate ion-specific parameters
            vth_i = calculate_thermal_velocity(kB, Ti, mi)           # Ion thermal velocity
            Oc_i = calculate_cyclotron_frequency(1, e, B, mi)       # Ion cyclotron frequency
            rho_i = calculate_average_gyroradius(vth_i, Oc_i)       # Ion gyroradius
            
            # Calculate ion collision and distribution terms
            U_i = calculate_collisional_term(nu_i, k_parallel, vth_i, k_perpendicular, rho_i, n_terms, omega_values, Oc_i)
            M_i = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, U_i)
            chi_i = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, alpha_e, U_i, Te, Ti)
            
            # Add weighted contribution to totals
            M_i_total += frac * M_i      # Weight by ion fraction
            chi_i_total += frac * chi_i  # Weight by ion fraction

    return omega_values, M_i_total, M_e, chi_i_total, chi_e

# ================================================================================
# PLOTTING AND DATA ENDPOINTS
# ================================================================================
@app.route('/plot')
def plot():
    """
    Generate ISR spectra data and return as JSON for plotting.
    
    This endpoint performs the complete calculation chain and returns
    frequency and spectra arrays for client-side plotting.
    
    Returns:
        JSON object containing:
        - frequencies: Array of frequency values in MHz
        - spectra: Array of spectral intensity values
    """
    # Perform all ISR calculations
    omega_values, M_i_total, M_e, chi_i_total, chi_e = perform_calculations(user_values)
    
    # Calculate the final ISR spectra
    spectra = calcSpectra(M_i_total, M_e, chi_i_total, chi_e)
    
    # Prepare data for JSON response
    data = {
        "frequencies": (omega_values / (2 * np.pi * 1e6)).tolist(),  # Convert to MHz and list
        "spectra": spectra.tolist()                                  # Convert numpy array to list
    }
    
    return jsonify(data)

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================
def get_profile_data(profile_name):
    """
    Retrieve parameter data for a specified profile.
    
    Args:
        profile_name (str): Name of the profile to retrieve
        
    Returns:
        tuple: (profile_data, error)
            - profile_data: Dictionary of parameters if successful, None if error
            - error: Error dictionary if profile not found, None if successful
    """
    if profile_name in parameter_profiles:
        return parameter_profiles[profile_name], None
    return None, {"error": "Invalid profile"}

# ================================================================================
# APPLICATION ENTRY POINT
# ================================================================================
if __name__ == '__main__':
    """
    Start the Flask development server.
    
    Configuration:
    - debug=True: Enable debug mode for development
    - Runs on default host (127.0.0.1) and port (5000)
    """
    app.run(debug=True)

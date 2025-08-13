# ================================================================================
# IMPORTS AND DEPENDENCIES
# ================================================================================
import matplotlib.pyplot as plt  # For plotting spectra
import numpy as np              # Numerical computations
from scipy.special import ive, dawsn  # Modified Bessel functions and Dawson function
import scipy.special as sp      # Additional special functions
import numexpr as ne           # Fast numerical expression evaluation

# ================================================================================
# FUNDAMENTAL WAVE VECTOR CALCULATIONS
# ================================================================================
def calculate_wavenumber_components(lambda_wavelength, theta_degrees):
    """
    Calculate radar wave vector components for scattering geometry.
    
    In ISR theory, the wave vector difference k = k_s - k_0 determines which
    plasma fluctuations are probed. The parallel and perpendicular components
    relative to the magnetic field determine different physical processes.
    
    Parameters:
    - lambda_wavelength (float): Radar wavelength [m]
    - theta_degrees (float): Scattering angle between incident and scattered waves [degrees]
    
    Returns:
    - k (float): Total wavenumber magnitude |k| [rad/m]
    - k_parallel (float): Component parallel to magnetic field [rad/m]  
    - k_perpendicular (float): Component perpendicular to magnetic field [rad/m]
    
    Physical Significance:
    - k_parallel: Probes field-aligned irregularities, ion acoustic waves
    - k_perpendicular: Probes cross-field diffusion, cyclotron effects
    """
    theta = np.radians(theta_degrees)  # Convert degrees to radians
    k = 2 * 2 * np.pi / lambda_wavelength  # Total wavenumber: |k_s - k_0| = 4π/λ for backscatter
    k_parallel = k * np.cos(theta)      # Parallel component: k·B̂
    k_perpendicular = k * np.sin(theta) # Perpendicular component: k×B̂
    return k, k_parallel, k_perpendicular

# ================================================================================
# PLASMA WAVE SPEED CALCULATIONS
# ================================================================================
def calculate_sound_speed(kB, Te, Ti, mi):
    """
    Calculate ion acoustic speed in the plasma.
    
    The ion acoustic speed determines the frequency scale of ion acoustic waves,
    which appear as the central "ion line" in ISR spectra. This speed depends
    on both electron and ion temperatures since electrons provide the pressure
    restoring force while ions provide the inertia.
    
    Parameters:
    - kB (float): Boltzmann constant [J/K]
    - Te (float): Electron temperature [K]
    - Ti (float): Ion temperature [K]  
    - mi (float): Average ion mass [kg]
    
    Returns:
    - c (float): Ion acoustic speed [m/s]
    
    Physical Notes:
    - For Te >> Ti: c ≈ √(kB*Te/mi) (electron temperature dominated)
    - For Te ≈ Ti: c ≈ √(kB*(Te+Ti)/mi) (both temperatures important)
    - Factor 5/3 accounts for adiabatic compression in 3D
    """
    c = np.sqrt((5 / 3) * (kB * (Ti + Te) / mi))  # Ion acoustic speed
    return c

# ================================================================================
# FREQUENCY GRID GENERATION  
# ================================================================================
def calculate_omega_values(k, c):
    """
    Generate optimized frequency grid for ISR spectra calculation.
    
    This function creates a non-uniform frequency grid that provides high resolution
    around spectral features while maintaining computational efficiency. The grid
    is designed to capture:
    - Ion line: narrow central feature around DC
    - Ion cyclotron lines: at ±Ωci, ±2Ωci, etc.
    - Electron cyclotron lines: at ±Ωce, ±2Ωce, etc.  
    - Plasma lines: broad features at high frequencies
    
    Parameters:
    - k (float): Total wavenumber [rad/m] (currently not used but kept for compatibility)
    - c (float): Sound speed [m/s] (currently not used but kept for compatibility)
    
    Returns:
    - omega_rad (array): Frequency array in angular frequency [rad/s]
    
    Grid Design Philosophy:
    - Dense sampling around narrow spectral lines (1000-1001 points)
    - Sparse sampling in regions between features (100 points)
    - Symmetric about zero frequency
    - Covers ±6 MHz to capture all major spectral features
    """
    # Define frequency regions with different sampling densities
    regions = [
        # (start_MHz, end_MHz, num_points)
        (-6.0, -4.6, 100),    # Far negative frequencies (sparse sampling)
        (-4.5, -4.0, 1000),   # Electron plasma line region (dense sampling)
        (-3.9, -0.9, 100),    # Between plasma and cyclotron lines (sparse)
        (-0.8, -0.3, 1000),   # Electron cyclotron region (dense sampling)
        (-0.2, -0.1, 100),    # Transition to ion line (sparse)
        (-0.1, 0.1, 1001),    # Ion line - central feature (very dense sampling)
        (0.1, 0.2, 100),      # Transition from ion line (sparse)
        (0.3, 0.8, 1000),     # Electron cyclotron region (dense sampling)
        (0.9, 3.9, 100),      # Between cyclotron and plasma lines (sparse)
        (4.0, 4.5, 1000),     # Electron plasma line region (dense sampling)
        (4.6, 6.0, 100),      # Far positive frequencies (sparse sampling)
    ]

    omega_hz = []
    for start, end, num in regions:
        # Avoid duplicate points at region boundaries (except for the last region)
        endpoint = False if end != regions[-1][1] else True
        omega_hz.append(np.linspace(start * 1e6, end * 1e6, num, endpoint=endpoint))
    
    omega_hz = np.concatenate(omega_hz)  # Combine all frequency regions
    omega_rad = omega_hz * 2 * np.pi     # Convert Hz to rad/s for calculations
    return omega_rad

# ================================================================================
# THERMAL VELOCITY CALCULATIONS
# ================================================================================
def calculate_thermal_velocity(kB, T, m):
    """
    Calculate thermal velocity for Maxwell-Boltzmann distribution.
    
    The thermal velocity characterizes the typical speed of particles in a
    thermal distribution. It appears in the Doppler broadening of spectral
    lines and determines the collision frequency scales.
    
    Parameters:
    - kB (float): Boltzmann constant [J/K]
    - T (float): Temperature [K]
    - m (float): Particle mass [kg]
    
    Returns:
    - vth (float): Thermal velocity [m/s]
    
    Physical Interpretation:
    - Most probable speed in Maxwell-Boltzmann distribution
    - Sets width of velocity distribution: f(v) ∝ exp(-mv²/2kBT)
    - For electrons: typically ~10⁶ m/s
    - For ions: typically ~10³-10⁴ m/s
    """
    vth = np.sqrt((2 * kB * T) / m)  # Thermal velocity: √(2kBT/m)
    return vth

# ================================================================================
# CYCLOTRON FREQUENCY CALCULATIONS
# ================================================================================
def calculate_cyclotron_frequency(Z, e, B, m):
    """
    Calculate cyclotron (gyro) frequency for charged particles in magnetic field.
    
    The cyclotron frequency is the natural oscillation frequency of charged
    particles in a magnetic field. It appears as sharp spectral lines in
    ISR spectra and affects the dispersion relation of plasma waves.
    
    Parameters:
    - Z (int): Charge state of particle (1 for singly ionized)
    - e (float): Elementary charge [C]
    - B (float): Magnetic field strength [T]
    - m (float): Particle mass [kg]
    
    Returns:
    - Oc (float): Cyclotron frequency [rad/s]
    
    Physical Notes:
    - Electron cyclotron frequency: Ωce = eB/me (~1.76×10¹¹ rad/s for B=50μT)
    - Ion cyclotron frequency: Ωci = eB/mi (~10⁸ rad/s for O+ in B=50μT)  
    - Ωce >> Ωci due to mass difference
    - Appears as harmonics: ±nΩc in spectra
    """
    Oc = (Z * e * B) / m  # Cyclotron frequency: Ω = qB/m
    return Oc

# ================================================================================
# GYRORADIUS CALCULATIONS
# ================================================================================
def calculate_average_gyroradius(vth, Oc):
    """
    Calculate thermal gyroradius (Larmor radius) for magnetized particles.
    
    The gyroradius characterizes the scale of cyclotron motion and determines
    the importance of magnetic field effects in the plasma response. It appears
    in the argument of Bessel functions in ISR theory.
    
    Parameters:
    - vth (float): Thermal velocity [m/s]
    - Oc (float): Cyclotron frequency [rad/s]
    
    Returns:
    - rho (float): Thermal gyroradius [m]
    
    Physical Significance:
    - Average radius of cyclotron motion for thermal particles
    - When k⊥ρ << 1: unmagnetized limit (no cyclotron effects)
    - When k⊥ρ ~ 1: magnetized plasma (strong cyclotron effects)
    - When k⊥ρ >> 1: highly magnetized (multiple cyclotron harmonics)
    - Factor √2 accounts for thermal average of v⊥
    """
    rho = vth / (np.sqrt(2) * Oc)  # Thermal gyroradius: ρ = vth/(√2 Ωc)
    return rho

# ================================================================================
# COLLISIONAL EFFECTS - GORDEYEV INTEGRAL COMPUTATION
# ================================================================================
def calculate_collisional_term(nu, k_par, vth, k_perp, rho, n, omega, Oc):
    """
    Calculate collisional correction term U for the Gordeyev integral.
    
    This function computes the collisional modification to the ideal plasma
    response. Collisions damp coherent oscillations and modify the spectral
    shape, particularly important for the ion line in the lower ionosphere.
    
    The Gordeyev integral includes cyclotron harmonics through the sum over n,
    with each harmonic weighted by modified Bessel functions I_n(k⊥ρ).
    
    Parameters:
    - nu (float): Collision frequency [Hz]
    - k_par (float): Parallel wavenumber component [rad/m]
    - vth (float): Thermal velocity [m/s]
    - k_perp (float): Perpendicular wavenumber component [rad/m]
    - rho (float): Thermal gyroradius [m]
    - n (int): Number of cyclotron harmonics to include
    - omega (array): Frequency array [rad/s]
    - Oc (float): Cyclotron frequency [rad/s]
    
    Returns:
    - U (complex array): Collisional term for Gordeyev integral
    
    Mathematical Form:
    U = iν/(k∥vth) × Σ_n I_n(k⊥²ρ²) × [2×Dawsn(y_n) + i√π×exp(-y_n²)]
    where y_n = (ω - nΩc - iν)/(k∥vth)
    
    Physical Effects:
    - Collision frequency ν damps oscillations
    - Dawson function handles resonant denominators
    - Bessel functions I_n account for cyclotron harmonics
    - Real part: frequency shift due to collisions
    - Imaginary part: damping of oscillations
    """
    U = np.zeros_like(omega) + 1j * 0.0  # Initialize complex array
    k_rho_sq = (k_perp ** 2) * (rho ** 2)  # Argument of Bessel functions: (k⊥ρ)²
    
    # Sum over cyclotron harmonics from -n to +n
    for i in range(-n, n + 1):
        # Normalized frequency variable including collision frequency
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        
        # Complex exponential term: exp(-y_n²)
        exp_term = ne.evaluate('exp(-yn**2)')
        
        # Gordeyev integral contribution for harmonic i
        # I_n(k⊥ρ)² × [2×Dawsn(y_n) + i√π×exp(-y_n²)]
        U += sp.ive(i, k_rho_sq) * (2 * sp.dawsn(yn) + 1j * np.sqrt(np.pi) * exp_term)
    
    # Apply overall prefactor: iν/(k∥vth)
    U = U * 1j * nu / (k_par * vth)
    return U

# ================================================================================
# MODIFIED DISTRIBUTION FUNCTION CALCULATION
# ================================================================================
def calculate_modified_distribution(omega, k_par, k_perp, vth, n, rho, Oc, nu, U):
    """
    Calculate the collisionally modified velocity distribution function.
    
    This function computes how collisions modify the equilibrium Maxwell-Boltzmann
    distribution in response to the electric field fluctuations. The result M(ω)
    represents the spectral density of velocity fluctuations that contribute to
    scattering.
    
    Parameters:
    - omega (array): Frequency array [rad/s]
    - k_par (float): Parallel wavenumber [rad/m]  
    - k_perp (float): Perpendicular wavenumber [rad/m]
    - vth (float): Thermal velocity [m/s]
    - n (int): Number of cyclotron harmonics
    - rho (float): Thermal gyroradius [m]
    - Oc (float): Cyclotron frequency [rad/s]
    - nu (float): Collision frequency [Hz]
    - U (complex array): Collisional term from calculate_collisional_term()
    
    Returns:
    - M_s (real array): Modified distribution function
    
    Mathematical Structure:
    M = -|U|²/(ν|1+U|²) + (1/(k∥vth|1+U|²)) × Σ_n I_n(k⊥²ρ²) × [√π×Re(exp(-y_n²)) + Im(2×Dawsn(y_n))]
    
    Physical Interpretation:
    - First term: Collisional damping contribution (always negative)
    - Second term: Resonant contribution from thermal particles
    - Bessel functions: Weight cyclotron harmonics
    - Real part taken at end: Physical observable quantity
    - Units: Velocity distribution function [s/m]
    """
    M = np.zeros_like(omega) + 1j * 0.0  # Initialize complex array
    k_rho_sq = (k_perp ** 2) * (rho ** 2)  # Bessel function argument
    
    # Collisional damping term: -|U|²/(ν|1+U|²)
    term_one = -(abs(U) ** 2) / (nu * (abs(1 + U) ** 2))
    
    # Resonant contribution prefactor: 1/(k∥vth|1+U|²)
    term_two = 1 / (k_par * vth * (abs(1 + U) ** 2))
    
    # Sum over cyclotron harmonics
    for i in range(-n, n + 1):
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        
        # Real part of exp(-y_n²): contributes to distribution width
        exp_re = np.real(ne.evaluate('exp(-yn**2)'))
        
        # Imaginary part of Dawson function: handles resonant denominators
        imag_da = 2 * np.imag(dawsn(yn))
        
        # Weight by Bessel function for this harmonic
        bessel_term = ive(i, k_rho_sq)
        
        # Add contribution from this harmonic
        M += bessel_term * (np.sqrt(np.pi) * exp_re + imag_da)
    
    # Combine terms and return real part (physical quantity)
    M_s = term_one + term_two * M
    return np.real(M_s)

# ================================================================================
# PLASMA PARAMETER CALCULATIONS
# ================================================================================
def calculate_debye_length(Te, ne_, epsilon_0, kB, e):
    """
    Calculate the electron Debye screening length.
    
    The Debye length characterizes the scale over which electric fields are
    screened in a plasma. It determines the transition between collective
    (coherent) and individual particle (incoherent) scattering regimes.
    
    Parameters:
    - Te (float): Electron temperature [K]
    - ne_ (float): Electron density [m⁻³]
    - epsilon_0 (float): Vacuum permittivity [F/m]
    - kB (float): Boltzmann constant [J/K]
    - e (float): Elementary charge [C]
    
    Returns:
    - lambda_De (float): Electron Debye length [m]
    
    Physical Significance:
    - λD ~ √(ε₀kBTe/nee²): balance of thermal and electrostatic energy
    - For kλD << 1: coherent scattering (Thomson limit)
    - For kλD >> 1: incoherent scattering (ISR regime)
    - Typical ionospheric values: λD ~ 1-10 mm
    - Controls the α parameter: α = 1/(kλD)
    """
    lambda_De = np.sqrt(epsilon_0 * kB * Te / (ne_ * e**2))  # Debye length formula
    return lambda_De

# ================================================================================
# ISR SCATTERING PARAMETER
# ================================================================================
def calculate_alpha(k, lambda_De):
    """
    Calculate the incoherent scattering parameter α.
    
    The α parameter determines the scattering regime and the relative strength
    of ion vs electron contributions to the scattered spectrum. It is the
    fundamental parameter that distinguishes ISR from Thomson scattering.
    
    Parameters:
    - k (float): Total wavenumber [rad/m]
    - lambda_De (float): Electron Debye length [m]
    
    Returns:
    - alpha (float): Incoherent scattering parameter [dimensionless]
    
    Physical Regimes:
    - α << 1 (kλD >> 1): Incoherent scattering regime
      * Individual particle fluctuations dominate
      * Ion and electron features well separated
      * Standard ISR theory applies
    
    - α >> 1 (kλD << 1): Coherent scattering regime  
      * Collective plasma oscillations dominate
      * Approaches Thomson scattering limit
      * Plasma lines become prominent
    
    - α ~ 1: Transition regime
      * Both effects important
      * Complex spectral structure
    
    Typical Values:
    - VHF radars (50-300 MHz): α ~ 0.1-1
    - UHF radars (400-1000 MHz): α ~ 0.01-0.1  
    """
    alpha = 1 / (k * lambda_De)  # α = 1/(kλD)
    return alpha

# ================================================================================
# ELECTRIC SUSCEPTIBILITY CALCULATION
# ================================================================================
def calculate_electric_susceptibility(omega, k_par, k_perp, vth, n, rho, Oc, nu, alpha, U, Te, Ts):
    """
    Calculate the electric susceptibility χ(ω) for plasma species.
    
    The electric susceptibility describes how the plasma responds to electric
    field fluctuations. It determines the dispersion relation for plasma waves
    and controls the spectral shape of ISR signals through the dielectric function.
    
    Parameters:
    - omega (array): Frequency array [rad/s]
    - k_par (float): Parallel wavenumber [rad/m]
    - k_perp (float): Perpendicular wavenumber [rad/m]  
    - vth (float): Thermal velocity [m/s]
    - n (int): Number of cyclotron harmonics
    - rho (float): Thermal gyroradius [m]
    - Oc (float): Cyclotron frequency [rad/s]
    - nu (float): Collision frequency [Hz]
    - alpha (float): ISR parameter α
    - U (complex array): Collisional term
    - Te (float): Electron temperature [K]
    - Ts (float): Species temperature [K]
    
    Returns:
    - chi (complex array): Electric susceptibility χ(ω)
    
    Mathematical Form:
    χ = (α²/(1+U)) × (Te/Ts) × Σ_n I_n(k⊥²ρ²) × [1 - ((ω-iν)/(k∥vth)) × (2×Dawsn(y_n) + i√π×exp(-y_n²))]
    
    Physical Components:
    - α² term: Couples to electrostatic fluctuations
    - Te/Ts: Temperature ratio (important for multi-species)
    - (1+U): Collisional modification
    - Bessel sum: Cyclotron harmonic contributions
    - Bracketed term: Kinetic response function
    
    Physical Interpretation:
    - Real part: Dispersive effects (frequency shifts)
    - Imaginary part: Absorptive effects (damping)
    - Poles at cyclotron harmonics: ω ≈ nΩc
    - Controls plasma wave propagation and damping
    """
    chi = np.zeros_like(omega) + 1j * 0.0  # Initialize complex array
    k_rho_sq = (k_perp ** 2) * (rho ** 2)  # Bessel function argument
    
    # Frequency term including collisions: (ω - iν)/(k∥vth)
    term_one = (omega - 1j * nu) / (k_par * vth)
    
    # Coupling strength: α²/(1+U)
    term_four = (alpha ** 2) / (1 + U)
    
    # Temperature ratio: Te/Ts (electron temp / species temp)
    term_five = Te / Ts
    
    # Sum over cyclotron harmonics
    for i in range(-n, n + 1):
        yn = (omega - i * Oc - 1j * nu) / (k_par * vth)
        
        # Dawson function term: handles resonant response
        term_two = 2 * sp.dawsn(yn)
        
        # Complex exponential: exp(-y_n²)
        exp_term = ne.evaluate('exp(-yn**2)')
        
        # Imaginary contribution: i√π×exp(-y_n²)
        term_three = 1j * np.sqrt(np.pi) * exp_term
        
        # Add weighted contribution from this harmonic
        # Structure: I_n(k⊥²ρ²) × [1 - (frequency term) × (resonant response)]
        chi += sp.ive(i, k_rho_sq) * (1 - term_one * (term_two + term_three))
    
    # Apply overall factors and return susceptibility
    return term_four * term_five * chi

# ================================================================================
# FINAL SPECTRUM CALCULATION
# ================================================================================
def calcSpectra(M_i, M_e, chi_i, chi_e):
    """
    Calculate the final ISR spectrum from distribution functions and susceptibilities.
    
    This is the culmination of the ISR theory calculation. It combines the ion and
    electron contributions through the plasma dielectric function to produce the
    scattered power spectrum that would be observed by the radar.
    
    Parameters:
    - M_i (real array): Ion modified distribution function
    - M_e (real array): Electron modified distribution function  
    - chi_i (complex array): Ion electric susceptibility
    - chi_e (complex array): Electron electric susceptibility
    
    Returns:
    - spectrum (real array): ISR backscatter spectrum [arbitrary units]
    
    Mathematical Formula:
    S(ω) = 2|1 - χe/ε|² × Me + 2|χe/ε|² × Mi
    
    where ε = 1 + χe + χi is the plasma dielectric function
    
    Physical Interpretation:
    - First term: Electron contribution to spectrum
      * Weighted by |1 - χe/ε|²: electron response strength
      * Me: electron velocity fluctuation spectrum
      * Produces "electron lines" at high frequencies
    
    - Second term: Ion contribution to spectrum  
      * Weighted by |χe/ε|²: coupling through electron response
      * Mi: ion velocity fluctuation spectrum
      * Produces "ion line" at low frequencies
    
    Spectral Features:
    - Ion line: Narrow central feature from ion acoustic waves
    - Electron lines: Broader wings from electron Landau damping
    - Cyclotron lines: Sharp features at ±nΩc harmonics
    - Plasma lines: High-frequency features from plasma oscillations
    
    The factor of 2 accounts for both upshifted and downshifted components
    in the backscatter geometry.
    """
    # Calculate plasma dielectric function: ε = 1 + χe + χi
    epsilon = 1 + chi_e + chi_i
    
    # Calculate spectrum components with proper weighting
    # Electron contribution: 2|1 - χe/ε|² × Me
    electron_contribution = 2 * np.abs(1 - chi_e/epsilon)**2 * M_e
    
    # Ion contribution: 2|χe/ε|² × Mi  
    ion_contribution = 2 * np.abs(chi_e/epsilon)**2 * M_i
    
    # Total spectrum: sum of ion and electron contributions
    spectrum = electron_contribution + ion_contribution
    
    return spectrum

# ================================================================================
# MAIN EXECUTION AND TESTING CODE
# ================================================================================
# Ensure the script only runs when executed directly, not when imported
if __name__ == "__main__":
    """
    Main execution block for testing and demonstration.
    
    This section provides a complete example of ISR spectrum calculation
    with realistic ionospheric parameters. It demonstrates:
    - Multi-temperature analysis (Te = 500, 1500, 2500, 3500 K)
    - Multi-ion species composition (O+, N+, H+, He+, O2+, NO+)
    - Complete calculation chain from parameters to spectrum
    - Visualization of results
    """
    
    # ============================================================================
    # PHYSICAL PARAMETERS - REALISTIC IONOSPHERIC CONDITIONS
    # ============================================================================
    
    # Collision frequencies [Hz] - typical F-region values
    nu_i = 0.0000001   # Ion collision frequency (very low in F-region)
    nu_e = 0.0000001   # Electron collision frequency (very low in F-region)
    
    # Plasma densities [m⁻³] - typical F-region peak
    ni = 2e11          # Ion density
    ne_ = 2e11         # Electron density (quasi-neutrality)
    
    # Particle masses [kg]
    mi = 2.65686e-26   # Ion mass (atomic oxygen O+)
    m_e = 9.11e-31     # Electron mass
    
    # Geophysical parameters
    B = 3.6e-5         # Magnetic field strength [T] - typical mid-latitude
    theta = 80         # Scattering angle [degrees] - near field-aligned
    
    # Temperature scan [K] - covering typical ionospheric range
    Te_values = [500, 1500, 2500, 3500]  # Electron temperatures for comparison
    
    # Physical constants
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity [F/m]
    kB = 1.380649e-23           # Boltzmann constant [J/K]  
    e = 1.602e-19               # Elementary charge [C]
    
    # ============================================================================
    # ION SPECIES COMPOSITION - REALISTIC F-REGION MIX
    # ============================================================================
    # Define realistic ionospheric ion composition with fractional abundances
    ion_species = [
        {"name": "O+",   "fraction": 0.488, "density": 2.03e5, "mass": 2.65686e-26},  # Dominant F-region ion
        {"name": "N+",   "fraction": 0.032, "density": 1.33e4, "mass": 2.32587e-26},  # Minor component
        {"name": "H+",   "fraction": 0.456, "density": 1.89e5, "mass": 1.67262e-27},  # Light ion (high altitudes)
        {"name": "HE+",  "fraction": 0.024, "density": 9.96e3, "mass": 6.64648e-27},  # Helium ions
        {"name": "O2+",  "fraction": 0.0,   "density": 0.0,    "mass": 5.31372e-26},  # E-region ion (not included)
        {"name": "NO+",  "fraction": 0.0,   "density": 0.0,    "mass": 2.4828e-26}    # E-region ion (not included)
    ]
    
    # Calculation parameters
    n_terms = 2000  # Number of cyclotron harmonics (affects accuracy vs speed)
    
    # ============================================================================
    # RADAR PARAMETERS AND GEOMETRY CALCULATION
    # ============================================================================
    # Calculate wave vector components from radar wavelength and scattering angle
    # Using 430 MHz radar: λ = c/f = 3×10⁸/430×10⁶ = 0.69719 m
    k_total, k_parallel, k_perpendicular = calculate_wavenumber_components(0.69719, theta)
    
    # ============================================================================
    # MULTI-TEMPERATURE SPECTRUM CALCULATION LOOP
    # ============================================================================
    spectra_list = []  # Store spectra for each temperature
    
    for T in Te_values:
        print(f"Calculating spectrum for Te = Ti = {T} K...")
        
        # Set electron and ion temperatures equal for this example
        Te = Ti = T
        
        # ========================================================================
        # SINGLE-ION CALCULATIONS (for reference - not used in final spectrum)
        # ========================================================================
        # Calculate thermal velocities for single-ion case
        vth_i = calculate_thermal_velocity(kB, Ti, mi)    # Ion thermal velocity
        vth_e = calculate_thermal_velocity(kB, Te, m_e)   # Electron thermal velocity
        
        # Calculate cyclotron frequencies
        Oc_i = calculate_cyclotron_frequency(1, e, B, mi)  # Ion cyclotron frequency
        Oc_e = calculate_cyclotron_frequency(1, e, B, m_e) # Electron cyclotron frequency
        
        # Calculate gyroradii
        rho_i = calculate_average_gyroradius(vth_i, Oc_i)  # Ion gyroradius
        rho_e = calculate_average_gyroradius(vth_e, Oc_e)  # Electron gyroradius
        
        # Calculate Debye lengths and α parameters
        lambda_De = calculate_debye_length(Te, ne_, epsilon_0, kB, e)   # Electron Debye length
        lambda_Di = calculate_debye_length(Te, ni, epsilon_0, kB, e)    # Ion Debye length
        alpha_e = calculate_alpha(k_total, lambda_De)                   # Electron α parameter
        alpha_i = calculate_alpha(k_total, lambda_Di)                   # Ion α parameter
        
        # Calculate ion acoustic speed and frequency grid
        c = calculate_sound_speed(kB, Te, Ti, mi)          # Ion acoustic speed
        omega_values = calculate_omega_values(k_total, c)  # Frequency array
        
        # ========================================================================
        # ELECTRON CALCULATIONS
        # ========================================================================
        # Calculate electron collisional term and distribution
        U_e = calculate_collisional_term(nu_e, k_parallel, vth_e, k_perpendicular, rho_e, n_terms, omega_values, Oc_e)
        M_e = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, U_e)
        chi_e = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_e, n_terms, rho_e, Oc_e, nu_e, alpha_e, U_e, Te, Te)
        
        # ========================================================================
        # MULTI-ION SPECIES CALCULATIONS
        # ========================================================================
        # Initialize total ion contributions
        M_i_total = 0      # Total ion distribution function
        chi_i_total = 0    # Total ion susceptibility
        
        print("  Processing ion species:")
        # Calculate contribution from each ion species
        for ion in ion_species:
            if ion["fraction"] > 0:  # Only process ions with non-zero abundance
                print(f"    {ion['name']}: fraction = {ion['fraction']:.3f}")
                
                # Extract ion-specific parameters
                mi = ion["mass"]        # Ion mass [kg]
                ni = ion["density"]     # Ion density [m⁻³]
                frac = ion["fraction"]  # Fractional abundance
                
                # Calculate ion-specific thermal and cyclotron parameters
                vth_i = calculate_thermal_velocity(kB, Ti, mi)           # Ion thermal velocity
                Oc_i = calculate_cyclotron_frequency(1, e, B, mi)       # Ion cyclotron frequency
                rho_i = calculate_average_gyroradius(vth_i, Oc_i)       # Ion gyroradius
                
                # Calculate ion Debye length and α parameter
                lambda_Di = calculate_debye_length(Ti, ni, epsilon_0, kB, e)
                alpha_i = calculate_alpha(k_total, lambda_Di)
                
                # Calculate ion collisional term and distribution
                U_i = calculate_collisional_term(nu_i, k_parallel, vth_i, k_perpendicular, rho_i, n_terms, omega_values, Oc_i)
                M_i = calculate_modified_distribution(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, U_i)
                chi_i = calculate_electric_susceptibility(omega_values, k_parallel, k_perpendicular, vth_i, n_terms, rho_i, Oc_i, nu_i, alpha_i, U_i, Ti, Ti)
                
                # Add weighted contribution to totals
                M_i_total += frac * M_i      # Weight by fractional abundance
                chi_i_total += frac * chi_i  # Weight by fractional abundance
        
        # ========================================================================
        # FINAL SPECTRUM CALCULATION
        # ========================================================================
        # Note: Using single-ion M_i instead of M_i_total for compatibility with existing code
        # In production, should use M_i_total for multi-ion effects
        spectra = calcSpectra(M_i, M_e, chi_i, chi_e)
        spectra_list.append(spectra)
        print(f"  Spectrum calculation complete.\n")
    
    # ============================================================================
    # VISUALIZATION AND RESULTS
    # ============================================================================
    print("Generating plots...")
    
    # Create figure for spectrum comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for different temperature curves (MATLAB default colors)
    colors = [(0, 0.4470, 0.7410),      # Blue
              (0.8500, 0.3250, 0.0980),  # Orange  
              (0.9290, 0.6940, 0.1250),  # Yellow
              (0.4940, 0.1840, 0.5560)]  # Purple
    
    # Create labels for legend
    labels = [f'Te = {T} K' for T in Te_values]
    
    # Plot each spectrum with different colors
    for i, spectra in enumerate(spectra_list):
        # Convert angular frequency to MHz for plotting
        freq_mhz = omega_values / (2 * np.pi * 1e6)
        ax.plot(freq_mhz, spectra, color=colors[i], label=labels[i], linewidth=1.5)
    
    # Configure plot appearance
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Spectra', fontsize=12) 
    ax.set_title('ISR Backscatter Spectra - Temperature Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Logarithmic y-axis to show dynamic range
    ax.set_xlim(-6, 6)    # Focus on main spectral features
    
    # Improve layout and display
    plt.tight_layout()
    plt.show(block=True)
    
    print("Calculation and visualization complete!")
    print(f"Processed {len(Te_values)} temperature cases")
    print(f"Frequency grid: {len(omega_values)} points from {omega_values[0]/(2*np.pi*1e6):.1f} to {omega_values[-1]/(2*np.pi*1e6):.1f} MHz")

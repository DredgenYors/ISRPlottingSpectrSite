import matplotlib.pyplot as plt
import numpy as np
from spectratestingmodified_chirag_edit import calculate_omega_values, calculate_wavenumber_components, calculate_sound_speed

# Define parameters
lambda_wavelength = 0.69719
theta_degrees = 60
kB = 1.380649e-23
Te = 500
Ti = 500
mi = 2.65686e-26

# Calculate omega_values
k, _, _ = calculate_wavenumber_components(lambda_wavelength, theta_degrees)
c = calculate_sound_speed(kB, Te, Ti, mi)
omega_values = calculate_omega_values(k, c)

# Convert omega_values to frequency in MHz
frequency_MHz = omega_values / (2 * np.pi * 1e6)

# Generate dummy spectra data for demonstration
spectra_data = np.exp(-((frequency_MHz - 0)**2) / (2 * 0.5**2))  # Gaussian-like data

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frequency_MHz, spectra_data, label='Spectra')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Spectra')
plt.title('Spectra Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
Created on Apr 19, 2012

@author: marco
'''

import plasma_parameters as plasma_param
import numpy as np

# Physical Constants [MKS]
import physical_constants as phys_cons

import matplotlib.pyplot as plt
import warnings as warn

from gordeyev_integral import gordeyev_chirpz, read_je_sg, read_je_mk

class isrSpectrum(object):
	'''
	classdocs
	'''
	
	def __init__(self,N,fs,lambdaB,aspdeg,plasma,fdop=0,modgy=0,modnu=0,nue=[],nui=[],neutral=[]):
		'''
		Constructor
		'''

		'''
		def function isr_spectrum(N,f0,df,fradar,Ne,Bm,aspdeg,Te,mu,ion_mass,ion_comp,modgy,modnu,nue,nui):
		
		[spcisr,f,efunc,ifunc,fr] = ...
		
		ISR_SPECTRUM computes the incoherent scatter spectrum.
		
		Syntax:
			[] = isr_spectrum(N,f0,df,fradar,Ne,Bm,aspdeg,Te,mu,ion_mass,ion_comp,modgy,modnu)
			[] = isr_spectrum(N,f0,df,fradar,Ne,Bm,aspdeg,Te,mu,ion_mass,ion_comp,modgy,[],nue,nui)
			[] = isr_spectrum(N,f0,df,fradar,Ne,Bm,aspdeg,Te,mu,ion_mass,ion_comp,modgy,modnu,[],[],Nn,Tn,neu_mass)
			[] = isr_spectrum(N,f0,df,fradar,Ne,Bm,aspdeg,Te,mu,ion_mass,ion_comp,modgy,[],nue,nui,Nn,Tn,neu_mass)
		
		Examples:
			[] = isr_spectrum(4096,0,1,50E6,1E11,25E-6,45,1000,1,16,1,1,2);
			[] = isr_spectrum(4096,0,1,50E6,1E11,25E-6,45,1000,[1.5,2],[16,32],[0.5,0.5],1,2);
		
		Inputs:
		
			N: Number of frequency points
			f0: Initial frequency [Hz]
			df: Frequency resolution of computed spectrum [Hz]
			fdop: Doppler frequency shift [Hz] (added Dec 26, 2023)
		
			fradar: Radar frequency [Hz]
		
			Ne: Electron density [1/m^3]
			Bm: Magnetic field [Tesla]
			aspdeg: Aspect angle [deg] (0 deg perp to B)
			Te: Electron temperature [K]
			mu: Te/Ti temperature ratio
			ion_mass: ion mass number [amu] (in multiples of proton mass)
			ion_comp: ion composition (adding up to unity)
		
			modgy: Gordeyev integral model
			 0: Dougherty&Farley [1963] (BGK)
			 1: Woodman [1967] (Fokker-Planck)
			 2: Sulzer&Gonzalez [1999] (Coulomb collisions for 50MHz radar frequency)
			 3: Milla&Kudeki [2011] (Coulomb collisions for 3m Bragg radar wavelength)
		
			modnu: collision frequency model
			 0: No collisions
			 1: Woodman [1967] (Electron collision frequency), [2004] (Ion collision frequency)
			 2: Callen [2003] (Coulomb collisions)
			 3: Milla&Kudeki [2011] (parallel and perpendicular to B collision frequencies)
			 10: Neutral collisions
			 11: Neutral + Woodman Coulomb collisions
			 12: Neutral + Callen Coulomb collisions
			 13: Neutral + Milla&Kudeki Coulomb collisions
			nue: user-defined electron collision frequency [Hz]
			nui: user-defined ion collision frequency [Hz]
		
			Nn: Neutral density [1/m^3]
			Tn: Neutral temperature [K]
			neu_mass: neutral mass number [amu]
		
		Outputs:
			spcisr: Incoherent scatter spectrum
			f: vector of frequencies [Hz]
			efunc: structure of electron functions (Gordeyev, admittance, spectrum
			 component, particle ACF)
			ifunc: structure of ion functions (Gordeyev, admittance, spectrum
			 component, particle ACF)
			fr: Modified plasma frequency [Hz]
		
		Revisions:
			Author: Marco A. Milla
			Feb 9, 2008: Further testing and development is needed. Look for alternative
			methods for computing the Gordeyev integrals. General revision of the ChirpZ
			transform method is needed.
		
		'''

		# Setting bandwidth (sampling frequency) and number of frequency samples
		self.N = np.int64(N); N = self.N
		#self.fs = np.floor(fs); fs = self.fs
		self.fs = fs

		# Defining frequency array
		df = fs/N; f = (fs*(np.arange(0,N)-N/2))/N
		self.df = df
		self.f = f
		
		# Setting Bragg wavelength, wave number and radar frequency
		self.lambdaB = lambdaB # Bragg wavelength lambda_B = 2*lambda
		kB = 2*np.pi/lambdaB # Bragg wavenumber kB = 2*ko
		self.kB = kB # Bragg wavenumber kB = 2*ko
		self.fradar = phys_cons.c/(2*lambdaB) # Radar frequency
		
		# Setting aspect angle
		self.aspdeg = np.abs(aspdeg)	# Only possitive aspect angles
		aspdeg = self.aspdeg			# Aspect angle [deg]
		#asprad = np.radians(aspdeg)		# Aspect angle [rad]

		# Setting plasma configuration parameters
		self.plasma = plasma

#		Ni = plasma.Ni # Ion density [1/m^3]
#		mi = plasma.mi; # Ion mass [Kg]
		ion_mass = plasma.ion_mass # Ion mass
		ion_comp = plasma.ion_comp # Ion composition
		num_ions = plasma.num_ions # Number of ions
		ion_id = np.arange(num_ions)+1

		mu = plasma.mu; # Plasma temperature ratio (Te/Ti)
		
		#Ti = plasma.Ti; # Temperatures of ions [K]
		Ce = plasma.thermal_speed(0) # Electron thermal speed [m/s]
		Ci = plasma.thermal_speed(ion_id) # Ion thermal speed [m/s]

		Omge = plasma.gyro_frequency(0) # Electron gyro-frequency
		Omgi = plasma.gyro_frequency(ion_id) # Ion gyro-frequency

		he = plasma.Debye_length(0) # Electron Debye length
		# hi = sqrt((KB*Ti*e0)/(Ni*qe^2)); # Ion Debye length
		# hp = 1/sqrt(1/he^2+sum(1./hi.^2)); # Plasma Debye length

		# Electron and ion collisons
		nue = np.array(nue,ndmin=1)
		if len(nue)==0:
			nue = plasma.collision_frequency(0,modnu,kB=kB,aspdeg=aspdeg)
		nui = np.array(nui,ndmin=1)
		if len(nui)==0:
			nui = plasma.collision_frequency(ion_id,modnu,kB=kB,aspdeg=aspdeg)
		#print(nue)
		#print(nui)

		# Ion Gordeyev and admittance functions
		if (modgy==1) or (modgy==3):
			# Woodman [1967] and Milla&Kudeki [2011]
			modgyi = 1
		elif modgy==2:
			# Sulzer&Gonzalez [1999]
			modgyi = 1
		else:
			modgyi = 0

		ji = np.zeros((N,num_ions),dtype=complex);
		yi = np.zeros((N,num_ions),dtype=complex);
		acfi = np.zeros((N,num_ions),dtype=complex);

		for ni in range(num_ions):
			ji[:,ni], yi[:,ni], acfi[:,ni] = gordeyev_chirpz(N,fs,fdop,kB,Ci[ni],Omgi[ni],nui[ni],aspdeg=aspdeg,model=modgyi);

		# Electron Gordeyev and admittance functions
		if modgy==2:
			# Sulzer&Gonzalez [1999] (library for O+ plasma and 50MHz radar frequency)
			if (ion_mass==16) and (ion_comp==1):
			#	je = read_je_sulzer(f,Ne,Te,aspdeg);
				je = read_je_sg(); acfe = []
			else:
				warn.warn('ISRSPC: Sulzer&Gonzalez[1999] library was developed only for an O+ plasma.')
				return
			ye = 1j + 2*np.pi*f*je
		elif modgy==3:
			# Milla&Kudeki [2008] (library for O+ plasma and 3m radar wavelength)
			if (ion_mass==16) and (ion_comp==1):
				#je, acfe = read_je_mk(N,f0,df,Ne,Te,Ti,Bm,aspdeg);
				je, acfe = read_je_mk()
			else:
				warn.warn('ISRSPC: Milla&Kudeki[2011] library was developed only for an O+ plasma.')
				return
			ye = 1j + 2*np.pi*f*je
		else:
			je, ye, acfe = gordeyev_chirpz(N,fs,fdop,kB,Ce,Omge,nue,aspdeg=aspdeg,model=modgy);
		
#		print(nue)
  
		# Calculation of the incoherent scatter spectrum
		re_ji = np.dot(np.real(ji),ion_comp); re_je = np.real(je); tot_yi = np.dot(yi,ion_comp*mu);
		denisr = np.abs(1j*kB**2*he**2 + ye + tot_yi)**2;
		
		# Electron component of the spectrum
		spce = 2*((np.abs(1j*kB**2*he**2 + tot_yi)**2)*re_je)/denisr;
		# Ion component of the spectrum
		spci = 2*((np.abs(ye)**2)*re_ji)/denisr;
		
		# Normalized Spectrum
		spcisr = spce + spci;
		self.spc = spcisr
		self.nrcs = np.sum(spcisr)*(fs/N)

		# Returning electron functions
		self.je = je; self.ye = ye; self.acfe = acfe;
		self.spce = spce;

		# Returning ion functions
		self.ji = ji; self.yi = yi;	self.acfi = acfi;
		self.spci = spci;

		# Modified plasma frequency
		#if nargout>=5
		#	fr = sqrt(80.6*Ne*(1+3*kB^2*he^2)+(Omge*cos(asprad)/(2*pi))^2);
		#end



if __name__ == '__main__':
	
	Ne = 1E12
	Bm = 25E-6
	Te = 2000
#	Ti = [1000,1000]
#	ion_mass = [16,4]
#	ion_comp = [0.5,0.5]
	Ti = 1000.
	ion_mass = 16
	ion_comp = 1
	
	plasma = plasma_param.Plasma(Ne,Bm,Te,Ti,ion_mass,ion_comp)
	
	N = 2**15
	fs = 5E3
	fdop = -1000
	lambdaB = 3
	aspdeg = 15

#	fs = 40E3
#	frad = 445E6
#	lambdaB = phys_cons.c / frad / 2
#	aspdeg = 0.0
	modgy = 1
	modnu = 2

#	fs = 8E3
#	frad = 41E6
#	lambdaB = phys_cons.c / frad / 2
#	modgy = 1
#	modnu = 0

	isr_spc = isrSpectrum(N,fs,lambdaB,aspdeg,plasma,modgy=modgy,modnu=modnu,fdop=fdop)
	
	plt.figure(1)
	plt.plot(isr_spc.f,isr_spc.spc)
#	plt.plot(isr_spc.f,isr_spc.spce)
#	plt.plot(isr_spc.f,isr_spc.spci)
	plt.grid()
	plt.xlim([-fs/2,fs/2])
	plt.xlabel('Frequency [Hz]')
	plt.title('Eficiency factor %0.8f' % isr_spc.nrcs)
	plt.show()

	pass

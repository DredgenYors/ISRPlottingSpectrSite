'''
Created on Apr 19, 2012

@author: marco
'''
import numpy as np
import physical_constants as phys_cons

def  collision_frequencies(modnu,Ne,Te,Ti,ion_mass,ion_comp,Bm,kB=[],aspdeg=[],Nn=[],Tn=[],neu_mass=[]):

	'''
	def function [nue,nui] = collision_frequencies(modnu,Ne,Te,Ti,ion_mass,ion_comp,kB,Bm,aspdeg,Nn,Tn,neu_mass)

	COLLISION_FREQUENCIES

	Inputs:

		modnu: Collision frequency model
		0: No collisions
		1: Woodman [1967] (Ion collision frequencies), [2004] (Electron collision frequency)
		2: Callen [2003] (Electron and ion Coulomb collision frequencies)
		3: Milla&Kudeki [2010] (parallel and perpendicular to B collision frequencies)
		10: Neutral collision frequencies
		11: Neutral + Woodman Coulomb collision frequencies
		12: Neutral + Callen Coulomb collision frequencies
		13: Neutral + Milla&Kudeki Coulomb collision frequencies

		Ne: Electron density [1/m^3]
		Te: Electron temperature [K]
		Ti: Ion temperature [K]
		ion_mass: ion mass number [amu]
		ion_comp: composition (adding up to unity)

		kB: Bragg radar wavenumber [1/m]
		Bm: Magnetic field [Tesla]
		aspdeg: Aspect angle [deg] (0 deg perp to B)

		Nn: Neutral density [1/m^3]
		Tn: Neutral temperature [K]
		neu_mass: neutral mass number [amu]

	Outputs:
		nue, nui: electron and ion collision frequencies

	Revisions:
		Author: Marco Milla
		Feb 9, 2008: Further testing is needed. Look for references for the neutral
		collision formulae.

	'''

	#Ne,Te,Ti,ion_mass,ion_comp,Bm
	#Nn,Tn,neu_mass

	num_ions = len(ion_mass);
	Ni = Ne*ion_comp;

	# Coulomb collision frequencies
	modcc = (modnu % 10)
	if modcc == 1:
		# Woodman [1967] (only for H+,He+,O+)
		nue, nui = nu_Woodman1967(Ne,Ni,Te,Ti,ion_mass);
		# Woodman [2004] (aspect angle dependent - only for O+)
#		if (ion_mass==16) and (kB==2*np.pi/3):
		if (ion_mass==16):
			if (Bm!=0):
				nue = nue_Woodman2004(Ne,Te,kB,aspdeg);
	elif modcc == 2:
		# Callen [2003] (for a multi-component plasma)
		nue, nui = nu_Callen(Ne,Ni,Te,Ti,ion_mass);
	elif modcc == 3:
		# Milla&Kudeki [2011] (parallel & perpendicular to B collision frequencies)
		nue, nui = nu_MK2011(Ne,Ni,Te,Ti,ion_mass);
	else:
		# Zero collision frequency
		nue = 0;
		nui = np.zeros(num_ions);
 
	# Neutral collision frequencies
	#if fix(modnu/10)==1
	#	[nuen,nuin] = nu_neutral(ion_mass,Nn,Tn,neu_mass);
	#	nue = nue + nuen;
	#	nui = nui + nuin;
	#end

	return nue, nui


'''
%==============================================================================%
'''

# Electron Coulomb collision frequency [Woodman,2004]

def nue_Woodman2004(Ne,Te,kB,aspdeg):
	#function nue = nue_Woodman2004(Ne,Te,kB,aspdeg)

	#fradar = 49.92E6; kB = 2*pi*(fradar/(c/2)); % Using Jicamarca's wavelength by default.

	# It is assumed that electrons and ions have same density and temperature.
	# Ni = Ne; % Ion density [1/m^3]
	# Ti = Te; % Ion temperature [K]

	KB = phys_cons.KB
	me = phys_cons.me
	qe = phys_cons.qe
	e0 = phys_cons.e0

	Ce = np.sqrt(KB*Te/me); # Electron thermal speed [m/s]

	he2 = (KB*Te*e0)/(Ne*qe**2); # Electron Debye length [m]
	# hi2 = (KB*Ti*e0)./(Ni*qe**2); # Ion Debye length [m]
	# hp = 1/sqrt(1/he2+sum(1/hi2)); % Plasma Debye length [m]
	hp = np.sqrt(he2/2);

	lnD = np.log(24*np.pi*Ne*hp**3);
	nue0 = Ne*qe**4*lnD/(8*np.sqrt(2)*np.pi*e0**2*me**2*Ce**3);

	sinthetac = 2*np.pi*nue0/(kB*Ce);
	asprad = np.deg2rad(aspdeg)
	sinnorm = np.sin(asprad)/sinthetac;
	nue = nue0*(1.06 + 7.55*sinnorm - 2*sinnorm**2 + 0.27*sinnorm**3);
	nue = np.min([nue,nue0*11.7]);

	return nue


# Electron and ion Coulomb collision frequencies [Woodman,1967]

def nu_Woodman1967(Ne,Ni,Te,Ti,ion_mass):
	
	#function [nue,nui] = nu_Woodman1967(Ne,Ni,Te,Ti,ion_mass)

	KB = phys_cons.KB
	me = phys_cons.me
	mp = phys_cons.mp
	qe = phys_cons.qe
	e0 = phys_cons.e0

	mi = mp*ion_mass; # Ion mass [kg]

	Ce = np.sqrt(KB*Te/me); # Electron thermal speed [m/s]
	Ci = np.sqrt(KB*Ti/mi); # Ion thermal speed [m/s]

	he2 = (KB*Te*e0)/(Ne*qe**2); # Electron Debye length [m]
	hi2 = (KB*Ti*e0)/(Ni*qe**2); # Ion Debye length [m]
	hp = 1/np.sqrt(1/he2+np.sum(1/hi2)); # Plasma Debye length [m]

	num_ions = len(ion_mass);
	W = 0.601*np.eye(num_ions+1);
	Hy = np.argwhere(ion_mass==1);
	He = np.argwhere(ion_mass==4);
	Ox = np.argwhere(ion_mass==16);
	ee = num_ions;
	W[Hy,Hy] = 0.601; W[Hy,He] = 0.853; W[Hy,Ox] = 1.015; W[Hy,ee] = 0.0176;
	W[He,Hy] = 0.352; W[He,He] = 0.601; W[He,Ox] = 0.854; W[He,ee] = 0.0090;
	W[Ox,Hy] = 0.185; W[Ox,He] = 0.351; W[Ox,Ox] = 0.601; W[Ox,ee] = 0.0048;
	W[ee,Hy] = 1.127; W[ee,He] = 1.128; W[ee,Ox] = 1.128; W[ee,ee] = 0.6010;

	lnD = np.log(24*np.pi*Ne*hp**3);

	Ns = np.concatenate((Ni,[Ne]))
	nue = (qe**4*lnD)/(8*np.sqrt(2)*np.pi*e0**2*me**2*Ce**3);
	nue = nue*np.dot(Ns,W[ee,:]);

	nui = (qe**4*lnD)/(8*np.sqrt(2)*np.pi*e0**2*mi**2*Ci**3);
	nui = nui*np.dot(Ns,W[0:num_ions,:].T);

	return nue, nui

'''
%==============================================================================%
'''

# Electron and ion Coulomb collision frequencies [Callen,2003]

def nu_Callen(Ne,Ni,Te,Ti,ion_mass,multi=0):

	#function [nue,nui] = nu_Callen(Ne,Ni,Te,Ti,ion_mass)
	#physical_constants;

	KB = phys_cons.KB
	me = phys_cons.me
	mp = phys_cons.mp
	qe = phys_cons.qe
	e0 = phys_cons.e0

	mi = mp*ion_mass; # Ion mass [Kg]

	Ce = np.sqrt(KB*Te/me); # Electron thermal speed [m/s]
	Ci = np.sqrt(KB*Ti/mi); # Ion thermal speed [m/s]

	he2 = (KB*Te*e0)/(Ne*qe**2); # Electron Debye length
	hi2 = (KB*Ti*e0)/(Ni*qe**2); # Ion Debye length
	hp = 1/np.sqrt(1/he2+np.sum(1./hi2)); # Plasma Debye length

	# Electron Coulomb collisions
	mee = (me*me)/(me+me); Cee2 = Ce**2+Ce**2; Cee = np.sqrt(Cee2);
	lnDee = np.log(12*np.pi*e0*mee*Cee2*hp/qe**2);
	nuee = (qe**4*Ne*lnDee)/(6*np.sqrt(2*np.pi**3)*e0**2*me*mee*Cee**3);

	mei = (me*mi)/(me+mi); Cei2 = Ce**2+Ci**2; Cei = np.sqrt(Cei2);
	lnDei = np.log(12*np.pi*e0*mei*Cei2*hp/qe**2);
	nuei = (qe**4*Ni*lnDei)/(6*np.sqrt(2*np.pi**3)*e0**2*me*mei*Cei**3);

	nue = nuee + np.sum(nuei);

	# Ion Coulomb collisions
	mie = (mi*me)/(mi+me); Cie2 = Ci**2+Ce**2; Cie = np.sqrt(Cie2);
	lnDie = np.log(12*np.pi*e0*mie*Cie2*hp/qe**2);
	nuie = (qe**4*Ne*lnDie)/(6*np.sqrt(2*np.pi**3)*e0**2*mi*mie*Cie**3);

	nuii = np.zeros(len(Ni));
	for ii in range(len(Ni)):
		mii = (mi[ii]*mi)/(mi[ii]+mi);
		Cii2 = Ci[ii]**2+Ci**2; Cii = np.sqrt(Cii2);
		lnDii = np.log(12*np.pi*e0*mii*Cii2*hp/qe**2);
		nuii_i = (qe**4*Ni*lnDii)/(6*np.sqrt(2*np.pi**3)*e0**2*mi[ii]*mii*Cii**3);
		nuii[ii] = np.sum(nuii_i);
	
	nui = nuie + nuii;

	if multi==0:
		return nue, nui
	else:
		return nuee, nuei, nuie, nuii

'''
%==============================================================================%
'''
# Electron and ion Coulomb collision frequencies [Milla&Kudeki,2011]

def nu_MK2011(Ne,Ni,Te,Ti,ion_mass):

	#function [nue,nui] = nu_MK2011(Ne,Ni,Te,Ti,ion_mass)

	nuee, nuei, nuie, nuii = nu_Callen(Ne,Ni,Te,Ti,ion_mass,multi=1);

	nue = [np.sum(nuei),np.sum(nuei)+nuee];
	nui = nuie + nuii;

	return nue, nui


'''
%==============================================================================%

% Electron and ion neutral collision frequencies
function [nue,nui] = nu_neutral(ion_mass,Nn,Tn,neu_mass)

% ion_mass: ion molecular mass [amu]
% Nn: Neutral density [1/m^3]
% Tn: Neutral temperature [K]
% neu_mass: neutral molecular mass [amu]
%
% Note: This routine only considers the following neutral and ion species.
%	Neutrals [N2:28, O2:32, O:16]
%	Ions [NO+:30, O2+:32, O+:16]

physical_constants;

mi = mp*ion_mass; % Ion mass [Kg]

N2 = find(neu_mass==28); if ~isempty(N2), Nn_N2 = Nn(N2); else Nn_N2 = 0; end
O2 = find(neu_mass==32); if ~isempty(O2), Nn_O2 = Nn(O2); else Nn_O2 = 0; end
O  = find(neu_mass==16); if ~isempty(O) , Nn_O  = Nn(O) ; else Nn_O  = 0; end

% Electron-neutral collision frequency
nue = ( 4.11*Nn_N2*(Tn/300)^(0.95) + ...
		2.95*Nn_O2*(Tn/300)^(0.79) + ...
     	1.09*Nn_O *(Tn/300)^(0.85) ) * 1E-26*qe/me;

% Ion-neutral collision frequencies
nui = zeros(1,length(ion_mass));

% NO+ ion [30]
NOi = find(ion_mass==30);
nui(NOi) = ( 1.07*Nn_N2*(Tn/500)^(-0.16) + ...
			 1.06*Nn_O2*(Tn/500)^(-0.16) + ...
			 0.6 *Nn_O *(Tn/500)^(-0.19) ) * 1E-22*qe./mi(NOi);

% O2+ ion [32]
O2i = find(ion_mass==32);
nui(O2i) = ( 1.08*Nn_N2*(Tn/500)^(-0.17 ) + ...
			 2.02*Nn_O2*(Tn/500)^( 0.37 ) + ...
   			 0.61*Nn_O *(Tn/500)^(-0.019) ) * 1E-22*qe./mi(O2i);

% O+ ion [16]
Oi = find(ion_mass==16);
nui(Oi) = ( 0.89*Nn_N2*(Tn/500)^(-0.20) + ...
			1.16*Nn_O2*(Tn/500)^( 0.05) + ...
			0.89*Nn_O *(Tn/500)^( 0.36) ) * 1E-22*qe./mi(Oi);

'''

#class MyClass(object):
#	'''
#	classdocs
#	'''


#	def __init__(selfparams):
#		'''
#		Constructor
#		'''
#		

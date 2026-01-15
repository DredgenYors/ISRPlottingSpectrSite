'''
Created on Apr 21, 2012

@author: marco
'''
import numpy as np
from collision_frequencies import collision_frequencies

# Physical Constants (MKS)
import physical_constants as phys_cons

class Plasma(object):
	'''
	Plasma
	'''
	
	def __init__(self,Ne,Bm,Te,Ti,ion_mass,ion_comp):
		'''
		Constructor
		'''
		Ne = float(Ne)
		Te = float(Te)
		Ti = np.array(Ti,dtype=float,ndmin=1)
		ion_mass = np.array(ion_mass,dtype=float,ndmin=1)
		ion_comp = np.array(ion_comp,dtype=float,ndmin=1)

		self.Ne = Ne
		self.Bm = Bm
		self.Te = Te
		self.Ti = Ti
		self.mu = Te/Ti

		self.ion_mass = ion_mass
		self.ion_comp = ion_comp
		
		self.Ni = Ne*ion_comp
		self.num_ions = len(ion_mass)
		
		self.me = phys_cons.me
		self.mi = ion_mass*phys_cons.mp
		
		self.Ns = np.concatenate((np.array(self.Ne,ndmin=1),self.Ni))
		self.Ts = np.concatenate((np.array(self.Te,ndmin=1),self.Ti))
		self.ms = np.concatenate((np.array(self.me,ndmin=1),self.mi))
		
	def thermal_speed(self,sp):
		Cs = np.sqrt(phys_cons.KB*self.Ts[sp]/self.ms[sp])
		return Cs
	
	def plasma_frequency(self,sp):
		ws = np.sqrt((self.Ns[sp]*phys_cons.qe**2)/(self.ms[sp]*phys_cons.e0))
		return ws
		
	def Debye_length(self,sp):
		hs = np.sqrt((phys_cons.KB*self.Ts[sp]*phys_cons.e0)/(self.Ns[sp]*phys_cons.qe**2))
		return hs
	
	def gyro_frequency(self,sp):
		Omgs = phys_cons.qe*self.Bm/self.ms[sp]
		return Omgs
	
	def collision_frequency(self,sp,modnu,kB=[],aspdeg=[]):
		
		nue, nui = collision_frequencies(modnu,self.Ne,self.Te,self.Ti,self.ion_mass,self.ion_comp,self.Bm,kB=kB,aspdeg=aspdeg)
		sp = np.array(sp,ndmin=1)

		nus = np.empty_like(sp,dtype=float)
		for id in range(len(sp)):
			if sp[id]==0:
				if modnu==3:
					nus = nue
				else:
					nus[id] = nue
			else:
				nus[id] = nui[sp[id]-1]
		return nus
	
#	def Coulomb_collision_frequency(self,sp,modnu):
#		nu = 1
#		return nu
		
#	def neutral_collision_frequency(self,sp,neutral):
#		nu = 1
#		return nu


class Neutral(object):
	
	def __init__(self,Nn,Tn,neu_mass):

		self.Nn = Nn
		self.Tn = Tn
		self.neu_mass = neu_mass
		
		self.mn = neu_mass*phys_cons.mp

		
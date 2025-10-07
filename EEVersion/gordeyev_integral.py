'''
Created on Nov 27, 2012

@author: marco
'''

import numpy as np
from scipy.optimize import fsolve
import warnings as warn

def gordeyev_chirpz(N,fs,fdop,kB,C,Omg,nu,aspdeg,model=0):

# function [j,y,expf,iter] = gordeyev_chirpz(N,wo,dw,kb,C,Omg,nu,asprad,model)
#
# Calculates ion or electron admittance function.
# This routine includes collision effects.
#
# Inputs:
# N: Number of frequency points
# Fs: Sampling frequency - Frequency bandwidth
# wo: Frequency offset (2*pi*df) [rad/sec] 
# dw: Frequency sampling period (2*pi*df) [rad/sec]
# C: Thermal velocity [m/sec]
# Omg: Gyro-frequency [rad/sec]
# nu: Collision frequency [Hz]
# aspdeg: Complementary angle between radar beam and geomagnetic field [deg]
# kB: Bragg scatter wave number [1/m]
# model: 0: Farley's (BGK), 1: Woodman's (Fokker Planck).

	C = np.array(C,ndmin=1)
	Omg = np.array(Omg,ndmin=1)
	nu = np.array(nu,ndmin=1)
	
	N = np.int64(N)

	kB_C = np.sqrt(2)*kB*C[0]
	ws = 2*np.pi*fs/kB_C # Normalized sampling frequency
	wdop = 2*np.pi*fdop/kB_C
	wo = -ws/2 - wdop
	dw = ws/N; # Normalized delta frequency

	Larmor = np.abs(Omg[0])/kB_C # Larmor or Normalized Gyro-frequency
	aspdeg = np.abs(aspdeg)
	asprad = np.deg2rad(aspdeg)
	
	sin2_asp = np.sin(asprad)**2
	cos2_asp = np.cos(asprad)**2

	Psi = nu[0]/kB_C; # Normalized Collision frequency
	
	if (len(nu)>=2):
		Psi2 = nu[1]/(kB_C);
	else:
		Psi2 = Psi; # Normalized Collision frequency (perpendicular)

	#options = optimset('Display','off','TolFun',1E-10,'TolX',1E-10,'MaxFunEvals',1000);

	# Calculating the Normalized sampling interval (tau)
	if (Larmor==0) or (aspdeg==90):
		# If Earth Magnetic field is zero or aspect angle is pi/2,
		if model==1:
			if (Psi>10):
				tau = (2*Psi**2+1)/Psi;
			elif (np.abs(Psi)<0.1):
				tau = 2+Psi*(Psi**2+2*Psi+4)/(Psi**2+6);
			elif (Psi<-10):
				tau = np.log(2*Psi**2+1)/abs(Psi);
			else:
				fun = lambda t: (Psi*t+np.expm1(-Psi*t))/Psi**2-2;
				#tau = lsqnonlin(fun,1,0,np.Inf,options);
				tau = fsolve(fun,1); tau = tau[0];
		else:
			if (Psi>0):
				tau = 2/(np.sqrt(Psi**2+1)+Psi);
			else:
				tau = 2*(np.sqrt(Psi**2+1)-Psi);
	else:
		# If Earth Magnetic field is not zero,
		if (aspdeg!=0):
			# If Magnetic aspect angle is not zero,
			if model==1:
				if (Psi==0) and (Psi2==0):
					tau = 2/np.sin(asprad);
				else:
					gamma = np.arctan2(Psi2,Larmor)
					fun = lambda t: sin2_asp*(Psi*t+np.expm1(-Psi*t))/Psi**2 + \
									cos2_asp*(Psi2*t+np.cos(2*gamma)-np.exp(-Psi2*t)*np.cos((Larmor*t-2*gamma)*(Larmor*t<=2*gamma)))/(Psi2**2+Larmor**2) - 2;
					#tau = lsqnonlin(fun,1,0,np.Inf,options);
					tau = fsolve(fun,1); tau = tau[0];
			else:
				if (Psi>0):
					tau = 2/(np.sqrt(Psi**2+np.sin(asprad)**2)+Psi);
				else:
					tau = 2*(np.sqrt((Psi/np.sin(asprad))**2+1)-Psi/np.sin(asprad))/np.sin(asprad);
		else:
			# If Magnetic aspect angle is zero
			if model==1:
					gamma = np.arctan2(Psi2,Larmor);
					if (Psi2>10) or (Larmor>10):
						tau = 2*(Psi2**2+Larmor**2-np.cos(2*gamma)/2)/Psi2;
					elif (Psi2>0):
						fun = lambda x: (x+np.cos(2*gamma)-np.exp(-x)*np.cos((Larmor*x/Psi2-2*gamma)*(Larmor*x/Psi2<=2*gamma)))/(Psi2**2+Larmor**2)-2;
						#x = lsqnonlin(fun,(Psi2^2+Larmor^2),0,np.Inf,options);
						x = fsolve(fun,(Psi2**2+Larmor**2));
						tau = x[0]/Psi2;
					else:
						tau = np.Inf;
			else:
					if (Psi>0):
						tau = 1/Psi;
					else:
						tau = np.Inf;

	# Nm: Minimum number of samples per time interval (tau)
	# Ns: Number of samples per time interval (tau) 
	Nm = 256; Ns = np.maximum(N,Nm);
	dt = tau/Ns; # Sampling period
	order = 1; # Order of the Chirp-z integrator

	# The maximun frequency must be lower than (fs/2+abs(fdop)) (by a factor of 10)
	# If dt*(zs/2)>(pi/10), we have to reduce the sampling period.
	fact = np.ceil((dt*(ws/2+np.abs(wdop)))/(np.pi/10));
	dt = dt/fact;

	# Redefining the sampling period taking into account the Larmor frequency
	# If Larmor*dt>2*pi/Nl then use at least Nl_min intervals per period.
	if (Larmor>0):
		Nl = np.ceil((2*np.pi)/(Larmor*dt)); # Number of intervals per Larmor period
		Nl_min = 4*np.maximum(order,4); # Minimum Number of intervals per Larmor period
		if (Nl<Nl_min):
			Nl = Nl_min;
		else:
			Nl = np.ceil(Nl/Nl_min)*Nl_min; # Test this option
		dt = (2*np.pi)/Larmor/Nl;
	
	
	# Number of iterations for chirp-z transform
	T = 1.5*tau; niter = np.ceil(T/(Ns*dt));

	if ((ws/2+np.abs(wdop))*dt>np.pi):
		warn.warn('Gordeyev:OutOfNyquist','Out of Nyquist');

	# Computation of admittance function
	# Note: If Psi (Collision frequency) is zero both
	# Woodman's (FP) and Farley's (BGK) models are the same.
	if (Larmor==0):
		# If Magnetic field is zero	
		if (model==1) and (Psi!=0):
			# Woodman[1967] (Fokker Planck collision model)
			#f0 = lambda t: (Psi*t-1+np.exp(-Psi*t))/(2*Psi**2);
			#f1 = lambda t: (1-np.exp(-Psi*t))/(2*Psi);
			f0 = lambda t: (Psi*t+np.expm1(-Psi*t))/(2*Psi**2);
			f1 = lambda t: -np.expm1(-Psi*t)/(2*Psi);
		else:
			# Farley [1966] (BGK collision model)
			f0 = lambda t: Psi*t + (t**2)/4;
			f1 = lambda t: t/2;
	else:
		# If Magnetic field is not zero
		if (model==1) and (Psi!=0):
			# Woodman, 1967 (Fokker Planck collision model)
			gamma = np.arctan2(Psi2,Larmor)
			#f0 = lambda t: sin2_asp*(Psi*t-1+np.exp(-Psi*t))/(2*Psi**2) + \
			#				cos2_asp*(Psi2*t+np.cos(2*gamma)-np.exp(-Psi2*t)*np.cos(Larmor*t-2*gamma))/(2*(Psi2**2+Larmor**2));
			#f1 = lambda t: sin2_asp*(1-np.exp(-Psi*t))/(2*Psi) + \
			#				cos2_asp*(Psi2+Psi2*np.exp(-Psi2*t)*np.cos(Larmor*t-2*gamma)+Larmor*np.exp(-Psi2*t)*np.sin(Larmor*t-2*gamma))/(2*(Psi2**2+Larmor**2));
			f0 = lambda t: sin2_asp*(Psi*t+np.expm1(-Psi*t))/(2*Psi**2) + \
						   cos2_asp*(Psi2*t+np.cos(2*gamma)-np.exp(-Psi2*t)*np.cos(Larmor*t-2*gamma))/(2*(Psi2**2+Larmor**2));
			f1 = lambda t: -sin2_asp*np.expm1(-Psi*t)/(2*Psi) + \
						   cos2_asp*(np.sin(gamma)+np.exp(-Psi2*t)*np.sin(Larmor*t-gamma))/(2*np.sqrt(Psi2**2+Larmor**2));
		else:
			# Farley, 1966 (BGK collision model)
			f0 = lambda t: Psi*t + sin2_asp*(t**2/4) + cos2_asp*(np.sin(Larmor*t/2)**2/Larmor**2);
			f1 = lambda t: sin2_asp*t/2 + cos2_asp*np.sin(Larmor*t)/(2*Larmor);


	# Defining integrands of Gordeyev and admittance integrals
#	funj = lambda t: np.exp(-f0(t));
#	funy = lambda t: f1(t)*np.exp(-f0(t));
	funj = lambda n: np.exp(-f0(n*dt));
	funy = lambda n: f1(n*dt)*np.exp(-f0(n*dt));

	w = (np.arange(0,N) - N/2.)*dw - wdop; # Normalized frequency samples
	indl = np.where(np.abs(w)<=1)[0]; Nl = len(indl);
	indh = np.where(np.abs(w)>1)[0];

	niter = 0;

	# Calculating Chirp-z transform
	js, it = chirpz_isr(funj,Ns,wo*dt,dw*dt,niter,order);
	js = dt*js[0:N];
	if (Nl<N):
		ys, it = chirpz_isr(funy,Ns,wo*dt,dw*dt,niter,order);
		ys = dt*1j*ys[0:N];
	else:
		ys = np.empty(N,dtype=complex)

	if (model==0) and (Psi!=0):
		ys[indl] = 1j+(w[indl]-1j*Psi)*js[indl]; # Low frequency
		js[indh] = (ys[indh]-1j)/(w[indh]-1j*Psi); # High frequency
		ys = ys/(1-Psi*js);
		js = js/(1-Psi*js);
	else:
		ys[indl] = 1j+w[indl]*js[indl]; # Low frequency
		js[indh] = (ys[indh]-1j)/w[indh]; # High frequency
	
	js = js/(kB_C);

	acfs = funj(np.arange(0,N)*dt*it);

#	js = []; ys = []; acfs = []
	return js, ys, acfs


def chirpz_isr(fun,N,wo,dw,niter=0,order=0):
	'''
	Chirpz_isr calculates Chirp Z-Transform of fun[n]
	 (approximation of the Fourier Transform of fun(t))
	'''

	#if ~exist('niter','var'), niter = 1; end;
	#if ~exist('order','var'), order = 0; end;

	# Note: 1 FFT of 2N-vector costs more than 2 FFTs of N-vector
	N2 = 2**np.int64(np.ceil(np.log2(2*N-1))); M = N2 - N;
	niter = np.ceil(N*niter/M);

	n = np.arange(0,M);

	#W = np.exp(1j*dw*(n**2)/2);
	phi = dw*(n**2)/2;
	W = np.cos(phi)+1j*np.sin(phi);
	FW = np.fft.fft(np.concatenate((W[0:N],[np.exp(1j*dw*(M**2)/2)],W[-1:0:-1])),N2);

	#W1 = np.exp(-1j*(2*wo+dw*n)*n/2);
	phi = (2*wo+dw*n)*n/2;
	W1 = np.cos(phi)-1j*np.sin(phi);

	omg = wo+dw*n[0:N];
	F = np.zeros(N);
	err = 1; k = 0;
	while (err>1E-15) and ((k<niter) or (niter==0)):
		nk = n + k*M;
		[num,den] = compquadrule(nk,order,0);
		x = fun(nk)*num*W1;
		I = np.fft.ifft(np.fft.fft(x,N2)*FW,N2);
		#dF = I[0:N]*np.exp(-1j*omg*(k*M));
		phi = omg*(k*M);
		dF = I[0:N]*(np.cos(phi)-1j*np.sin(phi));
		F = F + dF;
		err = np.amax(np.abs(dF/F));
		k = k + 1;

	R = 0;
	if (order>=1):
		aux = np.mod(k*M,order); aux = np.mod(order-aux,order);
		ind = k*M+np.arange(0,aux+1);
		[num,den] = compquadrule(ind,order,1);
		for i in range(aux+1):
			R = fun(ind[i])*num[i]*np.exp(-1j*omg*ind[i]) + R;

	F = (F*np.conj(W[0:N])+R)/den;
		
	return F, k


def compquadrule(ind,order,last=0):
	'''
	COMPQUAD_WEIGHTS: Newton-Cotes composite quadrature weights for numerical integration.
	'''

	#if ~exist('last','var'), last = 0; end

	N = len(ind); aux = np.mod(ind,order);
	num = np.ones(N); den = 1;

	if order==1:
		num[:] = 2;
		den = 2;
	elif order==2:
		num[aux==0] = 2;
		num[aux==1] = 4;
		den = 3;
	elif order==3:
		num[aux==0] = 6;
		num[aux==1] = 9;
		num[aux==2] = 9;
		den = 8;
	elif order==4:
		num[aux==0] = 28;
		num[aux==1] = 64;
		num[aux==2] = 24;
		num[aux==3] = 64;
		den = 45;
	elif order==5:
		num[aux==0] = 190;
		num[aux==1] = 375;
		num[aux==2] = 250;
		num[aux==3] = 250;
		num[aux==4] = 375;
		den = 288;
	elif order==6:
		num[aux==0] = 82;
		num[aux==1] = 216;
		num[aux==2] = 27;
		num[aux==3] = 272;
		num[aux==4] = 27;
		num[aux==5] = 216;
		den = 140;
	elif order==7:
		num[aux==0] = 10514;
		num[aux==1] = 25039;
		num[aux==2] = 9261;
		num[aux==3] = 20923;
		num[aux==4] = 20923;
		num[aux==5] = 9261;
		num[aux==6] = 25039;
		den = 17280;
	elif order==8:
		num[aux==0] = 7912;
		num[aux==1] = 23552;
		num[aux==2] = -3712;
		num[aux==3] = 41984;
		num[aux==4] = -18160;
		num[aux==5] = 41984;
		num[aux==6] = -3712;
		num[aux==7] = 23552;
		den = 14175;

	if (order>=1) and (order<=8):
		num[ind==0] = num[ind==0]/2;
		if (last>0):
			if (aux[-1]+1)<=N:
				num[-aux[-1]-1] = num[-aux[-1]-1]*(ind[-aux[-1]-1]!=0)/2;
			num[np.max(N-aux[-1],0):N] = 0;
		if (last==2) and (aux[-1]!=0):
			last_order = np.min(aux[-1],N-1);
			last_ind = np.arange(0,last_order+1);
			[last_num, last_den] = compquadrule(last_ind,last_order,1);
			num[N-last_order-1:N] = num[N-last_order-1:N]+last_num*den/last_den;
	
	return num, den


def read_je_sg():
	pass

	
def read_je_mk():
	pass

if __name__ == '__main__':
	pass
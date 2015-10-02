import numpy as np
import matplotlib.pyplot as pl
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.fftpack as fft

def bin_ndarray(ndarray, new_shape, operation='sum'):
   """
   Bins an ndarray in all axes based on the target shape, by summing or
       averaging.

   Number of output dimensions must match number of input dimensions.

   Example
   -------
   >>> m = np.arange(0,100,1).reshape((10,10))
   >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
   >>> print(n)

   [[ 22  30  38  46  54]
    [102 110 118 126 134]
    [182 190 198 206 214]
    [262 270 278 286 294]
    [342 350 358 366 374]]
   """
   if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
      raise ValueError("Operation {} not supported.".format(operation))
   if ndarray.ndim != len(new_shape):
      raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
   compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
   flattened = [l for p in compression_pairs for l in p]
   ndarray = ndarray.reshape(flattened)
   for i in range(len(new_shape)):
      if operation.lower() == "sum":
         ndarray = ndarray.sum(-1*(i+1))
      elif operation.lower() in ["mean", "average", "avg"]:
         ndarray = ndarray.mean(-1*(i+1))
   return ndarray

def myFFT(x):
	"""
	Return the FFT of a real signal taking into account some normalization
	
	Parameters
	----------
	x : float 
	    Signal time series
	
	Returns
	-------
	float : Fourier coefficients of the real signal
	"""
	out = fft.rfft(x)
	return out / np.sqrt(len(out))

def myIFFT(x):
	"""
	Return the IFFT for a real signal taking into account some normalization
	
	Parameters
	----------
	x : float 
	    Fourier coefficients
	
	Returns
	-------
	float : signal
	"""
	out = fft.irfft(x)
	return out * np.sqrt(len(out))

def myTotalPower(f):
	"""
	Return the power spectrum of a signal
	
	Parameters
	----------
	f : float
	    Signal Fourier coefficients
	
	Returns
	-------
	float : total power
	"""
	return f[0]**2 + 2.0*np.sum(f[1:]**2)

class randomDemodulator(object):
	"""Summary
	
	Returns
	-------
	TYPE : Demodulate I and Q signals together
	"""
	def __init__(self, totalTime, dt, dtIntegration, stokes, beta, signalToNoise=0.0, seed=0, modulationType=0):
		"""Summary
		
		Parameters
		----------
		totalTime : TYPE
		    Description
		dt : TYPE
		    Description
		dtIntegration : TYPE
		    Description
		seed : int, optional
		    Description
		
		Returns
		-------
		TYPE : Description
		"""
		
		self.totalTime = totalTime
		self.dt = dt
		self.dtIntegration = dtIntegration
		self.seed = seed
		self.signalToNoise = signalToNoise
		self.modulationType = modulationType

		if (self.seed != 0):
			np.random.seed(self.seed)

# Read seeing power spectrum
		self.powerLites = np.loadtxt('powerSpectrumSeeing.dat')
		self.powerLites[:,1] = 10.0**self.powerLites[:,1]

# Number of samples of the original sample
		self.nSteps = int(totalTime / dt)
		self.times = np.arange(self.nSteps) * self.dt

# Frequency axis
		self.freq = fft.rfftfreq(self.nSteps, d=dt)

# Betas and Stokes parameters
		self.beta = beta
		self.stokes = stokes		

# Generate Gaussian noise with unit variance and multiply by the square root of the power spectrum
# to generate the noise with the appropriate power spectrum
		noise = np.random.randn(self.nSteps)
		noiseFFT = myFFT(noise)

		self.powerSeeing = np.interp(np.abs(self.freq), self.powerLites[:,0], self.powerLites[:,1])
		self.powerSeeing[0] = 0.0
		
		self.seeingFFT = np.sqrt(self.powerSeeing) * noiseFFT
		self.seeingFFT /= np.sqrt(myTotalPower(self.seeingFFT))
		self.seeing = myIFFT(self.seeingFFT)

# Make sure that the total power is unity
		print 'Total variance = ', np.sum(self.seeing**2), myTotalPower(self.seeingFFT)

# Compute the signal and its power spectrum
		self.signal = [None] * 4
		for i in range(4):
			self.signal[i] = self.stokes[i]*(1.0 + self.beta[i] * self.seeing)

# Generate modulation using a lambda/4 and lambda/2 polarimeter with random angles
		# self.modulation = [np.ones(self.nSteps), 2.0*np.random.rand(self.nSteps)-1.0, 2.0*np.random.rand(self.nSteps)-1.0, 2.0*np.random.rand(self.nSteps)-1.0]
		if (self.modulationType == 0):
			self.alphaModulation = 0.5*np.pi*np.random.rand(self.nSteps)
			self.betaModulation = 0.5*np.pi*np.random.rand(self.nSteps)
		else:
			temp = np.load('alphaBetaSamples.npz')
			self.alphaModulation = temp['arr_0'][0:self.nSteps]
			self.betaModulation = temp['arr_1'][0:self.nSteps]

		self.modulation = [np.ones(self.nSteps), \
			np.cos(2.0*self.alphaModulation) * np.cos(2.0*(self.alphaModulation-2.0*self.betaModulation)),\
			np.sin(2.0*self.alphaModulation) * np.cos(2.0*(self.alphaModulation-2.0*self.betaModulation)),\
			np.sin(2.0*(2.0*self.betaModulation-self.alphaModulation))]

		self.integrationTime = self.dtIntegration
		self.lengthSample = int(self.dtIntegration / self.dt)
		self.nSamples = int(self.dt / self.dtIntegration * self.nSteps)

		self.signalIntegrated = [None] * 2
		for i in range(2):
			temp = self.signal[0] * self.modulation[0]
			sign = (-1.0)**i
			for j in range(1,4):
				temp += sign * self.signal[j] * self.modulation[j]
			self.signalIntegrated[i] = bin_ndarray(temp, (self.nSamples,), operation='sum') 
			self.signalIntegrated[i] += np.mean(self.signalIntegrated[i]) / self.signalToNoise * np.random.randn(self.nSamples)

		self.tIntegrated = np.arange(self.nSamples) * self.dtIntegration

# Generate modulation matrix
		self.sparseM = [None] * 4
		self.sparseMStar = [None] * 4

		for state in range(4):

			sparseData = []
			sparseRow = []
			sparseCol = []
			loop = 0
			for i in range(self.nSamples):
				for j in range(self.lengthSample):
					sparseData.append(self.modulation[state][loop])
					sparseRow.append(i)
					sparseCol.append(loop)

					loop += 1

			self.sparseM[state] = sp.coo_matrix((sparseData, (sparseRow, sparseCol)), shape=(self.nSamples, self.nSteps))
			self.sparseMStar[state] = self.sparseM[state].transpose(copy=True)

		self.factor = 2*np.ones(self.nSteps)
		self.factor[0] = 1.0


	def forward(self, signal, beta, ray):
		return self.sparseM[ray].dot(1.0+beta*signal)

	def forwardPartial(self, signal, ray):
		return self.sparseM[ray].dot(signal)

	def backward(self, z, ray):
		return self.factor * myFFT(self.sparseMStar[ray].dot(z))

	def totalPower(self, z):
		return (z[0] * z[0].conj() + 2 * np.sum(z[1:] * z[1:].conj())).real / len(z)

	def softThreshold(self, x, lambdaValue):
		return np.fmax(0,1.0 - lambdaValue / np.fmax(np.abs(x),1e-10)) * x

	def hardThreshold(self, x, lambdaValue):
		xPar = np.copy(x)		
		xPar[np.abs(x) < lambdaValue] = 0.0
		return xPar

	def FISTA(self, initial=None, initialStokes=None, thresholdMethod='soft', niter=10, lambdaValue=1.0):
		"""
		Solve the l1 regularized problem using the FISTA algorithm, that solves the following problem:

		argmin_O  ||y - M*F^{-1}*alpha||_2 + \lambda ||alpha||_1
			
		Args:
		    rank (int, optional): rank of the solution
		    niter (int, optional): number of iterations
		
		Returns:
		    TYPE: Description
		"""
		if (initial == None):
			x = np.zeros(self.nSteps)
			I0 = 0.9
			Q0 = 0.1
			U0 = 0.2
			V0 = 0.3
			betaI = 10.0#self.beta[0]
			betaQ = 10.0#self.beta[1]
			betaU = 10.0#self.beta[2]
			betaV = 10.0#self.beta[3]

		else:
			x = np.copy(initial)
			I0, Q0, U0, V0 = initialStokes
		
		xNew = np.copy(x)
		y = np.copy(x)
				
		res = self.sparseMStar[0].dot(self.sparseM[0])
		largestEigenvalue = splinalg.eigsh(res, k=1, which='LM', return_eigenvectors=False)
		self.mu = 0.5 / (np.real(largestEigenvalue)) * 0.0002

		t = 1.0	
		
		normL1 = []
		normL2 = []
		normL0 = []

		for loop in range(niter):

			signal = myIFFT(x)
			forwI = self.forward(signal, betaI, 0)   # M1(t) * (1+betaI*N(t))
			forwQ = self.forward(signal, betaQ, 1)   # M2(t) * (1+betaQ*N(t))
			forwU = self.forward(signal, betaU, 2)   # M3(t) * (1+betaU*N(t))
			forwV = self.forward(signal, betaV, 3)   # M4(t) * (1+betaV*N(t))

			residual1 = self.signalIntegrated[0] - (I0 * forwI + Q0 * forwQ + U0 * forwU + V0 * forwV)
			gradient1 = -2.0 * I0 * betaI * self.backward(residual1, 0) - 2.0 * Q0 * betaQ * self.backward(residual1, 1) - \
				2.0 * U0 * betaU * self.backward(residual1, 2) - 2.0 * V0 * betaV * self.backward(residual1, 3)

			residual2 = self.signalIntegrated[1] - (I0 * forwI - Q0 * forwQ - U0 * forwU - V0 * forwV)
			gradient2 = -2.0 * I0 * betaI * self.backward(residual2, 0) + 2.0 * Q0 * betaQ * self.backward(residual2, 1) + \
				2.0 * U0 * betaU * self.backward(residual2, 2) + 2.0 * V0 * betaV * self.backward(residual2, 3)
		
			gradient = gradient1 + gradient2
			
			if (thresholdMethod == 'hardLambda'):
				xNew = self.hardThreshold(y - self.mu * np.real(gradient), lambdaValue)

			if (thresholdMethod == 'hardPercentage'):
				xNew = self.hardThreshold(y - self.mu * np.real(gradient), lambdaValue)

			if (thresholdMethod == 'soft'):				
				xNew = self.softThreshold(y - self.mu * np.real(gradient), lambdaValue)
				xNew[0] = 0.0
				xNew /= np.sqrt(myTotalPower(xNew))
								

			if (thresholdMethod == 'L2'):
				xNew = y - self.mu * np.real(gradient)
			
			tNew = 0.5*(1+np.sqrt(1+4.0*t**2))

			y = xNew + (t-1.0) / tNew * (xNew - x)

			t = tNew

			x = np.copy(xNew)
			
			normResidual = np.linalg.norm(residual1 + residual2)
			normSolutionL1 = np.linalg.norm(x, 1)
			normSolutionL0 = np.linalg.norm(x, 0)
			
			if (loop % 10):
				
# Stokes parameters
				I0 = 0.5 * np.sum(forwI * (self.signalIntegrated[0]+self.signalIntegrated[1])) / np.sum(forwI**2)

				A = np.zeros((3,3))
				A[0,0] = np.sum(forwQ**2)
				A[1,1] = np.sum(forwU**2)
				A[2,2] = np.sum(forwV**2)
				A[0,1] = np.sum(forwQ * forwU)
				A[1,0] = A[0,1]
				A[0,2] = np.sum(forwQ * forwV)
				A[2,0] = A[0,2]
				A[1,2] = np.sum(forwU * forwV)
				A[2,1] = A[1,2]
				b = np.zeros(3)
				b[0] = 0.5 * np.sum(forwQ * (self.signalIntegrated[0]-self.signalIntegrated[1]))
				b[1] = 0.5 * np.sum(forwU * (self.signalIntegrated[0]-self.signalIntegrated[1]))
				b[2] = 0.5 * np.sum(forwV * (self.signalIntegrated[0]-self.signalIntegrated[1]))
				Q0, U0, V0 = np.linalg.solve(A,b)

				if (I0 < 0):
					I0 = 1.0
				if (np.abs(Q0) > 1.0):
					Q0 = 1e-3
				if (np.abs(U0) > 1.0):
					U0 = 1e-3
				if (np.abs(V0) > 1.0):
					V0 = 1e-3
			
# Seeing amplitude			
				M1N = self.forwardPartial(signal, 0)     # M1(t) * N(t)
				M2N = self.forwardPartial(signal, 1)     # M2(t) * N(t)
				M3N = self.forwardPartial(signal, 2)     # M3(t) * N(t)
				M4N = self.forwardPartial(signal, 3)     # M4(t) * N(t)
				
				M1One = self.forwardPartial(np.ones(self.nSteps), 0)     # M1(t) * 1(t)
				M2One = self.forwardPartial(np.ones(self.nSteps), 1)     # M2(t) * 1(t)
				M3One = self.forwardPartial(np.ones(self.nSteps), 2)     # M3(t) * 1(t)
				M4One = self.forwardPartial(np.ones(self.nSteps), 3)		# M4(t) * 1(t)

				A = np.zeros((3,3))
				A[0,0] = Q0**2 * np.sum(M2N**2)
				A[1,1] = U0**2 * np.sum(M3N**2)
				A[2,2] = V0**2 * np.sum(M4N**2)
				
				A[0,1] = Q0 * U0 * np.sum(M3N * M2N)
				A[1,0] = A[0,1]
				A[0,2] = Q0 * V0 * np.sum(M4N * M2N)
				A[2,0] = A[0,2]
				A[1,2] = U0 * V0 * np.sum(M4N * M3N)
				A[2,1] = A[1,2]

				b = np.zeros(3)

				b[0] = 0.5 * Q0 * np.sum(M2N * (self.signalIntegrated[0]-self.signalIntegrated[1])) - \
					Q0**2 * np.sum(M2One * M2N) - Q0 * U0 * np.sum(M3One * M2N) - Q0 * V0 * np.sum(M4One * M2N)
				b[1] = 0.5 * U0 * np.sum(M3N * (self.signalIntegrated[0]-self.signalIntegrated[1])) - \
					U0 * Q0 * np.sum(M2One * M3N) - U0**2 * np.sum(M3One * M3N) - U0 * V0 * np.sum(M4One * M3N)
				b[2] = 0.5 * V0 * np.sum(M4N * (self.signalIntegrated[0]-self.signalIntegrated[1])) - \
					V0 * Q0 * np.sum(M2One * M4N) - V0 * U0 * np.sum(M3One * M4N) - V0**2 * np.sum(M4One * M4N)
				
				betaI = np.abs((0.5 * I0 * np.sum(M1N * (self.signalIntegrated[0]+self.signalIntegrated[1])) - \
					I0**2 * np.sum(M1One * M1N)) / (I0**2 * np.sum(M1N**2)))

				betaQ, betaU, betaV = np.abs(np.linalg.solve(A,b))


			if (loop % 50 == 0):
				print "It {0:4d} - l2={1:10.3e} - l1={2:10.4f} - l0={3:5.1f}% - I={4:11.5f} - Q/I={5:11.5f} - U/I={6:11.5f} - V/I={7:11.5f} - bI={8:11.5f} - bQ={9:11.5f} - bU={10:11.5f} - bV={11:11.5f}".format(loop, normResidual, 
					normSolutionL1, 100.0*normSolutionL0 / self.nSteps, I0, Q0/I0, U0/I0, V0/I0, betaI, betaQ, betaU, betaV)

			normL2.append(normResidual)
			normL1.append(normSolutionL1)
			normL0.append(normSolutionL0)
			
		return x, (I0, Q0, U0, V0), (betaI, betaQ, betaU, betaV), normL2, normL1, normL0

	def demodulateTrivial(self):
		forwI = self.sparseM[0].dot(np.zeros(self.nSteps)+1.0)
		forwQ = self.sparseM[1].dot(np.zeros(self.nSteps)+1.0)
		forwU = self.sparseM[2].dot(np.zeros(self.nSteps)+1.0)
		forwV = self.sparseM[3].dot(np.zeros(self.nSteps)+1.0)
		
		I0 = 0.5 * np.sum(forwI * (self.signalIntegrated[0]+self.signalIntegrated[1])) / np.sum(forwI**2)

		A = np.zeros((3,3))
		A[0,0] = np.sum(forwQ**2)
		A[1,1] = np.sum(forwU**2)
		A[2,2] = np.sum(forwV**2)
		A[0,1] = np.sum(forwQ * forwU)
		A[1,0] = A[0,1]
		A[0,2] = np.sum(forwQ * forwV)
		A[2,0] = A[0,2]
		A[1,2] = np.sum(forwU * forwV)
		A[2,1] = A[1,2]
		b = np.zeros(3)
		b[0] = 0.5 * np.sum(forwQ * (self.signalIntegrated[0]-self.signalIntegrated[1]))
		b[1] = 0.5 * np.sum(forwU * (self.signalIntegrated[0]-self.signalIntegrated[1]))
		b[2] = 0.5 * np.sum(forwV * (self.signalIntegrated[0]-self.signalIntegrated[1]))
		Q0, U0, V0 = np.linalg.solve(A,b)

		return I0, Q0, U0, V0

# totalTime = 1.0      # s
# dt = 0.001           # s
# dtIntegration = 0.01 #s

# beta = np.asarray([15.0, 100.0, 100., 100.0])
# stokes = np.asarray([1.0, 1.2e-3, 5.e-3, 0.001])

# out = randomDemodulator(totalTime, dt, dtIntegration, stokes, beta, seed=123, signalToNoise=1e3)
# coefFourier, stokes, beta, normL21, normL11 = out.FISTA(thresholdMethod = 'soft', niter = 600, lambdaValue = 0.000000051)

# stI, stQ, stU, stV = out.demodulateTrivial()

# print "Q/I_original={0} - Q/I_inferred={1} - Q/I_trivial={2} - diff={3}".format(out.stokes[1] / out.stokes[0], stokes[1] / stokes[0], \
# 	stQ/stI, out.stokes[1] / out.stokes[0]-stokes[1] / stokes[0])
# print "U/I_original={0} - U/I_inferred={1} - U/I_trivial={2} - diff={3}".format(out.stokes[2] / out.stokes[0], stokes[2] / stokes[0], \
# 	stU/stI, out.stokes[2] / out.stokes[0]-stokes[2] / stokes[0])
# print "V/I_original={0} - V/I_inferred={1} - V/I_trivial={2} - diff={3}".format(out.stokes[3] / out.stokes[0], stokes[3] / stokes[0], \
# 	stV/stI, out.stokes[3] / out.stokes[0]-stokes[3] / stokes[0])

# pl.close('all')
# f, ax = pl.subplots(nrows=1, ncols=4, figsize=(18,6))
# coefFourier[0] = 0.0
# Nt = myIFFT(coefFourier)
# Nt /= np.sqrt(myTotalPower(coefFourier))

# stokesPar = ['I', 'Q', 'U', 'V']
# loop = 0
# for loop in range(4):	
# 	ax[loop].plot(out.times, out.signal[loop])
# 	ax[loop].plot(out.times, stokes[loop] / stokes[0] * (1.0 + beta[loop]*Nt))
# 	ax[loop].set_xlabel('Time [s]')
# 	ax[loop].set_ylabel('Stokes {0}'.format(stokesPar[loop]))
# 	ax[loop].annotate

# pl.tight_layout()


# ax[0,0].plot(out.times, out.signal[0])
# ax[0,0].plot(out.times, stokes[0] *(1.0+beta[0]*Nt))

# ax[0,1].plot(out.signal[1])
# ax[0,1].plot(stokes[1] / stokes[0] *(1.0+beta[1]*Nt))

# ax[1,0].plot(out.signal[2])
# ax[1,0].plot(stokes[2] / stokes[0] * (1.0+beta[2]*Nt))

# ax[1,1].plot(out.signal[3])
# ax[1,1].plot(stokes[3] / stokes[0] * (1.0+beta[3]*Nt))

# ax[2,0].semilogy(np.abs(myFFT(out.seeing)))
# ax[2,0].semilogy(np.abs(myFFT(Nt)))

# ax[2,1].semilogy(normL21)
# ax[2,1].semilogy(normL11)

# ax[3,0].plot(out.signalIntegrated[0])
# ax[3,0].plot(out.signalIntegrated[1])

# ax[3,1].plot(out.seeing)
# ax[3,1].plot(Nt)

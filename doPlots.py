import numpy as np
import matplotlib.pyplot as pl
import randomFourierComplete as rn
from ipdb import set_trace as stop
import seaborn as sn

def nPhotons(deltaL, D, px, Temp, E, t):	
	k = 1.3806503e-16
	h = 6.62606876e-27
	c = 2.99792458e10
	wave = 6302.0e-8

	Ephot = h * c / wave
	area = np.pi * (D / 2.0)**2
	solidAngle = 2.0 * np.pi * (1.0 - np.cos(px / 206265.0 / 2))
	
	B = 2.0 * h * c**2 / wave**5 / (np.exp(h*c/(wave*k*Temp))-1.0)  # erg s^-1 cm^-2 * A^-1 * sr^-1

	return E * B * deltaL * area * solidAngle * t / Ephot



def exampleNoNoise():
	"""Plot information for a case without noise
	
	Returns:
	    TYPE: None
	"""
	totalTime = 1.0      # s
	dt = 0.001           # s
	dtIntegration = 0.01 #s

	betaIn = np.asarray([15.0, 100.0, 100., 100.0])
	stokesIn = np.asarray([1.0, 1.2e-3, 5.e-3, 0.001])

	lambdas = [5e-3, 5e-5, 5e-7]

	pl.close('all')
	f, ax = pl.subplots(nrows=3, ncols=3, figsize=(15,12), sharex='col')

	stokesPar = ['I', 'Q', 'U', 'V']

	for i in range(3):

		out = rn.randomDemodulator(totalTime, dt, dtIntegration, stokesIn, betaIn, seed=123, signalToNoise=1e20)
		coefFourier, stokes, beta, normL2, normL1, normL0 = out.FISTA(thresholdMethod = 'soft', niter = 1000, lambdaValue = lambdas[i])

		stI, stQ, stU, stV = out.demodulateTrivial()

		print "Q/I_original={0} - Q/I_inferred={1} - Q/I_trivial={2} - diff={3}".format(out.stokes[1] / out.stokes[0], stokes[1] / stokes[0], \
			stQ/stI, out.stokes[1] / out.stokes[0]-stokes[1] / stokes[0])
		print "U/I_original={0} - U/I_inferred={1} - U/I_trivial={2} - diff={3}".format(out.stokes[2] / out.stokes[0], stokes[2] / stokes[0], \
			stU/stI, out.stokes[2] / out.stokes[0]-stokes[2] / stokes[0])
		print "V/I_original={0} - V/I_inferred={1} - V/I_trivial={2} - diff={3}".format(out.stokes[3] / out.stokes[0], stokes[3] / stokes[0], \
			stV/stI, out.stokes[3] / out.stokes[0]-stokes[3] / stokes[0])
		
		coefFourier[0] = 0.0
		Nt = rn.myIFFT(coefFourier)
		Nt /= np.sqrt(rn.myTotalPower(coefFourier))
		
		ax[i,0].plot(out.times, out.seeing, label='Original')
		ax[i,0].plot(out.times, Nt, label='Reconstructed')
		if (i == 2):
			ax[i,0].set_xlabel('Time [s]')
		ax[i,0].set_ylabel('Seeing random process')
		
		ax[i,0].text(0.8, 0.07, '$\lambda={0}$'.format(lambdas[i]))
		if (i == 0):
			ax[i,0].legend(loc='upper left', fontsize=15)		

		ax[i,1].semilogy(out.freq, rn.myFFT(out.seeing) * np.conj(rn.myFFT(out.seeing)), '.', label='Original')
		ax[i,1].semilogy(out.freq, rn.myFFT(Nt) * np.conj(rn.myFFT(Nt)), '.', label='Reconstructed')
		ax[i,1].set_ylim([1e-9,1])
		if (i == 2):
			ax[i,1].set_xlabel('Frequency [Hz]')
		ax[i,1].set_ylabel('Power spectrum')
		if (i == 0):
			ax[i,1].legend(fontsize=15)

		for j in range(4):
			ax[i,1].text(0.1,0.92-j*0.05,'{0}$_0$={1:10.7f}'.format(stokesPar[j],stokes[j] / stokes[0]), transform=ax[i,1].transAxes, fontsize=12)
			ax[i,1].text(0.1,0.92-(j+4)*0.05,r'$\beta_{0}$={1:5.2f}'.format(stokesPar[j],beta[j]), transform=ax[i,1].transAxes, fontsize=12)


		ax[i,2].loglog(normL2, label=r'$\ell_2$')
		ax[i,2].loglog(normL1, label=r'$\ell_1$')
		ax[i,2].loglog(normL0, label=r'$\ell_0$')
		if (i == 2):
			ax[i,2].set_xlabel('Iteration')
		ax[i,2].set_ylabel('Error norm')
		ax[i,2].set_xlim([1,1000])
		if (i == 0):
			ax[i,2].legend(fontsize=15)

	# ax[loop].set_xlabel('Time [s]')
	# ax[loop].set_ylabel('Stokes para{0}'.format(stokesPar[loop]))
	# 	ax[loop].text(0.05,0.9,'{0}$_0$={1:10.7f}'.format(stokesPar[loop],stokesIn[loop] / stokesIn[0]), transform=ax[loop].transAxes, fontsize=15)


		# ax[2,0].semilogy(np.abs(myFFT(out.seeing)))
# ax[2,0].semilogy(np.abs(myFFT(Nt)))

# ax[2,1].semilogy(normL21)
# ax[2,1].semilogy(normL11)

	pl.tight_layout()
	pl.savefig('../noNoise.pdf')


def exampleNoise():
	"""Plot information for a case with noise
	
	Returns:
	    TYPE: Description
	"""
	totalTime = 1.0      # s
	dt = 0.001           # s
	dtIntegration = 0.01 #s

	beta = np.asarray([15.0, 100.0, 100., 100.0])
	stokes = np.asarray([1.0, 1.2e-3, 5.e-3, 0.001])

	out = rn.randomDemodulator(totalTime, dt, dtIntegration, stokes, beta, seed=123, signalToNoise=3e3)
	coefFourier, stokes, beta, normL2, normL1, normL0 = out.FISTA(thresholdMethod = 'soft', niter = 1000, lambdaValue = 5e-6)

	stI, stQ, stU, stV = out.demodulateTrivial()

	print "Q/I_original={0} - Q/I_inferred={1} - Q/I_trivial={2} - diff={3}".format(out.stokes[1] / out.stokes[0], stokes[1] / stokes[0], \
		stQ/stI, out.stokes[1] / out.stokes[0]-stokes[1] / stokes[0])
	print "U/I_original={0} - U/I_inferred={1} - U/I_trivial={2} - diff={3}".format(out.stokes[2] / out.stokes[0], stokes[2] / stokes[0], \
		stU/stI, out.stokes[2] / out.stokes[0]-stokes[2] / stokes[0])
	print "V/I_original={0} - V/I_inferred={1} - V/I_trivial={2} - diff={3}".format(out.stokes[3] / out.stokes[0], stokes[3] / stokes[0], \
		stV/stI, out.stokes[3] / out.stokes[0]-stokes[3] / stokes[0])

	pl.close('all')
	f, ax = pl.subplots(nrows=1, ncols=3, figsize=(18,6))
	coefFourier[0] = 0.0
	Nt = rn.myIFFT(coefFourier)
	Nt /= np.sqrt(rn.myTotalPower(coefFourier))

	stokesPar = ['I', 'Q', 'U', 'V']
	loop = 0
	ax[0].plot(out.times, out.seeing, label='Original')
	ax[0].plot(out.times, Nt, label='Reconstructed')
	ax[0].set_xlabel('Time [s]')
	ax[0].set_ylabel('Seeing random process')
	
	ax[0].legend(loc='upper left', fontsize=15)		

	ax[1].semilogy(out.freq, rn.myFFT(out.seeing) * np.conj(rn.myFFT(out.seeing)), '.', label='Original')
	ax[1].semilogy(out.freq, rn.myFFT(Nt) * np.conj(rn.myFFT(Nt)), '.', label='Reconstructed')
	ax[1].set_ylim([1e-9,1])
	ax[1].set_xlabel('Frequency [Hz]')
	ax[1].set_ylabel('Power spectrum')
	ax[1].legend(fontsize=15)

	for j in range(4):
		ax[1].text(0.1,0.92-j*0.05,'{0}$_0$={1:10.7f}'.format(stokesPar[j],stokes[j] / stokes[0]), transform=ax[1].transAxes, fontsize=12)
		ax[1].text(0.1,0.92-(j+4)*0.05,r'$\beta_{0}$={1:5.2f}'.format(stokesPar[j],beta[j]), transform=ax[1].transAxes, fontsize=12)


	ax[2].loglog(normL2, label=r'$\ell_2$')
	ax[2].loglog(normL1, label=r'$\ell_1$')
	ax[2].loglog(normL0, label=r'$\ell_0$')
	ax[2].set_xlabel('Iteration')
	ax[2].set_ylabel('Error norm')
	ax[2].set_xlim([1,1000])
	ax[2].legend(fontsize=15)			

	pl.tight_layout()

def exampleNoiseMany():
	totalTime = 1.0      # s
	dt = 0.001           # s
	dtIntegration = 0.01 #s

	betaOrig = np.asarray([15.0, 100.0, 100., 100.0])
	stokesOrig = np.asarray([1.0, 1.2e-3, 5.e-3, 0.001])
	snRatio = np.asarray([1e2,3e2,1e3,5e3,1e4])

	inferredStokes = np.zeros((100,5,4))

	for j in range(5):
		for i in range(100):
			out = rn.randomDemodulator(totalTime, dt, dtIntegration, stokesOrig, betaOrig, signalToNoise=snRatio[j], modulationType=1)
			coefFourier, stokes, beta, normL2, normL1, normL0 = out.FISTA(thresholdMethod = 'soft', niter = 600, lambdaValue = 5e-6)
			inferredStokes[i,j,:] = [stokes[0], stokes[1] / stokes[0], stokes[2] / stokes[0], stokes[3] / stokes[0]]

	np.save('noisyCasesUniform.npy', inferredStokes)

	stop()

def doPlotNoisy():
	res = np.load('noisyCases.npy')
	snRatio = np.asarray([1e2,3e2,1e3,5e3,1e4])
	stokesOrig = np.asarray([1.0, 1.2e-3, 5.e-3, 0.001])

	mn = np.percentile(res - stokesOrig[None,None,:], [50.0-68.0/2,50.0,50+68.0/2], axis=0)

	stokesPar = ['I', 'Q', 'U', 'V']

	cmap = sn.color_palette()

	pl.close('all')
	f, ax = pl.subplots(ncols=2, nrows=2, figsize=(15,8), sharex='col')
	ax = ax.flatten()
	for i in range(4):
		ax[i].fill_between(snRatio, mn[0][:,i], mn[2][:,i])
		ax[i].semilogx(snRatio, mn[1][:,i], color=cmap[1])
		# ax[i].axhline()
		ax[i].semilogx(snRatio, + 1.0 / snRatio, '--', color=cmap[1])
		ax[i].semilogx(snRatio, - 1.0 / snRatio, '--', color=cmap[1])
		ax[i].semilogx(snRatio, + 1.0 / (np.sqrt(100.0)*snRatio), '--', color=cmap[2])
		ax[i].semilogx(snRatio, - 1.0 / (np.sqrt(100.0)*snRatio), '--', color=cmap[2])
		if (i >= 2):
			ax[i].set_xlabel('S/N')
		ax[i].set_ylabel(r'$\Delta${0}$_0$'.format(stokesPar[i]))

		print np.abs(mn[0][:,i] - mn[2][:,i]) / (2.0 / (np.sqrt(100.0)*snRatio))

	pl.tight_layout()
	pl.savefig('../noisySN.pdf')

	f, ax = pl.subplots(ncols=1, nrows=1, figsize=(8,6))
	for i in range(4):
		ax.semilogx(snRatio, np.abs(mn[0][:,i] - mn[2][:,i]) / (2.0 / (np.sqrt(100.0)*snRatio)), label='Stokes {0}'.format(stokesPar[i]))
	ax.set_xlabel('S/N')
	ax.set_ylabel(r'$\sigma_\mathrm{inferred}/\sigma_\mathrm{potential}$')
	pl.legend(loc='upper left')
	pl.tight_layout()

	pl.savefig('../ratioSN.pdf')

def doPlotNoisyUniform():
	res = np.load('noisyCasesUniform.npy')
	snRatio = np.asarray([1e2,3e2,1e3,5e3,1e4])
	stokesOrig = np.asarray([1.0, 1.2e-3, 5.e-3, 0.001])

	mn = np.percentile(res - stokesOrig[None,None,:], [50.0-68.0/2,50.0,50+68.0/2], axis=0)

	stokesPar = ['I', 'Q', 'U', 'V']

	cmap = sn.color_palette()

	factors = [1.0, np.sqrt(3.0), np.sqrt(3.0), np.sqrt(3.0)]

	pl.close('all')
	f, ax = pl.subplots(ncols=2, nrows=2, figsize=(15,8), sharex='col')
	ax = ax.flatten()
	for i in range(4):
		ax[i].fill_between(snRatio, mn[0][:,i], mn[2][:,i])
		ax[i].semilogx(snRatio, mn[1][:,i], color=cmap[1])
		# ax[i].axhline()
		# ax[i].semilogx(snRatio, + 1.0 / snRatio, '--', color=cmap[1])
		# ax[i].semilogx(snRatio, - 1.0 / snRatio, '--', color=cmap[1])
		ax[i].semilogx(snRatio, + factors[i] / (np.sqrt(100.0)*snRatio), '--', color=cmap[2])
		ax[i].semilogx(snRatio, - factors[i] / (np.sqrt(100.0)*snRatio), '--', color=cmap[2])
		if (i >= 2):
			ax[i].set_xlabel('S/N')
		ax[i].set_ylabel(r'$\Delta${0}$_0$'.format(stokesPar[i]))

		print np.abs(mn[0][:,i] - mn[2][:,i]) / (2.0 / (np.sqrt(100.0)*snRatio))

	pl.tight_layout()
	pl.savefig('../noisySNUniform.pdf')

	f, ax = pl.subplots(ncols=1, nrows=1, figsize=(8,6))
	for i in range(4):
		ax.semilogx(snRatio, np.abs(mn[0][:,i] - mn[2][:,i]) / (2.0 / (np.sqrt(100.0)*snRatio)), label='Stokes {0}'.format(stokesPar[i]))
	ax.set_xlabel('S/N')
	ax.set_ylabel(r'$\sigma_\mathrm{inferred}/\sigma_\mathrm{potential}$')
	pl.legend(loc='upper left')
	pl.tight_layout()

	pl.savefig('../ratioSNUniform.pdf')


print "Expected S/N={0}".format(np.sqrt(nPhotons(22e-3 * 1e-8, 400., 0.0162, 6000.0, 0.2, 0.01)))

sn.set_context("paper", font_scale=1.5)

# No noise
# exampleNoNoise()

# Noise
exampleNoiseMany()
doPlotNoisyUniform()
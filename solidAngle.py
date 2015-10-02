import numpy as np
import matplotlib.pyplot as pl
import emcee
import seaborn as sn

class variableAngle(object):
	"""Class that samples from a polarimeter with FLC"""
	def __init__(self, logPosteriorFun=0):

		pass
			
	def logPosterior(self, theta):
		"""Distribution giving the solid angle for a polarimeter made of two FLC
		dOmega = cos(2*(alpha-2*beta))
		
		Args:
		    theta (TYPE): value of the angles
		
		Returns:
		    TYPE: probability distribution for the solid angle
		"""
		if (theta[0] > 0.0 and theta[0] < np.pi/2.0 and theta[1] > 0.0 and theta[1] < np.pi/2.0):       
			return np.log(np.abs(np.cos(2.0*(theta[0]-2.0*theta[1]))))
		return -np.inf
		
	def sample(self):        
		"""Do the sample using emcee
		
		Returns:
		    TYPE: None
		"""
		ndim, nwalkers = 2, 200                          
		self.theta0 = np.asarray([ 0.2, 0.2 ]) 
		p0 = [self.theta0 + 0.01*np.random.randn(ndim) for i in range(nwalkers)]
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logPosterior)
		self.sampler.run_mcmc(p0, 1000)

class variableRetardance(object):
	"""Class that samples from a polarimeter with FLC"""
	def __init__(self, logPosteriorFun=0):

		pass
			
	def logPosterior(self, theta):      
		"""Distribution giving the solid angle for a polarimeter made of two LCVR
		dOmega = sin(beta)
		
		Args:
		    theta (TYPE): value of the angles
		
		Returns:
		    TYPE: probability distribution for the solid angle
		"""
		if (theta[0] > 0.0 and theta[0] < 2.0*np.pi and theta[1] > 0.0 and theta[1] < 2.0*np.pi):       
			return np.log(np.abs(np.sin(theta[1])))
		return -np.inf
		
	def sample(self):        
		"""Do the sampling using emcee
		
		Returns:
		    TYPE: None
		"""
		ndim, nwalkers = 2, 200                          
		self.theta0 = np.asarray([ 0.2, 0.2 ]) 
		p0 = [self.theta0 + 0.01*np.random.randn(ndim) for i in range(nwalkers)]
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logPosterior)
		self.sampler.run_mcmc(p0, 1000)

# First a polarimeter with a lambda/2 and lambda/4 with variable fast axis
out = variableAngle()
out.sample()
alphaModulation = out.sampler.chain[:,-100:,0].flatten()
betaModulation = out.sampler.chain[:,-100:,1].flatten()
nSteps = len(alphaModulation)
M = np.zeros((3,nSteps))

np.savez('alphaBetaSamplesVariableAngle.npz', alphaModulation, betaModulation)

M[0,:] = np.cos(2.0*alphaModulation) * np.cos(2.0*(alphaModulation-2.0*betaModulation))
M[1,:] = np.sin(2.0*alphaModulation) * np.cos(2.0*(alphaModulation-2.0*betaModulation))
M[2,:] = np.sin(2.0*(2.0*betaModulation-alphaModulation))

pl.close('all')
f, ax = pl.subplots(nrows=1, ncols=3, figsize=(15,4))
for i in range(3):
	sn.distplot(M[i,:], bins=100, kde=False, ax=ax[i])
	ax[i].set_xlabel('M$_{0}$'.format(i+2))
	ax[i].set_ylabel('Frequency')
ax[1].set_title('FLC')

pl.tight_layout()

pl.savefig('../modulationUniformVariableAngle.pdf')

pl.close('all')
f, ax = pl.subplots(nrows=2, ncols=3, figsize=(15,8))
for i in range(3):
	sn.distplot(M[i,:], bins=100, kde=False, ax=ax[0,i])
	ax[0,i].set_xlabel('M$_{0}$'.format(i+2))
	ax[0,i].set_ylabel('Frequency')

ax[1,0].plot(M[0,0:5000], M[1,0:5000], '.', alpha=0.1)
ax[1,0].set_xlabel('M$_2$')
ax[1,0].set_ylabel('M$_3$')
ax[1,1].plot(M[0,0:5000], M[2,0:5000], '.', alpha=0.1)
ax[1,1].set_xlabel('M$_2$')
ax[1,1].set_ylabel('M$_4$')
ax[1,2].plot(M[1,0:5000], M[2,0:5000], '.', alpha=0.1)
ax[1,2].set_xlabel('M$_3$')
ax[1,2].set_ylabel('M$_4$')

pl.tight_layout()

pl.savefig('modulationUniformVariableAngleFull.pdf')


# Second a polarimeter with modulators at 0 and 45 with variable retardances
out = variableRetardance()
out.sample()
alphaModulation = out.sampler.chain[:,-100:,0].flatten()
betaModulation = out.sampler.chain[:,-100:,1].flatten()
nSteps = len(alphaModulation)
M = np.zeros((3,nSteps))

np.savez('alphaBetaSamplesVariableRetardance.npz', alphaModulation, betaModulation)

M[0,:] = np.cos(betaModulation)
M[1,:] = np.sin(alphaModulation) * np.sin(betaModulation)
M[2,:] = -np.cos(alphaModulation) * np.sin(betaModulation)


pl.close('all')
f, ax = pl.subplots(nrows=1, ncols=3, figsize=(15,4))
for i in range(3):
	sn.distplot(M[i,:], bins=100, kde=False, ax=ax[i])
	ax[i].set_xlabel('M$_{0}$'.format(i+2))
	ax[i].set_ylabel('Frequency')
ax[1].set_title('LCVR')

pl.tight_layout()

pl.savefig('../modulationUniformVariableRetardance.pdf')

pl.close('all')
f, ax = pl.subplots(nrows=2, ncols=3, figsize=(15,8))
for i in range(3):
	sn.distplot(M[i,:], bins=100, kde=False, ax=ax[0,i])
	ax[0,i].set_xlabel('M$_{0}$'.format(i+2))
	ax[0,i].set_ylabel('Frequency')

ax[1,0].plot(M[0,0:5000], M[1,0:5000], '.', alpha=0.1)
ax[1,0].set_xlabel('M$_2$')
ax[1,0].set_ylabel('M$_3$')
ax[1,1].plot(M[0,0:5000], M[2,0:5000], '.', alpha=0.1)
ax[1,1].set_xlabel('M$_2$')
ax[1,1].set_ylabel('M$_4$')
ax[1,2].plot(M[1,0:5000], M[2,0:5000], '.', alpha=0.1)
ax[1,2].set_xlabel('M$_3$')
ax[1,2].set_ylabel('M$_4$')

pl.tight_layout()

pl.savefig('modulationUniformVariableRetardanceFull.pdf')
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class InvalidDimensionsException(Exception):
	pass

def plot_BVG(mean, covariance, plot_start = [-3, -3], plot_stop = [3, 3],
             num = [100, 100]):
	"""
	Plots the distribution of a bivariate gaussian specified
	by the parameters mean and covariance.

	Arguments:
        mean: 1x2 mean vector
        covariance: 2x2 covariance matrix
        plot_start: vector with starting points of plot on w1 and w2 axes.
        num: vector of the number of evenly distributed samples to be plotted
            along w1 and w2 axes respectively.
	"""
	if(mean.shape[0] != 2 or covariance.shape[1] != 2 or covariance.shape[0] != 2):
		raise InvalidDimensionsException("The dimensions of the parameters are" \
                                         " invalid. The mean value vector should" \
                                         " have 2 dimensions and the covariance" \
                                         " should be a 2x2 matrix.")

	# Generate evenly spaces samples over the two dimensions w1 and w2
	w1 = np.linspace(plot_start[0], plot_stop[0], num[0])
	w2 = np.linspace(plot_start[1], plot_stop[1], num[1])
	# Generate meshgrid coordinates using w1 and w2
	W1, W2 = np.meshgrid(w1, w2)
	# Put W1 and W2 into a single three-dimensional array
	pos = np.empty(W1.shape + (2,))
	pos[:, :, 0] = W1
	pos[:, :, 1] = W2
	# Create the bivariate probability distriX.shape
	P = multivariate_normal(mean, covariance)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	fig.subplots_adjust(top=0.85)
	ax.set_title('P(W|X,T)')

	ax.set_xlabel('W1')
	ax.set_ylabel('W2')
	plt.contourf(W1, W2, P.pdf(pos), offset = 0.01) #cset = ax.contourf(W1, W2, P.pdf(pos), zdir='p(w)', offset=-0.01, cmap=cm.viridis)
	plt.show()


def estimatePosterior(prior_mu, prior_cov, error_mean, error_variance, X, t, N):
	"""
	Estimates the posterior distribution of the parameters given t, X and the
	prior over the parameters w. Because the outputs are considered i.i.d, their
	covariance matrix is spherical and depends on the variance of the error.

	Returns the mean and the covariance of the posterior

	Arguments:
		prior_mu: mean of the prior
		prior_cov: covariance matrix of the prior
		error_mean: mean of the irreducible error
		error_variance: variance of irreducible error
		X: input matrix
		t: output vector
		N: number of inputs/ouputs to be used to estimate the likelihood
	"""
	# Get the subset of inputs/outputs to be used
	X = X[:N]
	t = t[:N]

	X_transpose = np.transpose(X)
	tmp = np.matmul(X_transpose, X)

	# Estimate the covariance of the posterior
	posterior_cov = np.linalg.inv((tmp / error_variance) + np.linalg.inv(prior_cov))

	# Estimate the mean of the posterior
	tmp3 = np.matmul(X_transpose, t)
	numerator = np.matmul(posterior_cov, tmp3)
	posterior_mean = numerator / error_variance


	posterior_mean = posterior_mean.flatten()
	return [posterior_mean, posterior_cov]



def generateOutputData(x, w, error_mean, error_variance):
	"""
	Calculates the result of t(i) = w(0)x(i) + w(1) + error for each x and
	returns a list with every t(i). Error is normally distributed.

	Arguments:
		x: input vector
		w: parameters used to calculate the output
		error_mean: mean of the error term
		error_variance: variance of the error term
	"""
	# Generate error samples for each output term
	err = np.random.normal(error_mean, error_variance, x.shape[0])
	# Return output values
	return [w[0]*x[0] + w[1] + e for (x, e) in zip(x, err)]


# Set irreducible error distribution parameters
error_mean = 0.0
error_variance = 0.02


# Generate 2-dimensional input data where the second dimension is always one
x = np.arange(-2, 2.02, 0.02)
np.random.shuffle(x)
x = np.reshape(x, (-1, 1))
x = np.hstack((x, np.ones(x.shape)))
# Generate output data
w = np.asarray([1.5, -0.8])
t = np.asarray(generateOutputData(x, w, error_mean, error_variance))
t = np.reshape(t, (-1, 1))

# Set parameters for spherical bivariate Gaussian prior
prior_variates = 2
parameter_variance = 1.0
prior_mu = np.zeros(2)
prior_cov = parameter_variance * np.identity(prior_variates)

# Visualize prior
plot_BVG(prior_mu, prior_cov)

# Set the amount of inputs to use in order to estimate the likelihood
N = 25

# Plot the posterior distribution of the parameters given x and t
mean, cov = estimatePosterior(prior_mu, prior_cov, error_mean, error_variance, x, t, N)
# Plot the posterior
plot_BVG(mean, cov)
# Sample parameters from the posterior
w1, w2 = np.random.multivariate_normal(mean, cov, 5).T
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Function samples')

ax.set_xlabel('x')
ax.set_ylabel('y')
# Plot original line
plt.plot(x[:,0], w[0]*x[:,0] + w[1])

for i in range(len(w1)):
	# Plot line based on sampled parameters
	plt.plot(x[:,0], w1[i]*x[:,0] + w2[i], color = "red", linestyle = "-", linewidth = 0.5)

plt.show()

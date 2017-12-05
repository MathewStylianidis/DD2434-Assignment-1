import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import exp, pi
from scipy.spatial.distance import cdist


def squared_exp_kernel(x1, x2, sigma = 1, lengthscale = 0.2):
    """
    Evaluates the value of k(x1,x2) where the kernel k is the squared exponential
    covariance function.

    Arguments:
        x1, x2: The vectors which are the input to the kernel function

        sigma: sigma is a scaling factor for the kernel. It determines the
        variation of function values from their mean. Small value of sigma
        characterize functions that stay close to their mean value while larger
        values allow more variation. If the signal variance is too large then
        the model is sensitive to outliers and can overfit.

        length: the lengthscale parameter of the squared exponential kernel. It
        describes how smooth a function is. For small lengthscale valu the
        function values change quickly while for large values characterize
        functions that change only slowly (smoother functions).

    Returns: the value of k(x1,x2)
    """
    numerator = np.transpose((x1 - x2)) * (x1 - x2)
    fraction = numerator / (lengthscale**2)
    kernel_value = (sigma**2) * exp(-fraction)
    return kernel_value


def estimate_sqr_exp_prior(sigma, lengthscale, sample_no = 1000):
    """
    Estimates a gaussian process prior with a squared exponential kernel K.

    Arguments:
        sigma: sigma is a scaling factor for the kernel. It determines the
        variation of function values from their mean. Small value of sigma
        characterize functions that stay close to their mean value while larger
        values allow more variation. If the signal variance is too large then
        the model is sensitive to outliers and can overfit.

        lengthscale: the lengthscale parameter of the squared exponential kernel. It
        describes how smooth a function is. For small lengthscale valu the
        function values change quickly while for large values characterize
        functions that change only slowly (smoother functions).

        sample_no: number of input samples to plot.

    Returns: a tuple with the training input vector, the mean and the covariance
    matrix of the prior.
    """

    # Initialize mean function vector with 0
    mu = np.zeros(sample_no)
    # Take <sample_no> values from the domain [-5, 5] to be used as input samples
    x = np.linspace(-5, 5, sample_no)
    X = np.reshape(x, (-1, 1))

    # Initialize Gram matrix K
    K = np.zeros((sample_no, sample_no))
    # Calculate Gram matrix K
    K = (sigma ** 2) * np.exp(-cdist(X, X, 'sqeuclidean') / (lengthscale ** 2))

    return (x, mu, K)

def estimate_sqr_exp_posterior(sigma, lengthscale, X, y, Z, error_var):
    """
    Estimates a Gaussian process posterior.

    Arguments:
        sigma: sigma is a scaling factor for the kernel. It determines the
        variation of function values from their mean. Small value of sigma
        characterize functions that stay close to their mean value while larger
        values allow more variation. If the signal variance is too large then
        the model is sensitive to outliers and can overfit.

        lengthscale: the lengthscale parameter of the squared exponential kernel. It
        describes how smooth a function is. For small lengthscale valu the
        function values change quickly while for large values characterize
        functions that change only slowly (smoother functions).

        X: matrix with training data input samples

        y: training data output values

        Z: matrix with input values which are unobserved

        error_var: variance of error which is independently distributed given x.

    Returns: A tuple with the mean and the covariance matrix of the posterior.
    """
    tr_set_size = x.shape[0]
    test_set_size = z.shape[0]
    K_X = np.zeros((tr_set_size, tr_set_size))
    K_X_star = np.zeros((test_set_size, test_set_size))
    K_X_X_star = np.zeros((tr_set_size, test_set_size))
    K_X_star_X = np.zeros((test_set_size, tr_set_size))

    # Estimate Gram matrix for training data
    K_X = (sigma ** 2) * np.exp(-cdist(X, X, 'sqeuclidean') / (lengthscale ** 2))
    # Estimate Gram matrix for unseen data
    K_X_star = (sigma ** 2) * np.exp(-cdist(Z, Z, 'sqeuclidean') / (lengthscale ** 2))
    # Estimate Gram matrix between training and unseen data
    K_X_X_star = (sigma ** 2) * np.exp(-cdist(X, Z, 'sqeuclidean') / (lengthscale ** 2))
    K_X_star_X = (sigma ** 2) * np.exp(-cdist(Z, X, 'sqeuclidean') / (lengthscale ** 2))

    # Turn y to a one dimensional matrix to be able to do matrix operations
    Y =  np.reshape(y, (-1, 1))

    # Store result to minimize computations
    tmp = np.matmul(K_X_star_X, np.linalg.inv(K_X + error_var * np.identity(tr_set_size)))

    # Estimate the mean of the posterior
    mean = np.matmul(tmp, Y)
    # Estimate covariance
    cov = K_X_star - np.matmul(tmp, K_X_X_star)

    return (mean, cov)


def plot_gp(x, mu, K, sample_functions = 1, title = ""):
    """
    Plots a gaussian process.

    Arguments:
        x = input range to plot for the x axis
        mu: mean of the gaussian process
        K: Gram matrix of the gaussian process
        sample_functions: number of sampled functions to plot from the prior
        title: title to show on plot
    """
    mu = mu.flatten()
    x = x.flatten()

    #Generate <sample_no> sample functions from the gaussian process
    samples = np.random.multivariate_normal(mu, K, sample_functions)

    for i in range(sample_functions):
        plt.plot(x, samples[i, :])

    # Plot predictive mean
    plt.plot(x, mu, color = "black", linestyle = '--')
    # Get predictive variance as vector
    var = np.diag(K)
    # Plot variance
    plt.fill_between(x, mu - np.sqrt(var), mu + np.sqrt(var), color = 'gray')

    plt.title(title)
    plt.show()



def plot_gp_posterior(x, mu, K, sample_functions = 1, training_x = None, training_y = None
                        , title = ""):
    """
    Plots a gaussian process posterior and also scatters a number of input points
    and their values (typically the ones used for estimating the posterior).

    Arguments:
        x = input range to plot for the x axis
        mu: mean of the gaussian process
        K: Gram matrix of the gaussian process
        sample_functions: number of sampled functions to plot from the prior
        training_x: input values used to train the posterior
        training_y: output values for training_x used to train the posterior
        title: title to show on plot
    """
    mu = mu.flatten()
    x = x.flatten()

    #Generate <sample_no> sample functions from the gaussian process
    samples = np.random.multivariate_normal(mu, K, sample_functions)

    for i in range(sample_functions):
        plt.plot(x, samples[i, :], zorder = 1)

    # Plot predictive mean
    plt.plot(x, mu, color = "black", linestyle = '--', zorder = 2)
    # Get predictive variance as vector
    var = np.diag(K)
    # Plot variance
    plt.fill_between(x, mu - np.sqrt(var), mu + np.sqrt(var), color = 'gray')

    plt.scatter(training_x, training_y, color = 'black', s = 35.0, zorder = 3)

    plt.title(title)
    plt.show()

def noisyCos(x, mean = 0, variance = 0.2):
    """
    Estimates y(i) = cos(x(i)) + epsilon for each x(i) in vector x where
    epsilon is a random variable normally distributed.

    Arguments:
        x: input vector
        mean: mean of normally distributed random variable epsilon
        variance: variance of normally distributed random variable epsilon

    Returns:
        A numpy vector with the output of the noisy cosine function given x.
    """
    # Generate output error samples
    epsilon = np.random.normal(mean, variance, x.shape[0])
    return np.cos(x) + epsilon



############################MAIN PROGRAM#################################

# Plot prior function samples
lengthscale = 1.5
sigma = 1
x, mu, K = estimate_sqr_exp_prior(sigma, lengthscale)
plot_gp(x, mu, K, sample_functions = 5, title = "Lengthscale: " + str(lengthscale))

observation_no = 7
unseen_observation_no = 2000
# Generate input data
x = np.linspace(0, 2 * pi, observation_no)
# Generate unseen data
z = np.linspace(-2 * pi, 4 * pi, unseen_observation_no)
# Set mean and variance for error distribution around y
mean = 0
variance = 0.5
# Generate output data
y = noisyCos(x, mean, variance)

X = np.reshape(x, (-1, 1))
Z = np.reshape(z, (-1, 1))
# Estimate posterior based on X,Y,Z
mu, K = estimate_sqr_exp_posterior(sigma, lengthscale, X, y, Z, variance)
# Plot posterior samples
plot_gp_posterior(Z, mu, K, sample_functions = 10, training_x = x, training_y = y,
                    title = "Lengthscale: " + str(lengthscale))

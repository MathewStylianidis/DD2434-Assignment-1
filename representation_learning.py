import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as opt
from scipy.spatial.distance import sqeuclidean

def f(W, *args):
    """
    Evaluate and return the value of the objective at W

    Arguments:
        W: the variables to be changed in search of a minimum
        args: tuple with parameter values needed to be passed to this function.
    Returns:
        Value of objective at W
    """
    # Get parameters needed to evaluate the objective function from args
    N = args[0]
    D = args[1]
    S = args[2]
    sigma_e = args[3]

    # Reshape W to a 20x20 matrix
    W = np.reshape(W, (10, 2))

    # Evaluate C matrix
    C = np.matmul(W, np.transpose(W)) + ((sigma_e**2) * np.identity(D))
    C_inv = np.linalg.inv(C)

    return (N / 2) * (D * np.log(2 * np.pi) + np.log(np.linalg.det(C)) +
            np.trace(np.matmul(C_inv,S)))

def df(W, *args):
    """
    Evaluate and return the value of the derivative of the objective at W

    Arguments:
        W: the variables to be changed in search of a minimum
        args: tuple with parameter values needed to be passed to this function.
    Returns:
        Value of the derivative of the objective at W
    """
    # Get parameters needed to evaluate the objective function from args
    N = args[0]
    D = args[1]
    S = args[2]
    sigma_e = args[3]

    # Reshape W to a 20x20 matrix
    W = np.reshape(W, (10, 2))

    # Evaluate C matrix
    C = np.matmul(W, np.transpose(W)) + ((sigma_e**2) * np.identity(D))
    C_inv = np.linalg.inv(C)
    df = -N * (np.matmul(C_inv, np.matmul(S, np.matmul(C_inv, W)))
                - np.matmul(C_inv, W))
    return np.reshape(df, (20,))




# Set number of input data
n = 100

# Generate input data
x = np.linspace(0.0, 4 * np.pi, n)
x = np.reshape(x, (-1, 1))
#print(x.shape)

# Generate data with non linear function and x as input
non_lin_x_1 = x * np.cos(x)
non_lin_x_2 = x * np.sin(x)
non_lin_x = np.hstack((non_lin_x_1, non_lin_x_2))
#print(non_lin_x)

# Generate A matrix for the linear transformation
A = np.random.normal(0, 1, (10,2))

# Generate output in final format
Y = np.matmul(non_lin_x, np.transpose(A))

# Plot actual X after the non linear transformation
plt.scatter(non_lin_x_1, non_lin_x_2)
#plt.show()


# Randomly initialize W with 20 values
# This is the flat form of the 10x2 weight matrix
W0 = np.asarray([np.random.normal(0, 100, 1) for i in range(20)])

# Prepare the rest of the parameters needed for gradient descent
N = n # Number of observations
D = 10 # Dimensions of output variable
S = np.cov(Y, rowvar = False)# Sample covariance of observations
#feature_means = np.mean(Y, axis=0)
sigma_e = 1 # Error standard deviation
args = (N, D, S, sigma_e) # Assemble final argument tuple for f and df w.r.t W


# Start the optimization using gradient descent
W = opt.fmin_cg(f, W0, fprime = df, args = args)
W = np.reshape(W, (10, 2))

Wtr = np.transpose(W)
M = np.dot(Wtr, W) + np.identity(2) * sigma_e
Minv = np.linalg.inv(M)
MinvWtr = np.matmul(Minv, Wtr)

#y = np.mean(Y, axis=0)
#new_X = np.matmul(MinvWtr, np.transpose(Y - y)) * 5
new_X = np.matmul(MinvWtr, np.transpose(Y)) * 5
new_X = np.transpose(new_X)

plt.scatter(new_X[:,0], new_X[:,1])
plt.show()

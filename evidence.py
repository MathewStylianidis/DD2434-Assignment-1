import itertools as it
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, pi
from scipy.spatial import distance

#Visualize dataset
def visualize(dataset):
    """
    Take a dataset and visualize it as a grid where blue is the 1 class and red
    is the -1 class.

    Arguments:
        dataset: the dataset to be visualized.
    Returns:
        None
    """
    #Get positive ang negative sample coordinates
    positive = [list(sample[0]) for sample in dataset if sample[1] is 1]
    negative = [list(sample[0]) for sample in dataset if sample[1] is -1]

    # Plot positive points
    plt.scatter([coord[0] for coord in positive], [coord[1] for coord in positive]
                , s = 1600)
    # Plot negative points
    plt.scatter([coord[0] for coord in negative], [coord[1] for coord in negative]
                , s = 1600, marker = 'X')

    # Widen the range on the x and y axes
    axes = plt.gca()
    axes.set_xlim([-2,2])
    axes.set_ylim([-2,2])

    plt.show()

def M_i(i, x = 0, t = 0, theta = None):
    """
    Calls the i_th model and returns its result.

    Arguments:
        i: index of model to be called
        x: input vector to be handed to the model
        t: output label of input vector
        theta: vector with parameter values
    """
    if i == 0:
        return M_0()
    elif i == 1:
        return M_1(x, t, theta)
    elif i == 2:
        return M_2(x, t, theta)
    elif i == 3:
        return M_3(x, t, theta)
    return None

def M_0():
    return (1/512)**(1/9)

def M_1(x, t, theta):
    return 1 / (1 + exp(-t * x[0] * theta[0]))

def M_2(x, t, theta):
    return 1 / (1 + exp(-t * (x[0] * theta[0] + x[1] * theta[1])))

def M_3(x, t, theta):
    return 1 / (1 + exp(-t * (x[0] * theta[0] + x[1] * theta[1] + theta[2])))

def theta_prior(mean_vector, var, sample_no):
    """
    Gaussian parameter prior with spherical covariance matrix.

    Arguments:
    mean_vector: mean of multivariate gaussian
    var: variance of spherical covariance matrix
    sample_no: samples to be returned

    Returns:
        A list with the sample vectors drawn from the multivariate Gaussian.
    """
    return np.random.multivariate_normal(mean_vector,
                                        var * np.identity(mean_vector.shape[0]),
                                        sample_no)


def monte_carlo_integration(i, dataset, mean_vector, var, sample_no):
    """
    Perform Monte Carlo integration given the model and return the evidence.

    Arguments:
        i: index of model to be used
        dataset: the dataset whose  probability is to be approximated given the
            model and marginalizing over the parameters with a Monte Carlo.
        mean_vector: mean of multivariate gaussian prior of parameters
        var: variance of spherical covariance matrix of parameters
        sample_no: sample number to be used for the approximation of the integral.

    Returns:
        The evidence of model i on the given dataset.
    """
    param_vectors = theta_prior(mean_vector, var, sample_no)

    prob_sum = 0
    for param_vector in param_vectors:
        prob_sum += estimate_dataset_probability(i, dataset, param_vector)
    return (prob_sum / sample_no)

def estimate_dataset_probability(i, dataset, theta):
    """
    Estimates the probability of the dataset given the model and theta

    Arguments:
        i: index of model to be called
        dataset of which the probability is estimated
        theta: vector with parameter values
    """
    product = 1
    for data_sample in dataset:
        # Get sample coordinates into a list x
        x = []
        x.append(data_sample[0][0])
        x.append(data_sample[0][1])
        # Get sample output label
        t = data_sample[1]
        product *= M_i(i, x, t, theta)
    return product

def get_evidence_matrix(datasets, mean_vector, var, sample_no, m):
    """
    Performs monte carlo integration for every model over all the datasets
    and builds a nxm matrix with the evidence for each dataset for every
    model.

    Arguments:
        datasets: a list of n datasets for which the evidence will be computed
            model and marginalizing over the parameters with a Monte Carlo.
        mean_vector: mean of multivariate gaussian prior of parameters
        var: variance of spherical covariance matrix of parameters
        sample_no: sample number to be used for the approximation of the integral.
        m: number of models

    Returns:
        A nxm matrix with the evidence for each dataset and model where n is the
        number of datasets and m is the number of models.
    """

    # Initialize nxm matrix
    evidence_matrix = np.ones((len(datasets), m))

    for i, dataset in enumerate(datasets):
        # For each dataset
        for model_idx in range(m):
            # For each model
            # Approximate evidence by monte carlo integration
            tmp = monte_carlo_integration(model_idx, dataset, mean_vector, var, sample_no)
            evidence_matrix[i][model_idx] = tmp

    return evidence_matrix



def create_index_set(evidence):
    """
     Call as create_index_set(evidence) where evidence
     is an n x m matrix describing the evidence for n data sets and m models.
     Pass the whole evidence matrix to the function rather than summing over the evidence for each data set.

     Debugged and fixed by Arvid Fahlström Myrman.
    """
    E = evidence.sum(axis=1)
    # change 'euclidean' to 'cityblock' for manhattan distance
    dist = distance.squareform(distance.pdist(evidence, 'euclidean'))
    np.fill_diagonal(dist, np.inf)

    L = []
    D = list(range(E.shape[0]))
    L.append(E.argmin())
    D.remove(L[-1])

    while len(D) > 0:
        # add d if dist from d to all other points in D
        # is larger than dist from d to L[-1]
        N = [d for d in D if dist[d, D].min() > dist[d, L[-1]]]

        if len(N) == 0:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmax()])

        D.remove(L[-1])

    # reverse the resulting index array
    return np.array(L)[::-1]

def sort_matrix(matrix, index_set):
    """
    Create a row sorted version of evidence matrix according to index_set.

    Arguments:
        matrix: matrix to be sorted
        index_set: correct order of indices for the sorted matrix to be created

    Returns:
        A sorted version of matrix according to index_set
    """
    sorted_matrix = np.ones(matrix.shape)
    sorted_idx = 0

    for i in index_set:
        sorted_matrix[sorted_idx, :] = matrix[i, :]
        sorted_idx += 1

    return sorted_matrix


# Generate coordinates
coordinates = [(a,b) for a,b in it.product([-1,0,1], repeat=2)]

# Generate all possible dataset label combinations
labels = [a for a in it.product([-1, 1], repeat = 9)]

#Concatenate coordinates with labels to create all possible datasets
datasets = [list(zip(coordinates,label)) for label in labels]
#visualize(datasets[422])

# Set prior mean and sigma
mean_vector = np.zeros(3)
var = 10**3
m = 4
sample_no = 1000
# Create evidence matrix and index set
evidence_matrix = get_evidence_matrix(datasets, mean_vector, var, sample_no, m)
index_set = create_index_set(evidence_matrix)
sorted_evidence_matrix = sort_matrix(evidence_matrix, index_set)

#print(sorted_evidence_matrix)


# Plot evidence for each model
x_axis_values = [i for i in range(evidence_matrix.shape[0])]
fig = plt.figure()
ax = fig.add_subplot(111)
# Set axes titles
ax.set_xlabel('Datasets')
ax.set_ylabel('Evidence')
model_0 = ax.plot(x_axis_values, sorted_evidence_matrix[:, 0], label='Model 0')
model_1 = ax.plot(x_axis_values, sorted_evidence_matrix[:, 1], label='Model 1')
model_2 = ax.plot(x_axis_values, sorted_evidence_matrix[:, 2], label='Model 2')
model_3 = ax.plot(x_axis_values, sorted_evidence_matrix[:, 3], label='Model 3')
ax.legend()
plt.show()
# Plot the first 100 datasets only
x_axis_values = [i for i in range(100)]
fig = plt.figure()
ax = fig.add_subplot(111)
# Set axes titles
ax.set_xlabel('Datasets')
ax.set_ylabel('Evidence')
model_0 = ax.plot(x_axis_values, sorted_evidence_matrix[0:100, 0], label='Model 0')
model_1 = ax.plot(x_axis_values, sorted_evidence_matrix[0:100, 1], label='Model 1')
model_2 = ax.plot(x_axis_values, sorted_evidence_matrix[0:100, 2], label='Model 2')
model_3 = ax.plot(x_axis_values, sorted_evidence_matrix[0:100, 3], label='Model 3')
ax.legend()
plt.show()



# Get most and least probable datasets for each model
max = np.argmax(evidence_matrix, axis = 0)
min = np.argmin(evidence_matrix, axis = 0)
sum = np.sum(evidence_matrix, axis=0)
print(sum)

# Print most probable and least probable sets for each model except for M0
#M1
visualize(datasets[max[1]])
visualize(datasets[min[1]])
#M2
visualize(datasets[max[2]])
visualize(datasets[min[2]])
#M3
visualize(datasets[max[3]])
visualize(datasets[min[3]])

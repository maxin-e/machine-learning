import numpy as np
import random
import matplotlib.pyplot as plt
import time


def sig(x):
    res = 1 / (1 + np.exp(-x))
    return res


def dsig(x):
    dsig = (sig(x)) * (1 - sig(x))
    return dsig


def initialize_parameters(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes):
    # layer 2
    W2 = np.random.rand(numOfHiddenNodes, numOfInputNodes) * 0.01
    b2 = np.random.rand(numOfHiddenNodes, 1) * 0.01

    # layer 3
    W3 = np.random.rand(numOfOutputNodes, numOfHiddenNodes) * 0.01
    b3 = np.random.rand(numOfOutputNodes, 1) * 0.01

    res = {"W2": W2,
           "b2": b2,
           "W3": W3,
           "b3": b3}

    return res


def update_activations(input, parameters):
    # layer 2
    Z2 = np.dot(parameters["W2"], input) + parameters["b2"]
    A2 = sig(Z2)

    # layer 3
    Z3 = np.dot(parameters["W3"], A2) + parameters["b3"]
    A3 = sig(Z3)

    res = {"A1": input,
           "Z2": Z2,
           "A2": A2,
           "Z3": Z3,
           "A3": A3}

    return res


def compute_costs(parameters, activations):
    D3 = -np.sum([activations["A1"], -activations["A3"]], axis = 0)

    D2 = np.multiply(np.dot(parameters["W3"].T, D3), dsig(activations["Z2"]))
    res = {"D2": D2,
           "D3": D3}

    return res


def compute_derivative(costs, activations):
    dW3 = np.dot(costs["D3"], activations["A2"].T)
    db3 = costs["D3"]
    dW2 = np.dot(costs["D2"], activations["A1"].T)
    db2 = costs["D2"]

    res = {"dW3": dW3,
           "db3": db3,
           "dW2": dW2,
           "db2": db2}

    return res


def update_deltas(derivatives, deltas):
    deltas["DW3"] += derivatives["dW3"]
    deltas["DW2"] += derivatives["dW2"]
    deltas["Db3"] += derivatives["db3"]
    deltas["Db2"] += derivatives["db2"]
    return deltas


def reinitialize_deltas(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes):
    DeltaW2 = np.zeros((numOfHiddenNodes, numOfInputNodes))
    DeltaW3 = np.zeros((numOfOutputNodes, numOfHiddenNodes))
    Deltab2 = np.zeros((numOfHiddenNodes, 1))
    Deltab3 = np.zeros((numOfOutputNodes, 1))

    return {"DW2": DeltaW2,
            "Db2": Deltab2,
            "DW3": DeltaW3,
            "Db3": Deltab3}


def update_parameters(parameters, deltas, numOfSamples):
    parameters["W3"] = parameters["W3"] - ALPHA * (1 / numOfSamples * deltas["DW3"] + LAMBDA * parameters["W3"])
    parameters["W2"] = parameters["W2"] - ALPHA * (1 / numOfSamples * deltas["DW2"] + LAMBDA * parameters["W2"])
    parameters["b2"] = parameters["b2"] - ALPHA * (1 / numOfSamples * deltas["Db2"])
    parameters["b3"] = parameters["b3"] - ALPHA * (1 / numOfSamples * deltas["Db3"])


""" ALGORITHM STARTS HERE"""

# initialising constants
ALPHA = 0.9
LAMBDA = 0.0001

# initial parameters
numOfInputNodes = 8
numOfHiddenNodes = 3
numOfOutputNodes = 8
parameters = initialize_parameters(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes)

# initialising examples
samples = np.identity(8)

# initialise error dynamics

costs = {"D3": 1}  # initializising cost dictionary

# LEARNING THE WEIGHTS

output_D3 = [[] for i in range(8)]

while np.amax(np.absolute(costs["D3"])) > 0.04:

    np.random.permutation(samples)

    deltas = reinitialize_deltas(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes)

    for j, sample in enumerate(samples):

        A1 = sample.reshape(8, 1)

        # FORWARD PROPAGATION
        activations = update_activations(A1, parameters)

        # BACKPROPAGATION
        costs = compute_costs(parameters, activations)
        derivatives = compute_derivative(costs, activations)

        output_D3[j].append(np.amax(np.absolute(costs["D3"])))

        deltas = update_deltas(derivatives, deltas)

    update_parameters(parameters, deltas, len(samples))

"""TEST THE NEURAL NETWORK"""
def test(INPUT):
    INPUT.reshape(8, 1)

    # layer 3
    Z2 = np.dot(parameters["W2"], INPUT).reshape(3, 1) + parameters["b2"]
    A2 = sig(Z2)

    # layer 3
    Z3 = np.dot(parameters["W3"], A2).reshape(8, 1) + parameters["b3"]
    A3 = sig(Z3)

    return A3


INPUT = np.array([1, 0, 0, 0, 0, 0, 0, 0])
print(test(INPUT))
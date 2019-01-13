"""This modules implement learning and inference in a vanilla
Hoepfield network."""

import numpy as np

def update_activity(weights, prev_activity):
    """Follow the Hopfield network's dynamics for one step.

    Given the network weights and the previous network activity,
    compute the network activity at the next time step.

    Input arguments:
    weights -- a KxK array of synaptic weights
               weights[i, j] represent the weight between neurons i and j
               weights[i, i] == 0 for all i (no recurrent connections)
               weights[i, j] == weights[j, i] for all i, j (symmetric weights)

    prev_activity -- a K-dim array of neural activity
                     prev_activity[i] is the activity of neuron i at the
                     previous point in time (either -1 or 1)

    Output:
    activity -- a K-dim of neural activity
    """

    # compute the activity before thresholding
    raw_activity = np.dot(weights, prev_activity)
    # threshold activity
    activity = np.where(raw_activity >= 0., 1., -1.)

    return activity


def store_pattern_inefficient(weights, pattern):
    """Store a new pattern in an Hopfield network.

    NOTE: this is an inefficient version of the store_pattern
    function that I later optimized (see store_pattern).

    The weights are modified in-place (i.e., the old weights
    are overwritten)

    Input arguments:
    weights -- a KxK array of synaptic weights
               weights[i, j] represent the weight between neurons i and j
               weights[i, i] == 0 for all i (no recurrent connections)
               weights[i, j] == weights[j, i] for all i, j (symmetric weights)

    pattern -- a K-dim array containing the pattern to be stored
               pattern[i] is either -1 or 1
    """

    k = weights.shape[0]
    for i in range(k):
        for j in range(i+1, k):
            prod = pattern[i] * pattern[j]
            weights[i, j] += prod
            weights[j, i] = weights[i, j]


def store_pattern(weights, pattern):
    """Store a new pattern in an Hopfield network.

    The weights are modified in-place (i.e., the old weights
    are overwritten)

    Input arguments:
    weights -- a KxK array of synaptic weights
               weights[i, j] represent the weight between neurons i and j
               weights[i, i] == 0 for all i (no recurrent connections)
               weights[i, j] == weights[j, i] for all i, j (symmetric weights)

    pattern -- a K-dim array containing the pattern to be stored
               pattern[i] is either -1 or 1
    """
    
    weights += np.outer(pattern, pattern)
    # set elements on the diagonal to 0
    for i in range(weights.shape[0]):
        weights[i,i] = 0.

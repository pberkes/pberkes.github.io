import numpy as np
import pylab as plt
import mdp

def show_letter(pattern, sz=(20,16)):
    """Plot activity pattern."""
    if sz!=None:
        pattern = np.reshape(pattern, sz)
    plt.figure(1)
    plt.clf()
    plt.imshow(1. - pattern,
               cmap=plt.cm.gray, interpolation='nearest')
    plt.draw()

def load_data():
    """Return the data for Exercise 3."""
    data = np.load('letters_ACS.npz')
    return data['letters'], data['labels']



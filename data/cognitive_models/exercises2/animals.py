import numpy as np
import scipy as sp
import scipy.ndimage
import pylab as plt

from hopfield import update_activity, store_pattern


# --------------------------------------------------------------------
# Some functions that will be useful for the exercises
# --------------------------------------------------------------------

def load_binary_image(filename):
    """Load a binary image and return a +1/-1 pattern."""

    image = sp.ndimage.imread(filename)
    # transform in floating point +1/-1 array
    pattern = image.astype('f')
    pattern = np.where(pattern == 0., 1., -1.)
    return pattern.flatten()


def show_pattern(pattern, fig_num=1):
    size = round(np.sqrt(pattern.shape[0]))
    plt.figure(fig_num)
    plt.clf()
    plt.imshow(pattern.reshape(size, size), cmap=plt.cm.gray, interpolation='nearest')
    plt.draw()


def random_flip(pattern, prob_flip=0.1):
    """Flip random elements of a pattern.

    Return a copy of the pattern with random elements flipped with
    probability 'prob_flip'.
    """

    flip = (np.random.rand(*pattern.shape) <= prob_flip)
    corrupt = pattern.copy()
    corrupt[flip] *= -1
    return corrupt


# --------------------------------------------------------------------
# Your code begins here
# --------------------------------------------------------------------

# load images
animals_filenames = ['lion_small.png',
                     'tiger_small.png', 'monkey_small.png']
animal_patterns = []
for fname in animals_filenames:
    animal_patterns.append(load_binary_image(fname))

# show one of the patterns
show_pattern(animal_patterns[0], fig_num=1)

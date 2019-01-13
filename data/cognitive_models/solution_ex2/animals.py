import scipy.ndimage
import numpy as np
import scipy as sp
import pylab as plt

from hopfield import update_activity, store_pattern


def load_binary_image(filename):
    """Load a binary image and return a +1/-1 pattern."""

    image = sp.ndimage.imread(filename)
    # transform in floating point array
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


# load images
animals_filenames = ['zebra_small.png', 'lion_small.png',
                     'tiger_small.png', 'monkey_small.png']
animal_patterns = []
for fname in animals_filenames:
    animal_patterns.append(load_binary_image(fname))

# show one of the patterns
show_pattern(animal_patterns[0], fig_num=1)

# create the weights for a new Hopfield network
dim = animal_patterns[0].shape[0]
weights = np.zeros((dim, dim))
# store patterns
for idx in range(3):
    store_pattern(weights, animal_patterns[idx])

# select one pattern and add noise to it
corrupt = random_flip(animal_patterns[1], prob_flip=0.3)
show_pattern(corrupt, fig_num=1)

# reconstruct pattern
activity = corrupt
for i in range(1,5):
    activity = update_activity(weights, activity)
    show_pattern(activity, fig_num=i+1)



mist = load_binary_image('mistery.png')
activity = update_activity(weights, mist)
show_pattern(activity)

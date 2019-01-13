import unittest
import numpy as np

from hopfield import update_activity, store_pattern

def random_activity(k):
    """Return random activity vector of length 'k'."""
    activity = np.random.rand(k)
    return np.where(activity > 0.5, 1., -1.)


class TestHopfieldActivation(unittest.TestCase):

    # Add tests for Hopfield network activitation function (update_activity)
    pass


class TestHopfieldLearning(unittest.TestCase):

    # Add tests for Hopfield network learninr function (store_pattern)
    pass

    
if __name__ == '__main__':
    unittest.main()

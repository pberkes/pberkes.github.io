import unittest
import numpy as np

from sampling import sample_h

class TestRBMSampling(unittest.TestCase):

    def test_sample_h(self):
        # I: number of visible units, J: number of hidden units
        I, J = 2, 4

        # the weights
        weights = np.array([[10000., 0., 10000., 0.],
                            [0., 10000., 0., 10000.]])

        # the four possible inputs for the visible units
        v = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [1., 1.]])

        # COMPLETE THIS ARRAY WITH THE RESULT OF EXERCISE 2a
        # expected probabilities for the four inputs
        # expected_probs[k,j] = probability that the hidden unit j is 1
        expected_probs = np.array([[???, ???, ???, ???],
                                   [???, ???, ???, ???],
                                   [???, ???, ???, ???],
                                   [???, ???, ???, ???]])

        # loop over the four possible inputs
        for k in range(4):
            # sample 100 times from the hidden units
            h = np.zeros((100, J))
            for n in range(100):
                prob, h[n, :] = sample_h(weights, v[k, :])

            # check inferred probabilities
            np.testing.assert_array_almost_equal(prob, expected_probs[k, :], 8)

            # check sampled values
            distr = h.mean(axis=0)
            np.testing.assert_array_almost_equal(distr, expected_probs[k, :], 1)

if __name__ == '__main__':
    unittest.main()

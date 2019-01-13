import unittest
import numpy as np

from hopfield import update_activity, store_pattern

def random_activity(k):
    """Return random activity vector of length 'k'."""
    activity = np.random.rand(k)
    return np.where(activity > 0.5, 1., -1.)

class TestHopfieldActivation(unittest.TestCase):

    def test_simple_network(self):
        weights = np.array([[    0., 2000., -1000.],
                            [ 2000.,    0.,     0.],
                            [-1000.,    0.,     0.]])

        test_cases = [(np.array([0., 1., -1.]), np.array([1., 1., 1.])),
                      (np.array([1.,  0., 0.]), np.array([1., 1., -1.])),
                      (np.array([-1.,  1., 1.]), np.array([1., -1., 1.]))]
        
        for activity, expected in test_cases:
            activity = update_activity(weights, activity)
            np.testing.assert_array_equal(activity, expected)


class TestHopfieldLearning(unittest.TestCase):

    def test_random(self):
        n = 10000
        k = 5
        weights = np.zeros((k, k))
        for _ in range(n):
            x = random_activity(k)
            store_pattern(weights, x)
        np.testing.assert_array_almost_equal(weights/n, np.zeros((k, k)), 1)

    def test_same(self):
        n = 1000
        k = 3
        weights = np.zeros((k, k))
        for _ in range(n):
            if np.random.rand() > 0.5:
                x = np.array([1., 1., 1.])
            else:
                x = np.array([-1., -1., -1.])
            store_pattern(weights, x)

        for i in range(1, k):
            for j in range(i+1, k):
                self.assertAlmostEqual(weights[i, j], n)
                self.assertAlmostEqual(weights[j, i], n)

    def test_fuzz(self):
        n = 10000
        p1 = 0.9
        p2 = 0.4
        p3 = p1*p2 + (1.-p1)*(1.-p2)
        desired = np.array([[0.,        2*p1 - 1., 2*p2 - 1.],
                            [2*p1 - 1.,        0., 2*p3 - 1.],
                            [2*p2 - 1., 2*p3 - 1.,        0.]])
        
        weights = np.zeros((3, 3))
        for _ in range(n):
            x = np.zeros(3)
            x[0] = 1. if np.random.rand() <= 0.5 else -1.
            x[1] = x[0] if np.random.rand() <= p1 else -x[0]
            x[2] = x[0] if np.random.rand() <= p2 else -x[0]
            store_pattern(weights, x)

        np.testing.assert_array_almost_equal(weights/n, desired, 1)

    
if __name__ == '__main__':
    unittest.main()

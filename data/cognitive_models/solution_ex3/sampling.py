import numpy as np

def sample_h(weights, v):
    """Sample from hidden units given the state of visible units.
    
    Arguments:
    w -- IxJ array
         w[i, j] is the weight between visible unit 'i' and hidden unit 'j'

    v -- I-dimensional array
         v[i] is the state (0 or 1) of visible unit i
    
    Returns (prob_h, h):
    prob_h -- J-dimensional array
              prob_h[j] is the probability of unit j being +1, i.e.,
              prob_h[j] = P( h[j]=1 | v)
    
    h -- J-dimensional array
         h is one sample from prob_h, i.e.
         h[j] = +1 with probability probs_h[j]
         h[j] = 0  with probability 1. - probs_h[j]
    """
    
    # compute P(h=1 | v)
    prob_h = 1. / (1. + np.exp(- np.dot(v, weights)))
    # sample h
    h = np.where(prob_h > np.random.random(prob_h.shape), 1., 0.)
    return prob_h, h

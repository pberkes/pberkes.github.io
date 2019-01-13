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
    data = np.load('letter_S.npy')
    return data


data = load_data()
show_letter(data[10,:])

rbm  =  mdp.nodes.RBMNode(100)
for i in range(100):
    rbm.train(data)
    
v = np.random.randint(0, 2, size=(1, 320))
# this is the probability of v=1
pv = v
ph, h = rbm.sample_h(pv)
pv, v = rbm.sample_v(ph)
show_letter(v)
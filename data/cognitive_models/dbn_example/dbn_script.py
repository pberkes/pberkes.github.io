"""Train a DBN on handwritten digits using greedy learning."""

import numpy as np
import mdp

from utils import load_data, show_letter


letters, labels = load_data()
show_letter(letters[0,:])



# ================================
# constants
# ================================

I = 20*16            # dimensionality of the data
L = 3                # dimensionality of the labels
N_HIDDEN = 100       # number of hidden units
N = letters.shape[0] # total number of letters

N_EPOCHS = 500       # number of training iterations




# ================================
# define a 3-layers DBN
# ================================
obs = mdp.nodes.RBMNode(N_HIDDEN)
hid = mdp.nodes.RBMNode(N_HIDDEN)
top = mdp.nodes.RBMWithLabelsNode(N_HIDDEN, L)





# ================================
# train the DBN greedily (step-by-step)
# ================================

# shuffle input data
permutation = np.random.permutation(N)
shuffled_letters = letters[permutation,:]
shuffled_labels = labels[permutation,:]

# ---- train first layer
print '\n------ train first layer'
v = shuffled_letters
for n in range(N_EPOCHS):
    is_verbose = (n % 50 == 0)
    obs.train(v, verbose=is_verbose)

# compute "output" of first layer
ph, h = obs.sample_h(v)

# ---- train second layer
print '\n------ train second layer'
v = ph
for n in range(N_EPOCHS):
    is_verbose = (n % 50 == 0)
    hid.train(v, verbose=is_verbose)

# compute "output" of second layer
ph, h = hid.sample_h(v)

# ---- train third layer (with labels)
print '\n------ train third layer'
v = ph
for n in range(N_EPOCHS):
    is_verbose = (n % 50 == 0)
    top.train(v, shuffled_labels, verbose=is_verbose)





# ================================
# use the network for recognition
# ================================

misterious_letter = letters[8,:] # easy
#misterious_letter = letters[80,:] # hard
show_letter(misterious_letter)

# start with random label vector
l = np.random.randint(2, size=(1,L))

label_guess = []

# sample up
ph, h = obs.sample_h(misterious_letter[None,:])
ph, h = hid.sample_h(ph)
# sample up and down a few times, colelct guesses
for i in range(100):
    pt, t = top.sample_h(ph, l)
    pv, pl, v, l = top.sample_v(pt)
    # record label vector
    label_guess.append(l)

# compute probability of each label
prob = np.concatenate(label_guess).mean(axis=0)




# ================================
# use the network for generating letters
# ================================

# select which letter
l = np.array([[1., 0., 0.]])

# start with random visible units
v = np.random.randint(2, size=(1,N_HIDDEN))

# dream a while at the top level
for i in range(2000):
    pt, t = top.sample_h(v, l)
    pv, _, v, _ = top.sample_v(pt)

def dream(v, l, nsteps=1):
    for _ in range(nsteps):
        # propagate down and visualize
        pt, t = top.sample_h(v, l)
        pv, _, v, _ = top.sample_v(pt)
    pv1, v1 = hid.sample_v(v)
    pv2, v2 = obs.sample_v(v1)
    show_letter(pv2)
    return v

v = dream(v, l, nsteps=200)

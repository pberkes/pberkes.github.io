"""Save letters A,C,S from the database for the exercises."""

import scipy as sp
import scipy.io as sio

# the data ends up in a nclasses x ntrain x sizepatch array 'data'

tmp = sio.loadmat('binaryalphadigs.mat')
NTRAIN = 39
CLASSES = [10, 12, 28] # A, C, S
NCLASSES = len(CLASSES)
I, L = 20*16, 3
N = NTRAIN*NCLASSES

# organize data
data = sp.zeros((NCLASSES, NTRAIN, I))
labels = sp.zeros((NCLASSES, NTRAIN, L))
for k in range(L):
    for m in range(NTRAIN):
        data[k,m,:] = (tmp['dat'][CLASSES[k],m].ravel()).astype('d')
        labels[k,m,k] = 1.

# prepare observations, labels
letters = data.reshape(N, I)
labels = labels.reshape(N, L)

sp.savez('letters_ACS', letters=letters, labels=labels)

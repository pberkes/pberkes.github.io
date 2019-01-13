import numpy as np

def sample_A_or_B(prob):
    return 'A' if np.random.rand() < prob else 'B'

x = np.array([sample_A_or_B(0.3) for _ in range(10000)])

print (x == 'A').sum()
print (x == 'B').sum()

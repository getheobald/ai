import numpy as np
from scipy.special import expit, softmax

# using np to solve the formulas I wrote by hand
# for gradient of categorical crossentropy loss for the given NN

# SETUP

# inputs
xi = np.array([6, 4, 7, 2, 1])
yi = np.array([0, 1, 0])

# weight matrices
W = np.array([[0, 4, -4, 5, 4],
             [2, 2, -4, 3, 0]])
V = np.array([[1, -2],
              [2, 3],
              [1, 2]])

# W dot xi
z = W @ xi

#ReLU of W dot xi
h = np.maximum(0, z)

# V dot h
o = V @ h

# sigmoid of V dot h
sig = expit(o)

# softmax of sigmoid output
prob = softmax(sig)



# GRADIENTS

grad_shared = (prob - yi) * sig * (1 - sig)

# grad of L wrt W
# (softmax - yi) sig (1 - sig) V ReLU xi
d_relu = (z > 0).astype(float) # capture piecewise func by casting boolean
transpose_V = V.T # need V in the right shape to mult by rest of equation
grad_W = np.outer((transpose_V @ grad_shared) * d_relu, xi)


# grad of L wrt V
# (softmax - yi) sig (1 - sig) ReLU(W dot xi)
grad_V = np.outer(grad_shared, h)

print("grad of w1:", grad_W[0])
print("grad of w2:", grad_W[1])
print("grad of v1:", grad_V[0])
print("grad of v2:", grad_V[1])
print("grad of v3:", grad_V[2])
import numpy as np
from scipy.special import expit
from scipy.special import softmax

# input
xi = np.array([6, 4, 7, 2, 1])

# hidden layer
w1 = np.array([0, 4, -4, 5, 4])
b_w1 = 1
w2 = np.array([2, 2, -4, 3, 0])
b_w2 = 1

# output layer
v1 = np.array([1, -2])
b_v1 = -1
v2 = np.array([2, 3])
b_v2 = 0
v3 = np.array([1, 2])
b_v3 = 2

# hidden layer inputs
z1 = np.dot(w1, xi) + b_w1
z2 = np.dot(w2, xi) + b_w2

# hidden layer activation func (relu)
# defining it myself for clarity
def relu(x):
    return np.maximum(0, x)

h1 = relu(z1)
h2 = relu(z2)
h = np.array([h1, h2])

# output later inputs
u1 = np.dot(v1, h) + b_v1
u2 = np.dot(v2, h) + b_v2
u3 = np.dot(v3, h) + b_v3

# output layer activation func (sigmoid)
s1 = expit(u1)
s2 = expit(u2)
s3 = expit(u3)

# softmax
probabilities = softmax([s1, s2, s3])

# cross entropy loss for 1 class
loss = -np.log(probabilities[1])

print(loss) # 1.05973763730744
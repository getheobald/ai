import numpy as np

# transition matrix (rows = from, cols = to)
P = np.array([
    [0.0, 0.7, 0.3],  # S1
    [0.5, 0.5, 0.0],  # S2
    [0.2, 0.8, 0.0]   # S3
])

# lin alg method of getting stationary distribution (solve πP = π)
# rearrange to (P^T - I)π = 0, replace one equation with sum(π) = 1
A = (P.T - np.eye(3))
A[-1] = 1  # replace last row with normalization constraint
b = np.zeros(3)
b[-1] = 1  # sum(π) = 1

pi = np.linalg.solve(A, b)
print("Lin alg method:")
print(f"  π(S1) = {pi[0]:.4f}, π(S2) = {pi[1]:.4f}, π(S3) = {pi[2]:.4f}")

# iterative method of getting stationary distribution
P_inf = np.linalg.matrix_power(P, 10000)
pi2 = P_inf[0]  # all rows converge to π
print("\nIteration method:")
print(f"  π(S1) = {pi2[0]:.4f}, π(S2) = {pi2[1]:.4f}, π(S3) = {pi2[2]:.4f}")

# verification
print("\nVerification (πP should equal π):")
print(f"  πP = {pi @ P}")
print(f"  π  = {pi}")

P0 = np.array([0.5, 0.1, 0.4])
P2 = np.linalg.matrix_power(P, 2)
result = P0 @ P2
print(result[1])  # index 1 = S2
print(f"T^2 matrix is {P2}")

T = np.array([[0.5, 0.5],[1/3, 2/3]])
hidden_markov_result = np.linalg.matrix_power(T, 100)
print(f"Hidden markov stationary distribution is {hidden_markov_result}")
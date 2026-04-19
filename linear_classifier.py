import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)  # 100 points from -4 to 4

# (a) weight vector w = [4, -5] as an arrow from origin
plt.quiver(0, 0, 4, -5, angles='xy', scale_units='xy', scale=1, 
           color='red', label='Weight vector w = [4, -5]')

# (b) and (c) decision boundary y = (4/5)x - 2/5
# this line is BOTH the decision boundary AND orthogonal to w
y_boundary = (4/5)*x - 2/5
plt.plot(x, y_boundary, 'b-', label='Decision boundary (orthogonal to w)')

plt.axis('equal')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(0, color='black', linewidth=0.5)  # x axis
plt.axvline(0, color='black', linewidth=0.5)  # y axis
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
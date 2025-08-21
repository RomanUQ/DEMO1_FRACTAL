
# PROMT: "Generate a Python script to plot a 2D Gaussian function using Numpy and Matplotlib"

import numpy as np
import matplotlib.pyplot as plt

# Define the 2D Gaussian function
def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * \
           np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
                    ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))

# Create a meshgrid for x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Compute Z values (the Gaussian function)
Z = gaussian_2d(X, Y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1)

# Plot the 2D Gaussian as a contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('2D Gaussian Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()

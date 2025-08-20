
# PROMT: "convert this script to PyTorch and to use its Tensors instead of Numpy"

import torch
import matplotlib.pyplot as plt

# Define the 2D Gaussian function using PyTorch
def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    return (1 / (2 * torch.pi * sigma_x * sigma_y)) * \
           torch.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +  ####sin
                       ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))

# Create a meshgrid using PyTorch
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')  # Ensure the correct indexing for contour plots

# Evaluate the Gaussian function
Z = gaussian_2d(X, Y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1)

# ---- This is the gaussian * sin(5(x + y)) ----

#G = gaussian_2d(X, Y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1)         
#S = torch.sin(X + Y)   
#Z = G * S 

# Convert PyTorch tensors to NumPy arrays for plotting
X_np = X.numpy()
Y_np = Y.numpy()
Z_np = Z.numpy()

# Plot using Matplotlib
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_np, Y_np, Z_np, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('2D Gaussian Function (PyTorch)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()

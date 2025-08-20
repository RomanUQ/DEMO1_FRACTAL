import torch
import matplotlib.pyplot as plt

# Create a meshgrid using PyTorch
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')

Z = torch.sin(X + Y)         #### or change to cos

# Convert PyTorch tensors to NumPy arrays for plotting
X_np = X.numpy()
Y_np = Y.numpy()
Z_np = Z.numpy()

# Plot the sine stripes
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_np, Y_np, Z_np, levels=100, cmap='plasma')
plt.colorbar(contour)
plt.title('2D sin function and stripes')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()

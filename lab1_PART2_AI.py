
# PROMT: "generate the Mandelbrot set implemented in 
# PyTorch and that runs on the GPU (i.e. seems to run fast and allows fast rendering)"

import torch
import matplotlib.pyplot as plt

# Set image resolution and number of iterations
width, height = 1000, 1000
max_iter = 100

# Define coordinate range in the complex plane
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a 2D grid of complex numbers
x = torch.linspace(x_min, x_max, width, device=device)
y = torch.linspace(y_min, y_max, height, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')
C = X + 1j * Y

# Initialize Z and iteration count
Z = torch.zeros_like(C)
diverged = torch.zeros(C.shape, dtype=torch.int32, device=device)

# Mandelbrot iteration
for i in range(max_iter):
    mask = torch.abs(Z) <= 2
    Z[mask] = Z[mask] * Z[mask] + C[mask]
    diverged[mask & (torch.abs(Z) > 2)] = i

# Move data to CPU for plotting
result = diverged.cpu().numpy()

# Plot the Mandelbrot set
plt.figure(figsize=(8, 8))
plt.imshow(result.T, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.title('Mandelbrot Set (PyTorch, GPU)')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.colorbar(label='Iteration Count')
plt.show()

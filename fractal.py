import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pythag_tree(depth=20, angle_step=np.pi/4, baseLen=0.25):
    # base case: one branch pointing up
    x = torch.tensor([0.5], device=device)
    y = torch.tensor([0.0], device=device)
    theta = torch.tensor([np.pi/2], device=device)
    L = torch.tensor([baseLen], device=device)

    segs = []

    for _ in range(depth):
        # compute endpoints for all current branches
        x2 = x + L * torch.cos(theta)
        y2 = y + L * torch.sin(theta)

        start = torch.stack([x, y],  dim=1)
        end = torch.stack([x2, y2], dim=1)
        seg = torch.stack([start, end], dim=1)
        segs.append(seg)

        # spawn two children, rotated by angle_step
        new_theta = torch.cat([theta + angle_step, theta - angle_step])

        # both children shrink by sqrt(2)/2
        new_L = torch.cat([L, L]) * (np.sqrt(2)/2)

        # both children start at parents end (x2, y2)
        new_x = torch.cat([x2, x2])
        new_y = torch.cat([y2, y2])

        x = new_x
        y = new_y
        theta = new_theta
        L = new_L

    # concat all levels into one array
    return torch.cat(segs, dim=0).cpu().numpy()

segs = pythag_tree(depth=15)

# plot
ax = plt.gca()
ax.add_collection(LineCollection(segs)) # add every line segment at once
ax.set_aspect('equal')
#ax.set_xlim(0.45, 0.55)
#ax.set_ylim(0.70, 1.10)
plt.axis('off')
plt.show()




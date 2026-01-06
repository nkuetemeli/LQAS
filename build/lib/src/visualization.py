from torch import nn, optim
from utils_models import *
from utils_datasets import *
from QML.layered_qas import *
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy

"""
3 - chair
10 - monitor
11 - bathtub
4 - table
6 - toilet
9 - bed
13 - night-stand
16 - dresser
19 - sofa
28 - desk
"""

N_qubits = [6, 9, 12]
N_samples = 10


fig, axis = plt.subplots(N_samples, len(N_qubits), subplot_kw={'projection': '3d'})
fig.set_size_inches((6, 10))

for i, n_qubits in enumerate(N_qubits):

    params = get_args(n_qubits=n_qubits)
    size = 2**(params.n_qubits//3)

    seed_value = 10
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    labels = []
    for k in range(50):


        tensor_np, label_np = train_loader.dataset.__getitem__(k)

        if label_np not in labels and len(labels) < N_samples:
            ax = axis[len(labels), i]
            labels.append(label_np)
            tensor_np = tensor_np.reshape(size, size, size)


            # Threshold to create a voxel grid (non-zero density)
            threshold = 0.00
            voxel_grid = tensor_np > threshold  # Binary grid for where voxels exist

            # Colors based on density (grayscale values)
            colors = np.zeros(voxel_grid.shape + (4,))  # RGBA format
            colors[..., 0] = np.clip(tensor_np * 25, 0, 1)  # R (shades of grey)
            colors[..., 1] = np.clip(tensor_np * 25, 0, 1)  # G
            colors[..., 2] = np.clip(tensor_np * 25, 0, 1)  # B
            colors[..., 3] = np.clip(tensor_np * 20, 0, 1)  # Alpha tied to density

            # Plot the voxel model
            ax.voxels(voxel_grid, facecolors=colors, edgecolor=(0.5,0.5,0.5,0.1))

            ax.grid(False)  # Hide grid
            ax.set_axis_off()

            # Isometric view
            ax.view_init(elev=30, azim=-135)  # Adjust for an isometric perspective
            ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio equal
            # ax.set_zlim([0, size-(2*i)])

            if len(labels)==1:
                ax.set_title(f'k = {params.n_qubits//3}')
            if i==0:
                ax.text2D(-.12, 0., classes[label_np], rotation=90, verticalalignment='center')

# Save the figure
plt.savefig(f'ResultsFLFalse/viz.pdf', bbox_inches='tight')
plt.show()


# ############################################################################################
# # Save table for teaser
#
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
#
# params = get_args(n_qubits=9)
# size = 2 ** (params.n_qubits // 3)
#
# seed_value = 10
# np.random.seed(seed_value)
# torch.manual_seed(seed_value)
# torch.cuda.manual_seed(seed_value)
#
# # Get data
# classes, train_loader, val_loader, test_loader = get_data_loaders(params)
#
# tensor_np, label_np = train_loader.dataset.__getitem__(2)
# tensor_np = tensor_np.reshape(size, size, size)
#
# threshold = 0.00
# voxel_grid = tensor_np > threshold  # Binary grid for where voxels exist
#
# # Colors based on density (grayscale values)
# colors = np.zeros(voxel_grid.shape + (4,))  # RGBA format
# colors[..., 0] = np.clip(tensor_np * 25, 0, 1)  # R (shades of grey)
# colors[..., 1] = np.clip(tensor_np * 25, 0, 1)  # G
# colors[..., 2] = np.clip(tensor_np * 25, 0, 1)  # B
# colors[..., 3] = np.clip(tensor_np * 20, 0, 1)  # Alpha tied to density
#
# ax.voxels(voxel_grid, facecolors=colors, edgecolor=(0.5, 0.5, 0.5, 0.1))
#
# ax.grid(False)  # Hide grid
# ax.set_axis_off()
#
# # Isometric view
# ax.view_init(elev=30, azim=-135)  # Adjust for an isometric perspective
# ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio equal
# # ax.set_zlim([0, size-(2*i)])
#
# plt.savefig(f'ResultsFLFalse/viz_table.png', bbox_inches='tight')
# plt.show()
#
#

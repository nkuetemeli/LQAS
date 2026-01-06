import argparse
import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
import open3d as o3d
import torch
from torch.utils.data import random_split, DataLoader
from dataset import *

# Set the random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


def get_args(n_qubits=None):
    parser = argparse.ArgumentParser(description="Input argument parser")
    n_qubits = 9 if n_qubits is None else n_qubits
    n_layers = 20
    voxel_size = 1 / (2 ** (n_qubits / 3) - 1)

    which_model_net = 'ModelNet10'
    reduce_ds = False
    frozen_linear = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add arguments
    parser.add_argument('--n_qubits', type=int, default=n_qubits, help='')
    parser.add_argument('--voxel_size', type=int, default=voxel_size, help='')
    parser.add_argument('--n_layers', type=int, default=n_layers, help='')
    parser.add_argument('--which_model_net', type=str, default=which_model_net, help='')
    parser.add_argument('--frozen_linear', type=bool, default=frozen_linear, help='')
    parser.add_argument('--reduce_ds', type=bool, default=reduce_ds, help='')
    parser.add_argument('--device', type=torch.device, default=device, help='')
    parser.add_argument('--n_classes', type=int, default=int(which_model_net[-2:]), help='')

    # Parse arguments and return as an object
    return parser.parse_args()


def get_data_loaders(params):

    # Get datasets
    classes, train_dataset, val_dataset, test_dataset = (
        load_datasets(which_model_net=params.which_model_net, voxel_size=params.voxel_size, reduce=params.reduce_ds, factor=.1))
    batch_size = 64

    # Create dataloaders for the new training and validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return classes, train_loader, val_loader, test_loader


def load_datasets(which_model_net='ModelNet10', voxel_size=1/7, reduce=False, factor=1, dimension='1d'):
    classes, train_files, train_labels, test_files, test_labels = data_parser(which_model_net=which_model_net)

    # Instantiate the dataset and dataloader
    train_dataset = ModelNetDataset(train_files, train_labels,
                                    transform=lambda x: transform(x, voxel_size=voxel_size))
    test_dataset = ModelNetDataset(test_files, test_labels,
                                   transform=lambda x: transform(x, voxel_size=voxel_size))

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    if reduce:
        # Reduce the size of the training and test datasets
        factor = .1
        reduced_train_size = int(factor * train_size)  # Use % of the training set for faster training
        train_dataset, _ = random_split(train_dataset, [reduced_train_size, train_size - reduced_train_size])
        reduced_test_size = int(factor * test_size)  # Use % of the test set
        test_dataset, _ = random_split(test_dataset, [reduced_test_size, test_size - reduced_test_size])

    # Split train in train and val dataset
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)  # 20% of the dataset for validation
    train_size = dataset_size - val_size  # Remaining 80% for training
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return classes, train_dataset, val_dataset, test_dataset



def data_parser(which_model_net='ModelNet10'):
    input_dir = os.path.join('../data', which_model_net)
    classes = {i: f_name for i, f_name in
               enumerate([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])}
    # classes = {0: classes[2], 1: classes[7]}

    train_files = []
    train_labels = []
    test_files = []
    test_labels = []
    for class_label, class_name in classes.items():
        for f in os.listdir(os.path.join(input_dir, class_name, 'train')):
            if f.endswith('.off'):
                train_files.append(os.path.join(input_dir, class_name, 'train', f))
                train_labels.append(class_label)
        for f in os.listdir(os.path.join(input_dir, class_name, 'test')):
            if f.endswith('.off'):
                test_files.append(os.path.join(input_dir, class_name, 'test', f))
                test_labels.append(class_label)

    return classes, train_files, train_labels, test_files, test_labels


def normalize_pcd(pcd):
    # Assuming you already have a point cloud object `pcd`
    vertices = np.asarray(pcd.points)

    # Step 1: Find the min and max coordinates along each axis
    min_coords = np.min(vertices)
    max_coords = np.max(vertices)

    # Step 2: Compute the range for each axis
    range_coords = max_coords - min_coords
    range_coords = 1e-6 if range_coords == 0 else range_coords

    # Step 3: Normalize the point cloud
    normalized_vertices = (vertices - min_coords) / range_coords

    # Update the point cloud with normalized coordinates
    pcd.points = o3d.utility.Vector3dVector(normalized_vertices)
    return pcd


def save_pcd_as_image(pcd, filename="output.png", width=600, height=600):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=height, width=width)
    vis.add_geometry(pcd)

    view = vis.get_view_control()
    # view.set_zoom(1.2)  # Zoom in/out
    # view.set_front([0, 0, 1])  # Set viewing direction
    # view.set_lookat([.5, .9, 0])  # Set center point
    # view.set_up([0, 1, 0])  # Set the up direction
    # view.rotate(100.0, -200.0)  # Rotate view (dx, dy)

    view.set_front([0, 0, 1])  # Set viewing direction
    # view.set_lookat([.5, .9, 0])  # Set center point
    view.set_up([0., .6, 0])  # Set the up direction
    view.rotate(0.0, -360.0)  # Rotate view (dx, dy)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()

    print(f"Saved image as {filename}")
    return


def get_pcd_densities(pcd, voxel_size):
    # Example vertex coordinates
    vertices = np.asarray(pcd.points)

    # Initialize a dictionary to store the count of vertices per voxel
    voxel_point_count = {}

    # Get the voxel indices for each vertex, considering negative coordinates
    for vertex in vertices:
        voxel_index = tuple(np.round(vertex / voxel_size).astype(np.int32))  # Calculate voxel index for each vertex
        if voxel_index in voxel_point_count:
            voxel_point_count[voxel_index] += 1
        else:
            voxel_point_count[voxel_index] = 1

    # Calculate the density for each voxel
    total_points = len(vertices)
    voxel_density = {k: v / total_points for k, v in voxel_point_count.items()}

    return voxel_density


def transform(file, voxel_size=1/7, visualize=None):
    size = int(1 / voxel_size + 1)
    pcd = o3d.io.read_triangle_mesh(file)
    pcd = pcd.sample_points_uniformly(number_of_points=5000)
    pcd = normalize_pcd(pcd)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    voxel_density = get_pcd_densities(pcd, voxel_size=voxel_grid.voxel_size)
    voxel_density_tensor = np.zeros(2 ** (int(3 * np.log2(size))), dtype=np.float32)
    for x in np.arange(size).astype(int):
        for y in np.arange(size).astype(int):
            for z in np.arange(size).astype(int):
                voxel_density_tensor[
                    np.ravel_multi_index((x, y, z), (size, size, size))
                ] = voxel_density.get((x, y, z), 0)


    # voxel_density_tensor = (voxel_density_tensor != 0).astype(np.float64)
    voxel_density_tensor /= np.linalg.norm(voxel_density_tensor)

    # Visualize the voxel grid
    if visualize is not None:
        to_plot = None
        if visualize=='voxel':
            for i, voxel in enumerate(voxel_grid.get_voxels()):
                voxel.color = 3*(voxel_density.get(tuple(voxel.grid_index), 0) * np.ones((3, )) / max(voxel_density_tensor))
                voxel_grid.add_voxel(voxel)
                to_plot = voxel_grid
        elif visualize=='pcd':
                to_plot = pcd
        else:
            assert False, 'Unknown visualization option should be "voxel" or "pcd"'

        save_pcd_as_image(to_plot, filename="ResultsFLFalse/pcd_top.png")
        o3d.visualization.draw_geometries([to_plot])
    return voxel_density_tensor


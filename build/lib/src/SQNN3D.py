import numpy as np
import os


os.environ["OMP_NUM_THREADS"] = "1"
from dataset import *

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pennylane as qml
import csv
import matplotlib.pyplot as plt
import os
import open3d as o3d


class ModelNetDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list of str): List of file paths to the training data.
            labels (list): List of labels corresponding to each file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            voxel_density (Tensor): The voxel_density of sample Grid.
            label (Tensor or int): The corresponding label.
        """
        # Load the mesh and label
        file = self.file_paths[idx]
        label = self.labels[idx]

        # Apply transformations if any
        if self.transform:
            voxel_density = self.transform(file)

        return voxel_density, label


class SQCNN3D(nn.Module):
    def __init__(self, n_qubits, n_filters, grid_size, CU_layers=1, kernel_size=3, stride=1, n_classes=10,
                 rf_lambda=0.1):
        """
        Note: Input Should be of shape [batch_size, grid_size, grid_size, grid_size]

        Parameters:
            n_qubits : number of qubits per filter
            n_filters: number of filters
            CU_layers: number of layers of CU gates after data re-uploading
            kernel_size: size of kernel for data re-uploading (cubic kernel only)
            stride: stride with which kernel moves (same in all direction)
            n_classes: number of classes for classification
            rf_lambda: constant to multiply the RF_loss

        Output:
            logits: For cross entropy loss or augmax
            rf_loss: to add in cross entropy loss to introduce orthogonality in feature space for each filter
        """
        super(SQCNN3D, self).__init__()
        self.n_qubits = n_qubits
        self.n_filters = n_filters
        self.grid_size = grid_size
        self.CU_layers = CU_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_classes = n_classes
        self.rf_lambda = rf_lambda


        self.num_patches = ((grid_size - self.kernel_size) // self.stride + 1) ** 3

        # Quantum weights and parameters
        self.theta = nn.Parameter(0.01 * torch.randn(n_filters, CU_layers, n_qubits, 3))  # Parameters for PQC gates

        # Quantum device and QNode
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.device, interface="torch", diff_method="backprop")

        # Classical layer
        self.classical = nn.Linear(self.num_patches * n_filters * n_qubits, n_classes)

    def name(self):
        return f"SQCNN3D_N{self.n_qubits}_V{self.grid_size}_K{self.kernel_size}_S{self.stride}_CUL{self.CU_layers}_F{self.n_filters}_RFL{self.rf_lambda}_ModelNet{self.n_classes}"

    def _quantum_circuit(self, input_shape, theta):
        """
        Quantum circuit for the quanvolutional filter.
        Args:
            input_patch: Encoded feature patch.
            theta: Parameters for the PQC.
        """
        rotations = input_shape.shape[-1]
        if rotations%3 != 0:
            padding = 3 - (rotations % 3)
            input_shape = torch.nn.functional.pad(input_shape, (0, padding), "constant", 0)
        rotations = input_shape.shape[-1]

        # Encode input data using angle encoding
        # print(input_patch)
        w = -1
        for i in range(0, rotations, 3):
            # w = int(i / rotations * self.n_qubits)
            w = (w + 1) % self.n_qubits

            angle_1, angle_2, angle_3 = input_shape[:, i:i+3].T
            # print(input_shape[i])
            qml.Rot(angle_1, angle_2, angle_3, wires=w)

        # Apply parameterized quantum gates
        for layer in range(self.CU_layers):
            for i in range(self.n_qubits):
                qml.CRot(theta[layer, i, 0], theta[layer, i, 1], theta[layer, i, 2], wires=[i, (i + 1) % self.n_qubits])

        # Measure observable (Pauli-Z) for all qubits
        # return qml.state()
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)] + [qml.probs()]

    def forward(self, input_data):
        """
        Forward pass of the sQCNN-3D.
        Args:
            input_data: Voxelized point cloud input data.
        Returns:
            logits: Classification logits.
            loss_rf: Reverse fidelity loss.
        """
        batch_size = input_data.shape[0]
        grid_patches = self.extract_patches(input_data)

        quantum_outputs = []
        quantum_states = []

        for patch in grid_patches:
            quantum_outputs_patches = []
            quantum_states_patches = []

            for filter_idx in range(self.n_filters):
                    out = self.qnode(patch, self.theta[filter_idx])
                    quantum_filter_output, quantum_filter_state = torch.stack(out[:-1]).T, out[-1]

                    # Stack the filter_output to create the quantum outputs for this filter
                    quantum_outputs_patches.append(quantum_filter_output)
                    quantum_states_patches.append(quantum_filter_state)

            # Stack the filter_output to create the quantum outputs for this filter
            quantum_outputs.append(torch.stack(quantum_outputs_patches))
            quantum_states.append(torch.stack(quantum_states_patches))

        # Stack the outputs of all filters
        quantum_outputs = torch.stack(quantum_outputs, dim=1)
        quantum_outputs = quantum_outputs.transpose(0, 2).transpose(1, 2)  # Shape: (batch, n_filters, n_patches, n_qubits)

        # Compute Reverse Fidelity Loss (RF-Train)
        loss_rf = self.compute_rf_loss(quantum_states).to(torch.float32)

        # Flatten and pass through classical FCN
        quantum_features = quantum_outputs.reshape(batch_size, np.prod(quantum_outputs.shape[1:])).to(torch.float32)

        logits = self.classical(quantum_features)

        return logits, loss_rf

    def extract_patches(self, input_data):
        """
        Extract 3D patches from the input data.
        Args:
            input_data: Input voxelized data.
        Returns:
            List of patches to feed into the quanvolutional filters.
        """
        patches = []
        grid_size = input_data.shape[-1]
        size = int(np.cbrt(grid_size))
        for x in range(0, grid_size - self.kernel_size + 1, self.stride):
            for y in range(0, grid_size - self.kernel_size + 1, self.stride):
                for z in range(0, grid_size - self.kernel_size + 1, self.stride):
                    patch = input_data[:, x:x + self.kernel_size, y:y + self.kernel_size, z:z + self.kernel_size]
                    # print(patch.shape)
                    patches.append(patch.flatten(start_dim=1))  # Flatten for quantum encoding
        return patches

    def compute_rf_loss(self, quantum_states):
        """
        Compute the reverse fidelity loss to encourage diversity in features.
        Args:
            quantum_outputs: Outputs of the quantum circuits.
        Returns:
            loss_rf: Reverse fidelity loss.
        """
        loss_rf = 0
        for patch in quantum_states:
            for i in range(self.n_filters):
                for j in range(self.n_filters):
                    if i != j:
                        fidelity = self.calculate_fidelity(patch[i], patch[j])
                        loss_rf += torch.mean(fidelity)
        loss_rf /= (self.n_filters * (self.n_filters - 1))
        return loss_rf * self.rf_lambda

    def calculate_fidelity(self, state1, state2):
        """
        Calculate the fidelity between two quantum states.
        Fidelity: Ω(ψm, ψm') = |<ψm|ψm'>|^2
        """

        inner_product = torch.sum(state1 * state2, dim=-1)
        fidelity = torch.abs(inner_product) ** 2
        return fidelity



def fix_off_format(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check if the first line starts with 'OFF' and contains additional numbers
    if lines[0].startswith('OFF') and len(lines[0].strip()) > 3:
        # Separate 'OFF' from the rest of the first line
        header = 'OFF\n'
        count_line = lines[0][3:].strip() + '\n'

        # Replace the first line with the corrected version
        lines = [header, count_line] + lines[1:]

        # Write the fixed content to the output file, overwrite the original
        output_file = file_path
        with open(output_file, 'w') as f:
            f.writelines(lines)

        print(f"Fixed OFF file saved to {output_file}")


def data_parser(which_model_net='ModelNet10'):
    input_dir = os.path.join('./data', which_model_net)
    classes = {i: f_name for i, f_name in
               enumerate([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])}

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


def normalize_pcd(pcd, size=1):
    # Assuming you already have a point cloud object `pcd`
    vertices = np.asarray(pcd.points)

    # Step 1: Find the min and max coordinates along each axis
    min_coords = np.min(vertices)
    max_coords = np.max(vertices)

    # Step 2: Compute the range for each axis
    range_coords = max_coords - min_coords
    range_coords = 1e-6 if range_coords == 0 else range_coords

    # Step 3: Normalize the point cloud
    normalized_vertices = ((vertices - min_coords) / range_coords) * size

    # Update the point cloud with normalized coordinates
    pcd.points = o3d.utility.Vector3dVector(normalized_vertices)
    return pcd


def get_pcd_densities(pcd, voxel_size, binary_output=False):
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

    if binary_output:
        threshold = 0.0
        voxel_density = {k: 1 for k, v in voxel_density.items() if v > threshold}

    return voxel_density


def transform(file, voxel_size=8, visualize=False, binary_output=False):
    size = voxel_size
    pcd = o3d.io.read_triangle_mesh(file)
    pcd = pcd.sample_points_uniformly(number_of_points=5000)
    pcd = normalize_pcd(pcd, size)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    voxel_density = get_pcd_densities(pcd, voxel_size=voxel_grid.voxel_size, binary_output=binary_output)
    voxel_density_tensor = np.zeros((size, size, size), dtype=np.float32)
    for x in np.arange(size).astype(int):
        for y in np.arange(size).astype(int):
            for z in np.arange(size).astype(int):
                voxel_density_tensor[x, y, z] = voxel_density.get((x, y, z), 0)
    # voxel_density_tensor = (voxel_density_tensor != 0).astype(np.float64)
    if not binary_output:
        voxel_density_tensor /= np.linalg.norm(voxel_density_tensor)

    # Visualize the voxel grid
    if visualize:
        for i, voxel in enumerate(voxel_grid.get_voxels()):
            # voxel.color = 3 * (
            #             voxel_density.get(tuple(voxel.grid_index), 0) * np.ones((3,)) / max(voxel_density_tensor))
            voxel.color = 3 * (
                        voxel_density.get(tuple(voxel.grid_index), 0) * np.ones((3,)))
            voxel_grid.add_voxel(voxel)
        o3d.visualization.draw_geometries([voxel_grid])

    return voxel_density_tensor


# Function to calculate accuracy
def calculate_accuracy(history, num_classes, loader, model, name='Val'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)
    with torch.no_grad():  # Disable gradient computation for validation
        for state_vectors, labels in loader:
            outputs = model(state_vectors)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                total_per_class[label] += 1  # Increase total count for the true label
                if label == prediction:
                    correct_per_class[label] += 1  # Correct prediction for this class

    accuracy = correct / total
    print(f'{name:>5} accuracy: {accuracy:.4f}')
    history[name]['acc'].append(accuracy)

    per_class_accuracy = correct_per_class / total_per_class
    history[name]['per_class_acc'].append(per_class_accuracy)
    return accuracy


def load_datasets(which_model_net='ModelNet10', voxel_size=8, reduce=False, factor=1, binary_output=False):
    classes, train_files, train_labels, test_files, test_labels = data_parser(which_model_net=which_model_net)

    # Instantiate the dataset and dataloader
    train_dataset = ModelNetDataset(train_files, train_labels,
                                    transform=lambda x: transform(x, voxel_size=voxel_size,
                                                                  binary_output=binary_output))
    test_dataset = ModelNetDataset(test_files, test_labels,
                                   transform=lambda x: transform(x, voxel_size=voxel_size, binary_output=binary_output))

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


def accuracy_entropy(model, loader, device, n_classes, name='Val'):
    """
    Compute overall and per-class accuracy for a given DataLoader.

    Parameters:
        model (nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device (CPU/GPU) to use
        name (str): Name of the dataset (e.g., 'Val', 'Test').

    Returns:
        dict: Dictionary with 'acc' (overall accuracy) and 'per_class_acc' (list of per-class accuracies).
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(n_classes, device=device)
    class_total = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(inputs)

            # Find predicted labels
            _, predicted_labels = torch.max(outputs, dim=1)

            # Overall accuracy
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            # Per-class accuracy
            for i in range(n_classes):
                class_mask = labels == i
                class_correct[i] += (predicted_labels[class_mask] == labels[class_mask]).sum()
                class_total[i] += class_mask.sum()

    # Compute overall and per-class accuracy
    overall_acc = (total_correct / total_samples) * 100
    per_class_acc = [(class_correct[i] / class_total[i]).item() * 100 if class_total[i] > 0 else 0 for i in
                     range(n_classes)]
    print(f'{name:>5} accuracy: {overall_acc:.4f}')
    return {'acc': overall_acc, 'per_class_acc': per_class_acc}, overall_acc


if __name__ == '__main__':
    n_qubits = 4
    grid_size = 8
    n_filters = 2
    CU_layers = 2
    kernel_size = 4
    stride = 4

    # Load the train, val, and test dataset
    classes, train_dataset, val_dataset, test_dataset = load_datasets(which_model_net='ModelNet40',
                                                                      voxel_size=grid_size,
                                                                      reduce=False, binary_output=True)

    # Load model for training
    model = SQCNN3D(
        n_qubits=n_qubits,
        n_filters=n_filters,
        grid_size=grid_size,
        CU_layers=CU_layers,
        kernel_size=kernel_size,
        stride=stride,
        n_classes=len(classes),
        rf_lambda=0.1,
    )

    # Create csv to record data
    # ===================================================================================
    model_name = model.name()
    print(model_name)

    history = {
        'train_loss': [],
        'Train': {'acc': [], 'per_class_acc': []},
        'Val': {'acc': [], 'per_class_acc': []},
        'Test': {'acc': [], 'per_class_acc': []},
    }

    train_data_folder = f'results/training_data'
    weights_folder = f'results/weights'

    os.makedirs(train_data_folder, exist_ok=True)
    os.makedirs(weights_folder, exist_ok=True)

    csv_file_path = f'{train_data_folder}/{model_name}.csv'
    weights_path = f'{weights_folder}/{model_name}.pth'

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Acc', 'Test Acc'])
    # ===================================================================================

    # Training params
    num_epochs = 100
    batch_size = 32
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    total_params = sum(param.numel() for param in model.state_dict().values())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Total_number of parameters in the mode = {total_params}")
    print(f"Current device is {device}")

    # Create dataloaders for the new training and validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Send model to device
    model = model.to(device)

    # Save model weights if improved every epoch
    test_accuracy_previous = 0.0

    # Start Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, RF_loss = model(inputs)

            loss = loss_fn(outputs, labels) + RF_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            del inputs, labels

        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}')

        train_metrics, _ = accuracy_entropy(model, train_loader, device, n_classes=len(classes), name='Train')
        val_metrics, _ = accuracy_entropy(model, val_loader, device, n_classes=len(classes), name='Val')
        test_metrics, current_accuracy = accuracy_entropy(model, test_loader, device, n_classes=len(classes),
                                                          name='Test')

        # Update history
        for name, metrics in zip(['Train', 'Val', 'Test'], [train_metrics, val_metrics, test_metrics]):
            history[name]['acc'].append(metrics['acc'])
            history[name]['per_class_acc'].append(metrics['per_class_acc'])

        per_class_accuracy = history['Test']['per_class_acc'][-1]
        for i, acc in enumerate(per_class_accuracy):
            print(f'Acc. class {i:>2} {classes[i]:<12} -> {acc:.4f}')

        # Append data to the CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss,
                             history['Train']['acc'][-1],
                             history['Val']['acc'][-1],
                             history['Test']['acc'][-1]])

        # Save the best Model with highest test_accuracy
        if current_accuracy > test_accuracy_previous:
            test_accuracy_previous = current_accuracy
            torch.save(model.state_dict(), weights_path)

    plt.plot(history['Train']['acc'], label='Train')
    plt.plot(history['Val']['acc'], label='Val')
    plt.plot(history['Test']['acc'], label='Test')
    plt.legend()
    plt.show()
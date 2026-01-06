from torch.utils.data import Dataset


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
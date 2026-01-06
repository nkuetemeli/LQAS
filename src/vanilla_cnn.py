import torch.nn as nn
from pennylane import numpy as np
from torch import optim

from utils_datasets import *
from utils_models import *


class VoxelCNN(nn.Module):
    def __init__(self, params):
        super(VoxelCNN, self).__init__()
        self.params = params
        self.conv = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(4 * 2 * 2 * 2, params.n_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        size = int(np.cbrt(2**self.params.n_qubits))
        x = x.reshape(x.shape[0], 1, size, size, size)
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(-1, 4 * 2 * 2 * 2)
        x = self.fc(x)
        return x



def train(params, classes, train_loader, val_loader, test_loader):
    # Architecture search
    model = VoxelCNN(params).to(params.device)

    criterion = nn.CrossEntropyLoss()
    n_epochs = 20
    n_epochs_fine_tuning = 10

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.history = training_loop(params, model, criterion, optimizer, n_epochs,
                                          train_loader, val_loader, test_loader)

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Fine-tuning
    model.history = training_loop(params, model, criterion, optimizer, n_epochs_fine_tuning,
                                          train_loader, val_loader, test_loader, history=model.history)
    return model






if __name__ == '__main__':
    # Get params
    params = get_args()

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    # Train and save the model
    model = train(params, classes, train_loader, val_loader, test_loader)

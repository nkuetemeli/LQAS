import pennylane as qml
from pennylane import numpy as np
from torch import nn, optim

from utils_datasets import *
from utils_models import *


def candidate_layer(tape, id, prob, n_qubits, pruning_thresh=np.pi/10):

    # Deep copy the tape
    quantum_tape = []
    for operator in tape:
            operator_copy = operator.__class__(nn.Parameter(operator.parameters[0].clone().detach().data), wires=operator.wires)
            quantum_tape.append(operator_copy)

    # Modify quantum_tape
    n_qubits_coord = n_qubits // 3
    zero = lambda a: nn.Parameter(torch.tensor([0.]))
    # zero = lambda a: nn.Parameter(a * torch.randn(1))

    if id < 0:
        # Pruning
        for i, operator in enumerate(quantum_tape):
            normalized_theta = (operator.parameters[0] + np.pi) % (2 * np.pi) - np.pi
            if torch.rand(1) < prob and torch.abs(normalized_theta) < pruning_thresh:
                del quantum_tape[i]

    if id == 0:
        a = torch.sqrt(torch.tensor(1 / 1))
        for i in range(0, n_qubits):
            quantum_tape.append(qml.RX(zero(a), wires=[i]))

    if id == 1:
        a = torch.sqrt(torch.tensor(1 / 1))
        for i in range(0, n_qubits):
            quantum_tape.append(qml.RY(zero(a), wires=[i]))

    if id == 2:
        a = torch.sqrt(torch.tensor(1 / 1))
        for i in range(0, n_qubits):
            quantum_tape.append(qml.RZ(zero(a), wires=[i]))

    if id == 3:
        a = torch.sqrt(torch.tensor(1 / n_qubits_coord))
        for i in range(0, n_qubits, n_qubits_coord):
            for j in range(n_qubits_coord):
                quantum_tape.append(qml.CRX(zero(a), wires=[j + i, (j + 1) % n_qubits_coord + i]))

    if id == 4:
        a = torch.sqrt(torch.tensor(1 / n_qubits))
        for i in range(0, n_qubits):
            quantum_tape.append(qml.CRX(zero(a), wires=[i, (i + 1) % n_qubits]))

    if id == 5:
        a = torch.sqrt(torch.tensor(1 / n_qubits))
        for i in range(n_qubits -1, -1, -1):
            quantum_tape.append(qml.CRX(zero(a), wires=[i, (i + 1) % n_qubits]))

    if id == 6:
        a = torch.sqrt(torch.tensor(1 / (n_qubits**2 - n_qubits)))
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    quantum_tape.append(qml.CRX(zero(a), wires=[i, j]))

    if id == 7:
        a = torch.sqrt(torch.tensor(1 / (n_qubits**2 - n_qubits)))
        for i in range(n_qubits-1, -1, -1):
            for j in range(n_qubits):
                if i != j:
                    quantum_tape.append(qml.CRX(zero(a), wires=[i, j]))

    if id == 8:
        a = torch.sqrt(torch.tensor(1 / (n_qubits**2 - n_qubits)))
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    quantum_tape.append(qml.CRX(zero(a), wires=[j, i]))

    if id == 9:
        a = torch.sqrt(torch.tensor(1 / (n_qubits**2 - n_qubits)))
        for i in range(n_qubits-1, -1, -1):
            for j in range(n_qubits):
                if i != j:
                    quantum_tape.append(qml.CRX(zero(a), wires=[j, i]))

    return quantum_tape



class LayeredSuperModel(nn.Module):
    def __init__(self, params):
        super(LayeredSuperModel, self).__init__()
        self.params = params
        self.n_qubits = params.n_qubits
        self.n_classes = params.n_classes
        self.fc = nn.Linear(3 * params.n_qubits, params.n_classes)  # Move to device
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

        # Proper to candidate models
        self.quantum_weights = nn.ParameterList()
        self.quantum_tape = []
        self.candidate_architectures = []
        self.top_k_accuracies = []
        self.history = {
            'train_loss': [],
            'Train': {'acc': [], 'per_class_acc': []},
            'Val': {'acc': [], 'per_class_acc': []},
            'Test': {'acc': [], 'per_class_acc': []},
        }

        # Disable gradient updates for this layer
        if params.frozen_linear:
            for param in self.fc.parameters():
                param.data *= 10.
                param.requires_grad = False

    def sample_architecture(self, quantum_tape, id=None, prob=.5, pruning_thresh=np.pi/10):
        quantum_tape = candidate_layer(quantum_tape, id, prob, self.n_qubits, pruning_thresh)
        return quantum_tape

    def quantum_circuit(self, state_vector, quantum_tape):
        qml.StatePrep(state_vector, wires=range(self.n_qubits))

        circuit_from_tape(quantum_tape)

        expectation_vals = ([qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)] +
                            [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)] +
                            [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

        return expectation_vals

    def quantum_circuit_decorator(self, state_vector, quantum_tape):
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(state_vector):
            return self.quantum_circuit(state_vector, quantum_tape)

        return circuit(state_vector)

    def get_new_model(self, quantum_tape):
        candidate_model = CandidateModel(
            params=self.params,
            quantum_weights=nn.ParameterList([x.parameters[0] for x in quantum_tape]),
            quantum_tape=quantum_tape,
            history=self.history,
            candidate_architectures=self.candidate_architectures,
            top_k_accuracies = self.top_k_accuracies
        )
        return candidate_model

    def copy_fc_weights(self, model):
        self.fc.weight.data.copy_(model.fc.weight.data.clone())
        self.fc.bias.data.copy_(model.fc.bias.data.clone())
        return


class CandidateModel(LayeredSuperModel):
    def __init__(self, params, quantum_weights, quantum_tape, history, candidate_architectures, top_k_accuracies, p=.3):
        super(CandidateModel, self).__init__(params)
        self.quantum_weights = quantum_weights
        self.quantum_tape = quantum_tape
        self.history = history
        self.candidate_architectures = candidate_architectures
        self.top_k_accuracies = top_k_accuracies
        self.p = p

    def circuit(self, state_vector):
        return self.quantum_circuit_decorator(state_vector, self.quantum_tape)

    def forward(self, state_vectors):
        quantum_output = self.circuit(state_vectors)
        quantum_output = torch.vstack(quantum_output).to(torch.float32).T
        output = self.fc(quantum_output)
        return output





def train(params, classes, train_loader, val_loader, test_loader):

    # Architecture search
    super_model = LayeredSuperModel(params).to(params.device)
    n_generations = 10
    n_epochs = 5
    for generation in range(n_generations):
        print(f'\n------------\nGeneration {format(generation)}\n------------')
        super_tape = super_model.quantum_tape

        if generation % 3 == 0:
            ids = [0, 1, 2]
        elif generation % 3 == 1:
            ids = [-1, -1, -1]
        else:
            ids = (torch.randperm(10 - 3)[:3] + 3).tolist()

        candidate_tapes = []
        candidate_models = []
        candidate_histories = []
        tracks = []
        for i, id in enumerate(ids):
            print(f'\nArch {id}\n-------')
            candidate_tape = super_model.sample_architecture(super_tape, id=id)
            candidate_model = super_model.get_new_model(candidate_tape).to(params.device)
            candidate_model.copy_fc_weights(super_model)


            # Training loop
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(candidate_model.parameters(), lr=0.1)
            history = training_loop(params, candidate_model, criterion, optimizer, n_epochs, train_loader, val_loader, test_loader)

            candidate_tapes.append(candidate_tape)
            candidate_models.append(candidate_model)
            candidate_histories.append(history)
            tracks.append(max(history['Val']['acc']))


        # Update super_model
        best_record = torch.argmax(torch.tensor(tracks))

        # Update super_model
        super_model.quantum_weights = candidate_models[best_record].quantum_weights
        super_model.quantum_tape = candidate_tapes[best_record]
        super_model.candidate_architectures.append(ids[best_record])
        super_model.history = merge_nested_dicts(super_model.history, candidate_histories[best_record])
        super_model.copy_fc_weights(candidate_models[best_record])
        super_model.top_k_accuracies.append(tracks)
        print(f'Chosen architecture: {ids[best_record]}')

        if True:
            ##################################################################
            # Visualize the quantum circuit with specific input and parameters
            state_vector = train_loader.dataset.__getitem__(0)[0]
            vizualize_architechture(super_model, super_model.history, state_vector, super_model.quantum_tape, params.n_qubits)
            ##################################################################
    return


if __name__ == '__main__':
    # Get params
    params = get_args()

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    # Train and save the model
    train(params, classes, train_loader, val_loader, test_loader)

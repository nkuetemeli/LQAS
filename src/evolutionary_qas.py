import pickle
import random

import pennylane as qml
from torch import nn, optim

from utils_datasets import *
from utils_models import *


# Define the superset of candidate gates
def candidate_gates(n_qubits):
    return ([
                lambda wire, angle, i=i: qml.CRX(angle, wires=[wire, (wire + i)%n_qubits]) for i in range(1, n_qubits)  # Controlled-NOT
            ]+
            [
                lambda wire, angle: qml.RX(angle, wires=wire),  # RX rotation
                lambda wire, angle: qml.RY(angle, wires=wire),  # RY rotation
                lambda wire, angle: qml.RZ(angle, wires=wire),  # RZ rotation
                lambda wire, angle: qml.I(),  # Identity
            ])


class EvolutionarySuperModel(nn.Module):
    def __init__(self, params):
        super(EvolutionarySuperModel, self).__init__()
        self.params = params
        self.n_qubits = params.n_qubits
        self.n_layers = params.n_layers
        self.n_classes = params.n_classes
        self.fc = nn.Linear(3 * params.n_qubits, params.n_classes)  # Fully connected layer for classification
        self.activation = nn.ReLU()


        # Proper to candidate models
        self.quantum_weights = nn.Parameter(0.01 * torch.randn(len(candidate_gates(params.n_qubits)), self.n_qubits, self.n_layers))
        self.quantum_tape = []
        self.candidate_architectures = []
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

    def sample_architecture(self, candidate_architecture):

        quantum_tape = []
        for j in range(self.n_layers):
            for i in range(self.n_qubits):
                gate_idx = candidate_architecture[i, j]
                angle = self.quantum_weights[gate_idx, i, j]

                quantum_tape.append(candidate_gates(self.params.n_qubits)[gate_idx](i, angle))

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
            quantum_weights=self.quantum_weights,
            quantum_tape=quantum_tape,
            history=self.history,
            candidate_architectures=self.candidate_architectures,
        )
        return candidate_model

    def copy_fc_weights(self, model):
        self.fc.weight.data.copy_(model.fc.weight.data.clone())
        self.fc.bias.data.copy_(model.fc.bias.data.clone())
        return


class CandidateModel(EvolutionarySuperModel):
    def __init__(self, params, quantum_weights, quantum_tape, history, candidate_architectures, p=.0):
        super(CandidateModel, self).__init__(params)
        self.quantum_weights = quantum_weights
        self.quantum_tape = quantum_tape
        self.history = history
        self.candidate_architectures = candidate_architectures
        self.p = p

    def circuit(self, state_vector):
        return self.quantum_circuit_decorator(state_vector, self.quantum_tape)

    def forward(self, state_vectors):
        quantum_output = self.circuit(state_vectors)
        quantum_output = torch.vstack(quantum_output).to(torch.float32).T
        output = self.fc(quantum_output)
        return output


class EvolutionarySearch():
    def __init__(self, super_model, pop_size=10, k=5, n_generations=20):
        self.super_model = super_model
        self.pop_size = pop_size
        self.population = []
        self.accuracies = []
        self.top_k = []
        self.n = pop_size // 2
        self.m = pop_size // 2
        self.prob = .1
        self.k = k
        self.n_generations = n_generations
        self.top_k_accuracies = []


    def initialize_population(self, train_loader, val_loader, test_loader):
        accuracies = self.inference(self.super_model.candidate_architectures, train_loader, val_loader, test_loader)
        indices = torch.argsort(torch.tensor(accuracies), descending=True).tolist()[:self.pop_size]
        self.population = [self.super_model.candidate_architectures[i] for i in indices]
        self.accuracies = [accuracies[i] for i in indices]
        return

    def inference(self, population, train_loader, val_loader, test_loader, fine_tuning=False):
        accuracies = []
        for candidate_architecture in population:
            candidate_tape = self.super_model.sample_architecture(candidate_architecture)
            model = self.super_model.get_new_model(candidate_tape).to(self.super_model.params.device)
            model.copy_fc_weights(self.super_model)

            history = {
                'train_loss': [],
                'Train': {'acc': [], 'per_class_acc': []},
                'Val': {'acc': [], 'per_class_acc': []},
                'Test': {'acc': [], 'per_class_acc': []},
            }

            if fine_tuning:
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self.super_model.parameters(), lr=0.1)
                history = training_loop(self.super_model.params, model, criterion, optimizer, 1, train_loader, val_loader,
                                        test_loader)
                acc = history['Val']['acc'][-1]
            else:
                calculate_accuracy(self.super_model.params, history, val_loader, model, name='Val')
                acc = history['Val']['acc'][-1]
            accuracies.append(acc)
        return accuracies

    def crossover(self):
        architectures = []
        top_k_pop = [self.population[i] for i in self.top_k]
        for _ in range(self.n):
            arch1, arch2 = random.sample(top_k_pop, k=2)

            arch = torch.zeros_like(arch1)
            for i in range(arch.shape[0]):
                arch[i, :] = random.choice([arch1[i, :], arch2[i, :]])
            # for j in range(arch.shape[1]):
            #     arch[:, j] = random.choice([arch1[:, j], arch2[:, j]])
            architectures.append(arch)
        return architectures

    def mutate(self):
        architectures = []
        top_k_pop = [self.population[i] for i in self.top_k]
        candidate_architectures = random.sample(top_k_pop, k=self.m)

        for architecture in candidate_architectures:
            for i in range(architecture.shape[0]):
                for j in range(architecture.shape[1]):
                    if random.random() < self.prob:
                        architecture[i, j] = int(torch.randint(0, len(candidate_gates(self.super_model.params.n_qubits)), (1, 1)))

            architectures.append(architecture)
        return architectures

    def update_top_k(self):
        self.top_k = torch.argsort(torch.tensor(self.accuracies), descending=True).tolist()[:self.k]
        self.top_k_accuracies.append([self.accuracies[i] for i in self.top_k])
        print(self.top_k_accuracies[-1])
        return

    def evolve(self, train_loader, val_loader, test_loader, fine_tuning, filename):
        self.initialize_population(train_loader, val_loader, test_loader)
        self.update_top_k()

        for generation in range(self.n_generations):
            print(f'\n------------\nGeneration {format(generation)}\n------------')
            self.population = self.crossover() + self.mutate()
            self.accuracies = self.inference(self.population, train_loader, val_loader, test_loader, fine_tuning)
            self.update_top_k()

        # Save model
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        return



def train(params, classes, train_loader, val_loader, test_loader):
    # Architecture search
    super_model = EvolutionarySuperModel(params).to(params.device)

    n_rounds = 100
    n_epochs = 5
    for r in range(n_rounds):
        print(f'\n------------\nRound {format(r)}\n------------')
        candidate_architecture = torch.randint(0, len(candidate_gates(params.n_qubits)),
                                               (params.n_qubits, params.n_layers))

        candidate_tape = super_model.sample_architecture(candidate_architecture)
        candidate_model = super_model.get_new_model(candidate_tape).to(params.device)
        candidate_model.copy_fc_weights(super_model)

        # Training loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(candidate_model.parameters(), lr=0.1)
        history = training_loop(params, candidate_model, criterion, optimizer, n_epochs, train_loader, val_loader, test_loader)

        # Update super_model
        super_model.quantum_weights = candidate_model.quantum_weights
        super_model.quantum_tape = candidate_model.quantum_tape
        super_model.candidate_architectures.append(candidate_architecture)
        super_model.history = merge_nested_dicts(super_model.history, history)
        super_model.copy_fc_weights(candidate_model)

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

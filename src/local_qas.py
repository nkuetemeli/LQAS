import pickle
import random

import matplotlib.pyplot as plt

from QML.evolutionary_qas import *
from utils_models import *


class LocalSearch():
    def __init__(self, super_model, n_generations=20):
        self.super_model = super_model
        self.population = []
        self.accuracies = []
        self.top_k = []
        self.prob = .1
        self.k = 1
        self.n_generations = n_generations
        self.top_k_accuracies = []
        self.architecture = torch.randint(0, len(candidate_gates(super_model.params.n_qubits)),
                                               (super_model.params.n_qubits, params.n_layers))

    def inference(self, population, train_loader, val_loader, test_loader, fine_tuning=True):
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
                history = training_loop(params, model, criterion, optimizer, 1, train_loader, val_loader,
                                        test_loader)
                acc = history['Val']['acc'][-1]
            else:
                calculate_accuracy(self.super_model.params, history, val_loader, model, name='Val')
                acc = history['Val']['acc'][-1]
            accuracies.append(acc)
        return accuracies

    def mutate(self):
        architectures = []
        candidate_architectures = [self.architecture] * 10

        for architecture in candidate_architectures:
            for i in range(architecture.shape[0]):
                for j in range(architecture.shape[1]):
                    if random.random() < self.prob:
                        architecture[i, j] = int(torch.randint(0, len(candidate_gates(self.super_model.params.n_qubits)), (1, 1)))

            architectures.append(architecture)
        return architectures

    def update_top_k(self):
        top_k = torch.argsort(torch.tensor(self.accuracies), descending=True).tolist()[0]
        top_k_accuracy = self.accuracies[top_k]

        if len(self.top_k_accuracies) == 0 or top_k_accuracy > self.top_k_accuracies[-1]:
            self.architecture = self.population[top_k]
            self.top_k_accuracies.append(top_k_accuracy)
        else:
            self.top_k_accuracies.append(self.top_k_accuracies[-1])

        print(self.top_k_accuracies, self.accuracies)
        return

    def evolve(self, train_loader, val_loader, test_loader, filename):

        for generation in range(self.n_generations):
            print(f'\n------------\nGeneration {format(generation)}\n------------')
            self.population = self.mutate()
            self.accuracies = self.inference(self.population, train_loader, val_loader, test_loader)
            self.update_top_k()

        # Save model
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        return





if __name__ == '__main__':
    # Get params
    params = get_args()

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    # Filename to save the model
    filename = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '.sav'

    # # Train and save the model
    # train(params, filename, classes, train_loader, val_loader, test_loader)











    # Load model
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    # model = model.get_new_model(model.quantum_tape).to(params.device)
    # visualize_history(model, params, classes, train_loader, val_loader, test_loader)


    model.candidate_architectures = [torch.randint(0, len(candidate_gates(params.n_qubits)),
                                               (params.n_qubits, params.n_layers)) for _ in range(20)]


    evolution = LocalSearch(super_model=model)
    evolution.evolve(train_loader, val_loader, test_loader,
                     filename=f'ModelsFL'+str(params.frozen_linear)+'/localsearch_' + params.which_model_net + '_evolved_ftTrue.sav')


    for i, acc in enumerate(evolution.top_k_accuracies):
        plt.scatter(i, acc, c='g')
    plt.show()
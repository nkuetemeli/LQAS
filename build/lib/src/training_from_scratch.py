import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from QML.utils_datasets import *
from QML.utils_models import *
from QML.evolutionary_qas import *
from QML.layered_qas import *


# Random initialization of weights
def random_init(model):
    tape = []
    a = 0.01
    if isinstance(model.quantum_weights, nn.Parameter):
        num_paramameters = len([x.parameters[0] for x in model.quantum_tape if len(x.parameters) > 0])
        noisy_parameters = nn.Parameter(a * torch.randn(num_paramameters, device=model.params.device))
        counter = 0
        for operator in model.quantum_tape:
            if hasattr(operator, 'parameters') and len(operator.parameters) > 0:
                noisy_operator = operator.__class__(noisy_parameters[counter], wires=operator.wires)
                tape.append(noisy_operator)
                counter += 1
        model.quantum_tape = tape
        model.quantum_weights = noisy_parameters
    else:
        counter = 0
        for operator in model.quantum_tape:
            if hasattr(operator, 'parameters') and len(operator.parameters) > 0:
                noisy_operator = operator.__class__(nn.Parameter(nn.Parameter(a * torch.randn(1))), wires=operator.wires)
                tape.append(noisy_operator)
                counter += 1
        model.quantum_tape = tape
        model.quantum_weights = nn.ParameterList([x.parameters[0] for x in model.quantum_tape if len(x.parameters) > 0])

    model = model.get_new_model(model.quantum_tape).to(model.params.device)
    model.history = {
        'train_loss': [],
        'Train': {'acc': [], 'per_class_acc': []},
        'Val': {'acc': [], 'per_class_acc': []},
        'Test': {'acc': [], 'per_class_acc': []},
    }

    for param in model.fc.parameters():
        param.data = torch.randn(param.shape)
    return model


def train_from_scratch(params, filename, filename_trained_from_scratch):
    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    with open(filename, 'rb') as file:
        model_super = pickle.load(file)

    # Test models
    if 'evolution' in filename:
        best_architechture = model_super.population[model_super.top_k[0]]
        evolution_tape = model_super.super_model.sample_architecture(best_architechture)
        model = model_super.super_model.get_new_model(evolution_tape).to(params.device)
    else:
        model = model_super.get_new_model(model_super.quantum_tape).to(params.device)

    model = random_init(model)

    criterion = nn.CrossEntropyLoss()
    n_epochs = 20
    n_epochs_fine_tuning = 10

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    model.history = training_loop(params, model, criterion, optimizer, n_epochs,
                                          train_loader, val_loader, test_loader)

    optimizer = optim.Adam(model.parameters(), lr=0.03)  # Fine-tuning
    model.history = training_loop(params, model, criterion, optimizer, n_epochs_fine_tuning,
                                          train_loader, val_loader, test_loader, history=model.history)

    with open(filename_trained_from_scratch, 'wb') as file:
        pickle.dump(model, file)
    return


if __name__ == '__main__':
    params = get_args()


    evolution_file_ftTrue = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_ftTrue.sav'
    evolution_file_ftFalse = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_ftFalse.sav'
    layered_file = f'ModelsFL'+str(params.frozen_linear)+'/layered_' + params.which_model_net + '.sav'
    evolution_file_trained_from_scratch_ftTrue = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_trained_from_scratch_ftTrue.sav'
    evolution_file_trained_from_scratch_ftFalse = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_trained_from_scratch_ftFalse.sav'
    layered_file_trained_from_scratch = f'ModelsFL'+str(params.frozen_linear)+'/layered_' + params.which_model_net + '_trained_from_scratch.sav'


    # # # Train from scratch
    # train_from_scratch(params, evolution_file_ftTrue, evolution_file_trained_from_scratch_ftTrue)
    # train_from_scratch(params, evolution_file_ftFalse, evolution_file_trained_from_scratch_ftFalse)
    # train_from_scratch(params, layered_file, layered_file_trained_from_scratch)



    # Load models
    with open(evolution_file_trained_from_scratch_ftTrue, 'rb') as file:
        evolution_model_ftTrue = pickle.load(file)

    with open(evolution_file_trained_from_scratch_ftFalse, 'rb') as file:
        evolution_model_ftFalse = pickle.load(file)

    with open(layered_file_trained_from_scratch, 'rb') as file:
        layered_model = pickle.load(file)


    evolutionary_accuracies_ftTrue = [evolution_model_ftTrue.history[x]['acc'][-1] for x in ['Train', 'Val', 'Test']]
    evolutionary_accuracies_ftFalse = [evolution_model_ftFalse.history[x]['acc'][-1] for x in ['Train', 'Val', 'Test']]
    layered_accuracies = [layered_model.history[x]['acc'][-1] for x in ['Train', 'Val', 'Test']]

    evolutionary_color_ftTrue = 'y'
    evolutionary_color_ftFalse = 'g'
    layered_color = 'orange'

    plot_model_accuracies(accuracies=[layered_accuracies, evolutionary_accuracies_ftFalse, evolutionary_accuracies_ftTrue],
                          colors=[layered_color, evolutionary_color_ftFalse, evolutionary_color_ftTrue],
                          labels=['Layered', 'Evolutionary', 'Evolutionary ft'],
                          title='Training from scratch',
                          filename=f'ResultsFL{str(params.frozen_linear)}/benchmark_scratch{"_frozen_lin" if params.frozen_linear is True else""}'
                                   f'{"_m40" if "40" in params.which_model_net else ""}.pdf')

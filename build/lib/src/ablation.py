from torch import nn, optim
from utils_models import *
from utils_datasets import *
from QML.layered_qas import *
from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy



def train(params, filename, classes, train_loader, val_loader, test_loader, pruning_thresh=np.pi/10):

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
            candidate_tape = super_model.sample_architecture(super_tape, id=id, pruning_thresh=pruning_thresh)
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

    return super_model


if __name__ == '__main__':

    N_qubits = [6, 9, 12]
    pruning_threshs = [0] + [np.pi / k for k in [10, 4]]

    params = get_args()

    # Filename to save the model
    filename = f'ModelsFL'+str(params.frozen_linear)+'/ablation_' + params.which_model_net + '.sav'



    # models_pruning_threshs = []
    # models_N_qubits = []
    #
    # for pruning_thresh in pruning_threshs:
    #     params = get_args(n_qubits=9)
    #     classes, train_loader, val_loader, test_loader = get_data_loaders(params)
    #     model = train(params, '', classes, train_loader, val_loader, test_loader, pruning_thresh)
    #     models_pruning_threshs.append(model)
    #
    # for n_qubits in N_qubits:
    #     params = get_args(n_qubits=n_qubits)
    #     classes, train_loader, val_loader, test_loader = get_data_loaders(params)
    #     model = train(params, '', classes, train_loader, val_loader, test_loader, pruning_thresh=np.pi/10)
    #     models_N_qubits.append(model)
    #
    # # Save results
    # with open(filename, 'wb') as file:
    #     pickle.dump([models_pruning_threshs, models_N_qubits], file)





    # Load results
    with open(filename, 'rb') as file:
        models_pruning_threshs, models_N_qubits = pickle.load(file)


    colors = ['g', 'orange', 'y']
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(6, 2.4)
    for i, model in enumerate(models_pruning_threshs):
        acc = model.history['Val']['acc']
        label = r'$\pi$/' + str(int(np.pi/pruning_threshs[i])) if pruning_threshs[i] != 0 else 0
        ax[0].plot(acc, color=colors[i], label=rf't = {label}')
        ax[0].legend()
    for i, model in enumerate(models_N_qubits):
        acc = model.history['Val']['acc']
        ax[1].plot(acc, color=colors[i], label=f'k = {N_qubits[i] // 3}')
        ax[1].legend()
    ax[0].set_xlabel('Training epoch')
    ax[0].set_title('k = 3')
    ax[0].set_ylabel('Validation top 1 accuracy')
    ax[1].set_title(rf't = $\pi/${10}')
    ax[1].set_xlabel('Training epoch')
    plt.savefig(f'ResultsFL{str(params.frozen_linear)}/ablation.pdf', bbox_inches='tight')
    plt.show()
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from QML.utils_datasets import *
from QML.utils_models import *
from QML.evolutionary_qas import *
from QML.layered_qas import *
from QML.local_qas import *

def fine_tune(params, filename, filename_tuned):

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    with open(filename, 'rb') as file:
        model_super = pickle.load(file)


    # Fine-tune models
    if 'evolution' in filename:
        best_architechture = model_super.population[model_super.top_k[0]]
        evolution_tape = model_super.super_model.sample_architecture(best_architechture)
        model = model_super.super_model.get_new_model(evolution_tape).to(params.device)
        model.copy_fc_weights(model_super.super_model)
    elif 'local' in filename:
        best_architechture = model_super.architecture
        evolution_tape = model_super.super_model.sample_architecture(best_architechture)
        model = model_super.super_model.get_new_model(evolution_tape).to(params.device)
        model.copy_fc_weights(model_super.super_model)
    else:
        model = model_super.get_new_model(model_super.quantum_tape).to(params.device)
        model.copy_fc_weights(model_super)

    criterion = nn.CrossEntropyLoss()
    n_epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=0.03)
    model.history = training_loop(params, model, criterion, optimizer, n_epochs,
                                            train_loader, val_loader, test_loader)

    with open(filename_tuned, 'wb') as file:
        pickle.dump(model, file)
    return





if __name__ == '__main__':
    params = get_args()

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    # Load models
    evolution_file_ftTrue = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_ftTrue.sav'
    evolution_file_ftFalse = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_ftFalse.sav'
    layered_file = f'ModelsFL'+str(params.frozen_linear)+'/layered_' + params.which_model_net + '.sav'
    localsearch_file = f'ModelsFL'+str(params.frozen_linear)+'/localsearch_' + params.which_model_net + '_evolved_ftTrue.sav'
    evolution_file_fine_tuned_ftTrue = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_fine_tuned_ftTrue.sav'
    evolution_file_fine_tuned_ftFalse = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_fine_tuned_ftFalse.sav'
    layered_file_fine_tuned = f'ModelsFL'+str(params.frozen_linear)+'/layered_' + params.which_model_net + '_fine_tuned.sav'
    localsearch_file_fine_tuned = f'ModelsFL'+str(params.frozen_linear)+'/localsearch_' + params.which_model_net + '_evolved__fine_tuned_ftTrue.sav'

    # Fine-tune models
    # fine_tune(params, evolution_file_ftTrue, evolution_file_fine_tuned_ftTrue)
    # fine_tune(params, evolution_file_ftFalse, evolution_file_fine_tuned_ftFalse)
    # fine_tune(params, layered_file, layered_file_fine_tuned)
    # fine_tune(params, localsearch_file, localsearch_file_fine_tuned)

    # Load models
    with open(evolution_file_ftTrue, 'rb') as file:
        evolution_ftTrue = pickle.load(file)

    with open(evolution_file_ftFalse, 'rb') as file:
        evolution_ftFalse = pickle.load(file)

    with open(layered_file, 'rb') as file:
        layered = pickle.load(file)

    with open(localsearch_file, 'rb') as file:
        localsearch = pickle.load(file)

    with open(evolution_file_fine_tuned_ftTrue, 'rb') as file:
        evolution_model_ftTrue = pickle.load(file)

    with open(evolution_file_fine_tuned_ftFalse, 'rb') as file:
        evolution_model_ftFalse = pickle.load(file)

    with open(layered_file_fine_tuned, 'rb') as file:
        layered_model = pickle.load(file)

    with open(localsearch_file_fine_tuned, 'rb') as file:
        localsearch_model = pickle.load(file)

    evolutionary_color_ftTrue = 'y'
    evolutionary_color_ftFalse = 'g'
    layered_color = 'orange'
    local_color = 'r'

    # Compare models
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((5, 2.))
    k = 3

    for i in range(20):
        acc_evolved_ftTrue = evolution_ftTrue.top_k_accuracies[i][:k]
        acc_evolved_ftFalse = evolution_ftFalse.top_k_accuracies[i][:k]
        acc_layered = layered.top_k_accuracies[i][:k]
        acc_local = localsearch.top_k_accuracies[i]

        # Add legend entries only once
        if i == 0:
            ax.scatter([i] * k, acc_layered, c=layered_color, marker='*', label='Layered')
            ax.scatter([i] * k, acc_evolved_ftFalse, c=evolutionary_color_ftFalse, marker='o', label='Evolutionary')
            ax.scatter([i] * k, acc_evolved_ftTrue, c=evolutionary_color_ftTrue, marker='o', label='Evolutionary ft')
            ax.scatter(i, acc_local, c=local_color, marker='x', label='Local')
        else:
            ax.scatter([i] * k, acc_layered, c=layered_color, marker='*')
            ax.scatter([i] * k, acc_evolved_ftFalse, c=evolutionary_color_ftFalse, marker='o')
            ax.scatter([i] * k, acc_evolved_ftTrue, c=evolutionary_color_ftTrue, marker='o')
            ax.scatter(i, acc_local, c=local_color, marker='x')
    ax.set_xlabel('Evolution iteration')
    ax.set_ylabel('Validation top-1 accuracy')
    ax.set_xticks(np.arange(0, 20, 3))
    if "10" in params.which_model_net:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2)
        plt.xticks(np.arange(0, 20, 2), ['']*10)
        ax.set_xlabel('')
    # ax.set_ylim(bottom=0.5)
    plt.savefig(f'ResultsFL{str(params.frozen_linear)}/benchmark_evolution{"_frozen_lin" if params.frozen_linear is True else""}'
                f'{"_m40" if "40" in params.which_model_net else ""}.pdf', bbox_inches='tight')
    plt.show()

    evolutionary_accuracies_ftTrue = [evolution_model_ftTrue.history[x]['acc'][-1] for x in ['Train', 'Val', 'Test']]
    evolutionary_accuracies_ftFalse = [evolution_model_ftFalse.history[x]['acc'][-1] for x in ['Train', 'Val', 'Test']]
    layered_accuracies = [layered_model.history[x]['acc'][-1] for x in ['Train', 'Val', 'Test']]

    plot_model_accuracies(accuracies=[layered_accuracies, evolutionary_accuracies_ftFalse, evolutionary_accuracies_ftTrue],
                          colors=[layered_color, evolutionary_color_ftFalse, evolutionary_color_ftTrue],
                          labels=['Layered', 'Evolutionary', 'Evolutionary ft'],
                          title='Fine-tuning after search',
                          filename=f'ResultsFL{str(params.frozen_linear)}/benchmark_finetuning{"_frozen_lin" if params.frozen_linear is True else""}'
                                   f'{"_m40" if "40" in params.which_model_net else ""}.pdf')


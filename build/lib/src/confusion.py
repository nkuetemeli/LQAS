import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from QML.utils_datasets import *
from QML.utils_models import *
from QML.evolutionary_qas import *
from QML.layered_qas import *
from QML.local_qas import *
from QML.SQNN3D import *
import QML.SQNN3D as sqcnn
from QML.vanilla_cnn import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def eval(params, model, loader):
    # Move model to device
    model.to(params.device)

    gt_list = []
    pred_list = []

    model.eval()
    with torch.no_grad():  # Disable gradient computation for validation
        for state_vectors, labels in loader:
            state_vectors, labels = state_vectors.to(params.device), labels.to(params.device)
            try:
                outputs, _ = model(state_vectors)
            except:
                outputs = model(state_vectors)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Overall accuracy
            gt_list += labels.tolist()
            pred_list += predicted.tolist()
    return gt_list, pred_list

if __name__ == '__main__':
    params = get_args()

    # Get data
    classes, train_loader, val_loader, test_loader = get_data_loaders(params)

    # Load models
    evolution_file_fine_tuned_ftTrue = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_fine_tuned_ftTrue.sav'
    evolution_file_fine_tuned_ftFalse = f'ModelsFL'+str(params.frozen_linear)+'/evolutionary_' + params.which_model_net + '_evolved_fine_tuned_ftFalse.sav'
    layered_file_fine_tuned = f'ModelsFL'+str(params.frozen_linear)+'/layered_' + params.which_model_net + '_fine_tuned.sav'
    localsearch_file_fine_tuned = f'ModelsFL'+str(params.frozen_linear)+'/localsearch_' + params.which_model_net + '_evolved__fine_tuned_ftTrue.sav'
    vanillacnn_file_fine_tuned = f'ModelsFLFalse/vanillaCNN_' + params.which_model_net + '.sav'

    # Load models
    with open(evolution_file_fine_tuned_ftTrue, 'rb') as file:
        evolution_model_ftTrue = pickle.load(file)

    with open(evolution_file_fine_tuned_ftFalse, 'rb') as file:
        evolution_model_ftFalse = pickle.load(file)

    with open(layered_file_fine_tuned, 'rb') as file:
        layered_model = pickle.load(file)

    with open(localsearch_file_fine_tuned, 'rb') as file:
        localsearch_model = pickle.load(file)

    with open(vanillacnn_file_fine_tuned, 'rb') as file:
        vanillacnn_model = pickle.load(file)


    # Load model for training
    sqcnn_model = SQCNN3D(n_qubits=4, n_filters=2, grid_size=8, CU_layers=2, kernel_size=4, stride=4, n_classes=len(classes), rf_lambda=0.1,)
    weights_folder = f'results/weights'
    sqcnn_model.load_state_dict(torch.load(f'{weights_folder}/{sqcnn_model.name()}.pth'))




    # Load the train, val, and test dataset
    sqcnn_classes, sqcnn_train_dataset, sqcnn_val_dataset, sqcnn_test_dataset = sqcnn.load_datasets(which_model_net='ModelNet10',
                                                                      voxel_size=8,
                                                                      reduce=False, binary_output=True)

    # Create dataloaders for the new training and validation, and test sets
    sqcnn_train_loader = DataLoader(sqcnn_train_dataset, batch_size=64, shuffle=True)
    sqcnn_val_loader = DataLoader(sqcnn_val_dataset, batch_size=64, shuffle=False)
    sqcnn_test_loader = DataLoader(sqcnn_test_dataset, batch_size=64, shuffle=True)





    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
    fig.set_size_inches(4, 10)

    models = [layered_model, evolution_model_ftFalse, evolution_model_ftTrue, localsearch_model]
    names = ['Layered', 'Evolutionary', 'Evolutionary ft', 'Local search']

    models += [sqcnn_model] if params.frozen_linear else [vanillacnn_model]
    names += ['sQCNN-3D'] if params.frozen_linear else ['Vanila CNN']
    for i, (model, name) in enumerate(zip(models, names)):
        test_loader = sqcnn_test_loader if name =='sQCNN-3D' else test_loader
        gt_list, pred_list = eval(params, model, test_loader)
        conf_mat = confusion_matrix(np.array(gt_list), np.array(pred_list))
        ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                               display_labels=classes.values()).plot(ax=ax[i], xticks_rotation='vertical', cmap='Greens')
        ax[i].set_ylabel(f'{name}\nTrue label')
    plt.savefig(
        f'ResultsFL{str(params.frozen_linear)}/confusion{"_frozen_lin" if params.frozen_linear is True else ""}.pdf',
        bbox_inches='tight')
    plt.show()
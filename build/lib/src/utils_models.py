import pennylane as qml
from QML.Alt.utils import *
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


def circuit_from_tape(quantum_tape):
    for op in quantum_tape:
        if not isinstance(op, qml.I):
            qml.apply(op)
    return

def training_loop(params, model, criterion, optimizer, n_epochs, train_loader, val_loader, test_loader, history=None):
    # Move model to device
    model.to(params.device)
    history = {
        'train_loss': [],
        'Train': {'acc': [], 'per_class_acc': []},
        'Val': {'acc': [], 'per_class_acc': []},
        'Test': {'acc': [], 'per_class_acc': []},
    } if history is None else history

    for epoch in range(n_epochs):
        for state_vectors, labels in train_loader:
            # Move data to device
            state_vectors, labels = state_vectors.to(params.device), labels.to(params.device)

            optimizer.zero_grad()
            outputs = model(state_vectors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}')
        calculate_accuracy(params, history, train_loader, model, name='Train')
        calculate_accuracy(params, history, val_loader, model, name='Val')
        calculate_accuracy(params, history, test_loader, model, name='Test')

    return history


def merge_nested_dicts(dict1, dict2):
    merged_dict = {}

    # Copy 'train_loss' from both dictionaries
    merged_dict['train_loss'] = dict1.get('train_loss', []) + dict2.get('train_loss', [])

    # Merge 'Train', 'Val', 'Test' keys
    for key in ['Train', 'Val', 'Test']:
        merged_dict[key] = {}
        for sub_key in ['acc', 'per_class_acc']:
            merged_dict[key][sub_key] = dict1.get(key, {}).get(sub_key, []) + dict2.get(key, {}).get(sub_key, [])

    return merged_dict


def vizualize_architechture(model, history, state_vector, quantum_tape, n_qubits):
    ##################################################################
    # Visualize the quantum circuit with specific input and parameters
    # Create a QNode specifically for visualization
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def visualized_circuit():
        return model.quantum_circuit(state_vector, quantum_tape)

    fig, ax = qml.draw_mpl(visualized_circuit)()
    fig.suptitle(f"Model accuracy: Train: {round(history['Train']['acc'][-1], 2)}, "
                 f"Val: {round(history['Val']['acc'][-1], 2)}, "
                 f"Test: {round(history['Test']['acc'][-1], 2)}")
    plt.show()
    ##################################################################
    return

# Function to calculate accuracy
def calculate_accuracy(params, history, loader, model, name='Val'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    correct_per_class = np.zeros(params.n_classes)
    total_per_class = np.zeros(params.n_classes)
    with torch.no_grad():  # Disable gradient computation for validation
        for state_vectors, labels in loader:
            state_vectors, labels = state_vectors.to(params.device), labels.to(params.device)
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


# Plot losses and accuracy
def visualize_history(model, params, classes, train_loader, val_loader, test_loader):
    plt.plot(model.history['Train']['acc'], label='Train')
    plt.plot(model.history['Val']['acc'], label='Val')
    plt.plot(model.history['Test']['acc'], label='Test')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    plt.plot(model.history['train_loss'], label='Training loss')
    plt.legend()
    plt.show()

    print("\nTraining complete. \n--- \nAccuracy per classes")
    per_class_accuracy = model.history['Train']['per_class_acc'][-1]
    for i, acc in enumerate(per_class_accuracy):
        print(f'Acc. class {i:>2} {classes[i]:<12} -> {acc:.4f}')

    # Inspect some wrong classification
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for validation
        for state_vectors, labels in train_loader:
            state_vectors, labels = state_vectors.to(params.device), labels.to(params.device)

            outputs = model(state_vectors)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                if label != prediction and label == 7:
                    print(classes[label.item()], '->', classes[prediction.item()])
    return


def plot_model_accuracies(accuracies, colors, labels, title='', filename='', power=1):
    categories = ['Train', 'Validation', 'Test']
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    for acc, color, label in zip(accuracies, colors, labels):
        # Apply power transformation to accuracy values
        transformed_acc = [x**power for x in acc]
        transformed_acc += transformed_acc[:1]

        ax.fill(angles, transformed_acc, color=color, alpha=0.25)
        ax.plot(angles, transformed_acc, color=color, linewidth=2, label=label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add radial gridlines and labels
    yticks = np.linspace(0, 1, num=6)
    yticks = yticks[2:] if power != 1 else yticks
    ax.set_yticks(yticks**power)
    ax.set_yticklabels([f'{ytick:.1f}' for ytick in yticks])

    ax.set_title(title, y=1.05)
    plt.legend(loc='upper right')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    return

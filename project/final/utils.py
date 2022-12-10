import numpy as np
import torch



def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        xs: Numpy array of x-values (inputs).
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    return inputs


def load_labels(csv_path):
    """Load labels from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    y_cols = [i for i in range(len(headers)) if headers[i].startswith('y')]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=y_cols)

    return inputs


def evaluate(outputs, labels, name):
    total = outputs.shape[0]
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu()
    correct = (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} {name} images: {100 * correct / total}%')
    return correct


def analyze_class_accuracy(outputs, labels, k):
    total = outputs.shape[0]
    class_acc = [[0 for i in range(2)] for j in range(30)]
    _, predicted_topk = torch.topk(outputs.data, k, 1)
    for i in range(total):
        class_acc[labels[i]][1] += 1
        if labels[i] in predicted_topk[i]:
            class_acc[labels[i]][0] += 1

    print(class_acc)
    print("Class, #Correct, Total, Accuracy(%)")
    for j in range(30):
        print(j, class_acc[j][0], class_acc[j][1], 100 * class_acc[j][0] / class_acc[j][1])
    return class_acc


import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


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
    class_acc = [[0 for i in range(2)] for j in range(31)]
    class_acc[30][1] = total
    _, predicted_topk = torch.topk(outputs.data, k, 1)
    for i in range(total):
        class_acc[labels[i]][1] += 1
        if labels[i] in predicted_topk[i]:
            class_acc[labels[i]][0] += 1
            class_acc[30][0] += 1

    final = []
    print("Class, #Correct, Total, Accuracy(%)")
    for j in range(31):
        print(j, class_acc[j][0], class_acc[j][1], 100 * class_acc[j][0] / class_acc[j][1])
        final.append(100 * class_acc[j][0] / class_acc[j][1])
    return class_acc, final


def get_confusion_matrix(output, y_true, filename):
    _, y_pred = torch.max(output.data, 1)
    cf_matrix = confusion_matrix(y_true, y_pred)
    # cf_matrix / np.sum(cf_matrix) * 10
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(30)],
                         columns=[i for i in range(30)])

    plt.figure(figsize=(15, 9))
    sn.heatmap(df_cm, annot=False)
    plt.savefig(filename)
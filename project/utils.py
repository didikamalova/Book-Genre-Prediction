from PIL import Image
import pandas as pd
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


def get_outputs_csv(model, csv_path, output_path):
    device = torch.device('mps')
    outputs = []
    with torch.no_grad():
        df = pd.read_csv(csv_path)
        convert_tensor = torchvision.transforms.ToTensor()
        for index, row in tqdm(df.iterrows()):
            filename = row['filename']
            image = Image.open("224x224/" + filename.strip())
            image = convert_tensor(image)
            image = image.to(device=device).to(dtype=torch.float32)
            image = image.unsqueeze(0)
            output = model(image)
            softmax = torch.nn.Softmax(dim = 0)
            output = softmax(output)
            output = output.tolist()
            outputs.append(output)
    outputs_np = np.asarray(outputs)
    np.savetxt(output_path, outputs_np, delimiter=',')

# NOTE: takes ~2 min to fully load data, find a way to make this faster or so we don't have to do it everytime
def load_data(csv_path):
    images = []
    labels = []

    df = pd.read_csv(csv_path)

    # images = [Image.open("224x224/" + filename.strip()) for filename in df['filename']]
    for filename in df['filename']:
        img = Image.open("224x224/" + filename.strip())
        img.load()
        images.append(img)

    labels = [label for label in df['label']]

    return images, labels


def get_labels(model, data_loader):
    device = torch.device('mps')
    all_labels = torch.Tensor()
    all_predicted = torch.Tensor()

    model.to(device=device)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device=device).to(dtype=torch.float32)
            labels = labels.type(torch.LongTensor)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            all_labels = torch.concat((all_labels, labels))
            all_predicted = torch.concat((all_predicted, predicted))

    all_predicted = all_predicted.to(dtype=torch.int8)
    all_labels = all_labels.to(dtype=torch.int8)
    return all_predicted, all_labels


def get_confusion_matrix(model, data_loader):
    y_pred, y_true = get_labels(model, data_loader)
    cf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in range(20)],
                         columns=[i for i in range(20)])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')



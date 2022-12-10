from PIL import Image
import pandas as pd
import torchvision
import torch
import numpy as np
from tqdm import tqdm


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

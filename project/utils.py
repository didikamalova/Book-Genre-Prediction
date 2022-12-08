from PIL import Image
import pandas as pd


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

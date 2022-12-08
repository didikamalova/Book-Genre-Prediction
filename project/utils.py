from PIL import Image
import pandas as pd


def load_data(csv_path):
    images = []
    labels = []

    df = pd.read_csv(csv_path)

    images = [Image.open(filename) for filename in df['filename']]
    labels = [label for label in df['label']]

    return images, labels
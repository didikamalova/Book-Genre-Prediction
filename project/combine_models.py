import torch
import torchvision
import evaluate
import utils
from ImageDataset import ImageDataset
from cnn import Model


def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    return model


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

# For M1 Macs
device = torch.device('mps')
img_model = load_model('book_covers.pth')
# TRAINING & VAL DATASET
test_dir = 'bookcover30-labels-train.csv'
# NOTE: takes ~2 min to fully load data, find a way to make this faster or so we don't have to do it everytime
test_set = ImageDataset(test_dir, transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)




print(img_model(test_set[0]))

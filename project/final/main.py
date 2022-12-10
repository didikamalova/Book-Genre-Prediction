import numpy as np
import torch
from utils import load_labels, load_dataset, evaluate
from combined_models import OneParam, ThirtyParam


train_image_dir = 'prelim_outputs/train_image_output.csv'
train_title_dir = 'prelim_outputs/train_title_output.csv'
train_y_dir = 'prelim_outputs/train_labels.csv'

val_image_dir = 'prelim_outputs/train_image_output.csv'
val_title_dir = 'prelim_outputs/train_title_output.csv'
val_y_dir = 'prelim_outputs/train_labels.csv'


batch_size = 64
learning_rate = 0.001
num_epochs = 1000


model = ThirtyParam()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

train_x1 = torch.from_numpy(load_dataset(train_image_dir))
train_x2 = torch.from_numpy(load_dataset(train_title_dir))
train_y = torch.from_numpy(load_labels(train_y_dir)).type(torch.LongTensor)

val_x1 = torch.from_numpy(load_dataset(val_image_dir))
val_x2 = torch.from_numpy(load_dataset(val_title_dir))
val_y = torch.from_numpy(load_labels(val_y_dir)).type(torch.LongTensor)



print("=" * 50)
print("Initial Evaluation: ")
print("image only:")
evaluate(train_x1, train_y, "train")
print("title only:")
evaluate(train_x2, train_y, "train")
print("")
outputs = model(train_x1, train_x2)
evaluate(outputs, train_y, "train")

for ix, epoch in enumerate(range(num_epochs), start=1):
    optimizer.zero_grad()
    outputs = model(train_x1, train_x2)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    evaluate(outputs, train_y, "train")
    print("=" * 50)


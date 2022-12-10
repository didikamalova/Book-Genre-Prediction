import numpy as np
import torch
from utils import load_labels, load_dataset, evaluate
from combined_models import OneParam, ThirtyParam, FullyConnected


train_image_dir = 'prelim_outputs/train_image_output.csv'
train_title_dir = 'prelim_outputs/train_title_output.csv'
train_y_dir = 'prelim_outputs/train_labels.csv'

test_image_dir = 'final_outputs/test_image_output.csv'
test_title_dir = 'final_outputs/test_title_output.csv'

num_epochs = 1000

model = ThirtyParam()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


train_x1 = torch.from_numpy(load_dataset(train_image_dir))
train_x2 = torch.from_numpy(load_dataset(train_title_dir))
train_y = torch.from_numpy(load_labels(train_y_dir)).type(torch.LongTensor)


print("=" * 50)
print("Initial Evaluation: ")
outputs = model(train_x1, train_x2)
evaluate(outputs, train_y, "train")
evaluate(train_x1, train_y, "train")
evaluate(train_x2, train_y, "train")

for ix, epoch in enumerate(range(num_epochs), start=1):
    optimizer.zero_grad()
    outputs = model(train_x1, train_x2)
    loss = criterion(outputs, train_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    evaluate(outputs, train_y, "train")
    print("=" * 50)


PATH = './saved_comb_models/comb2.pth'
torch.save(model.state_dict(), PATH)

test_x1 = torch.from_numpy(load_dataset(test_image_dir))
test_x2 = torch.from_numpy(load_dataset(test_title_dir))

outputs = model(test_x1, test_x2)
# np.savetxt('./final_outputs/test_comb2_output.csv', outputs.detach().numpy(), delimiter=',')

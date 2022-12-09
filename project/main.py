import time
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn import Model
from ImageDataset import ImageDataset
from evaluate import evaluate, evaluate2


if __name__ == "__main__":
    np.random.seed(229)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # For M1 Macs
    device = torch.device('mps')

    # TRAINING & VAL DATASET
    train_val_data_dir = 'bookcover30-labels-train.csv'
    # NOTE: takes ~2 min to fully load data, find a way to make this faster or so we don't have to do it everytime
    train_val_set = ImageDataset(train_val_data_dir, transform)

    val_ratio = 1/9
    throw_ratio = 1/10
    throwaway, train_set, val_set = \
        torch.utils.data.random_split(train_val_set, [(1-val_ratio)*throw_ratio, (1-val_ratio)*(1-throw_ratio), val_ratio],
                                      generator=torch.Generator().manual_seed(229))

    print(len(train_set), len(val_set))

    # HYPERPARAMS
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 200
    reg_lambda = 1e-5

    # AUGMENTATION
    aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
    ])

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # TRAINING
    model = Model().to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=reg_lambda)

    start_time = time.time()

    best_model = model
    highest = 0

    print("=" * 50)
    print("Initial Evaluation: ")
    evaluate(model, train_loader, device, name="train")
    evaluate(model, val_loader, device, name="val")
    print("=" * 50)
    for ix, epoch in enumerate(range(num_epochs), start=1):  # loop over the dataset multiple times
        # TRAINING
        print(f'EPOCH {ix}')
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = aug(inputs.to(device=device).to(dtype=torch.float32))
            labels = labels.type(torch.LongTensor).to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            running_loss += loss.item()

        train_accuracy = evaluate(model, train_loader, device, name="train")
        train_loss = running_loss

        # VALIDATION
        with torch.no_grad():
            val_accuracy = evaluate(model, val_loader, device, name="val")
            if val_accuracy > highest:
                highest = val_accuracy
                best_model = model
        print("=" * 50)

    end_time = time.time()
    print(f"Total training time: {end_time - start_time} sec")

    PATH = './book_covers.pth'
    torch.save(model.state_dict(), PATH)

    # PREDICT
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    model = Model().to(device=device)
    model.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # Similar to the previous question, calculate model's output and the percentage as correct / total
            ### YOUR CODE HERE
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ### END YOUR CODE

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    
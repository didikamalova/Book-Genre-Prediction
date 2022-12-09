import time
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn import Model
from ImageDataset import ImageDataset
from evaluate import evaluate


if __name__ == "__main__":
    np.random.seed(229)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # For M1 Macs
    device = torch.device('mps')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(x)
    else:
        print("MPS device not found.")

    # TRAINING & VAL DATASET
    train_val_data_dir = 'bookcover30-labels-train.csv'
    # NOTE: takes ~2 min to fully load data, find a way to make this faster or so we don't have to do it everytime
    train_val_set = ImageDataset(train_val_data_dir, transform)

    val_ratio = 1/9
    val_len = int(len(train_val_set)*val_ratio)
    train_len = len(train_val_set) - val_len
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_len, val_len],
                                                       generator=torch.Generator().manual_seed(229))

    # TESTING DATASET
    test_data_dir = 'bookcover30-labels-test.csv'
    test_set = ImageDataset(test_data_dir, transform)

    print(len(train_set), len(val_set), len(test_set))

    # HYPERPARAMS
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 200

    reg = "l2"
    lambda_l1 = 0.00001
    lambda_l2 = 0.0001

    # AUGMENTATION
    aug = torchvision.transforms.Compose([
          torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # TRAINING
    model = Model().to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

            if (reg == 'l1'):
                l1 = 0
                for p in model.parameters():
                    l1 += p.abs().sum()
                    loss += lambda_l1 * l1
            elif (reg == 'l2'):
                l2 = 0
                for p in model.parameters():
                    l2 += p.pow(2.0).sum()
                    loss += lambda_l2 * l2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            running_loss += loss.item()

            # print statistics
            """
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            """
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

    
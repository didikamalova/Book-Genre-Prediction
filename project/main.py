import time
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn import Model
from ImageDataset import ImageDataset


if __name__ == "__main__":
    np.random.seed(229)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    device = torch.device('cuda')

    # TRAINING DATASET
    train_data_dir = 'bookcover30-labels-train.csv'
    train_set = ImageDataset(train_data_dir, transform)

    ix = 0
    print(train_set.get_data(ix))

    # TESTING DATASET
    test_data_dir = 'bookcover30-labels-test.csv'
    test_set = ImageDataset(test_data_dir, transform)

    # HYPERPARAMS
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 200

    # AUGMENTATION
    aug = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # TRAINING
    model = Model().to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    start_time = time.time()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

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

    
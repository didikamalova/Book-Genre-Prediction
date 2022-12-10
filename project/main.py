import time
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
from tqdm import tqdm
from cnn import Model, FocalLoss
from ImageDataset import ImageDataset
from evaluate import evaluate, get_labels


if __name__ == "__main__":
    train = False

    np.random.seed(229)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    torch.backends.cudnn.benchmark = True

    # For M1 Macs
    device = torch.device('mps')

    # # TRAINING & VAL DATASET
    # train_val_data_dir = 'bookcover30-labels-train.csv'
    # # NOTE: takes ~2 min to fully load data, find a way to make this faster or so we don't have to do it everytime
    # train_val_set = ImageDataset(train_val_data_dir, transform)

    # val_ratio = 1/9
    # val_len = int(len(train_val_set)*val_ratio)
    # train_len = len(train_val_set) - val_len
    # train_set, val_set = torch.utils.data.random_split(train_val_set, [train_len, val_len],
    #                                                    generator=torch.Generator().manual_seed(229))

    # TRAINING & VAL DATASET
    train_val_data_dir = 'bookcover30-labels-train.csv'
    # NOTE: takes ~2 min to fully load data, find a way to make this faster or so we don't have to do it everytime
    train_val_set = ImageDataset(train_val_data_dir, transform)

    val_ratio = 1/9
    throw_ratio = 2/3
    throwaway, train_set, val_set = \
        torch.utils.data.random_split(train_val_set, [(1-val_ratio)*throw_ratio, (1-val_ratio)*(1-throw_ratio), val_ratio],
                                    generator=torch.Generator().manual_seed(229))

    # TESTING DATASET
    test_data_dir = 'bookcover30-labels-test.csv'
    test_set = ImageDataset(test_data_dir, transform)

    print(len(train_set), len(val_set), len(test_set))

    # HYPERPARAMS
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    reg_lambda = 1e-6

    # AUGMENTATION
    aug = torchvision.transforms.Compose([
          torchvision.transforms.RandomHorizontalFlip(p=0.5),
          torchvision.transforms.RandomVerticalFlip(p=0.5)
    ])

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # # Display some images from all_data
    # figure = plt.figure(figsize=(15, 10))
    # num_rows = 1
    # num_cols = 3

    # ds_idx = [13, 29, 43]
    # for plot_idx, idx in enumerate(ds_idx):
    #     ax = plt.subplot(num_rows, num_cols, plot_idx + 1) # subplot indices begin at 1, not 0
    #     ax.title.set_text(train_val_set.get_class(train_val_set.get_label(idx)))
    #     plt.axis('off')
    #     plt.imshow(train_val_set.get_image(idx))
    # plt.show()

    # TRAINING
    if train:
        model = Model().to(device=device)
        criterion = FocalLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
            for i, (inputs, labels) in enumerate(tqdm(train_loader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs = aug(inputs.to(device=device).to(dtype=torch.float32))
                labels = labels.type(torch.LongTensor).to(device=device)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
    # torch.save(model.state_dict(), PATH)

    # PREDICT
    model = Model().to(device=device)
    model.load_state_dict(torch.load(PATH))

    # # OUTPUT
    # train_outputs = []
    # with torch.no_grad():
    #     df = pd.read_csv(train_val_data_dir)
    #     convert_tensor = torchvision.transforms.ToTensor()
    #     i = 0
    #     for index, row in tqdm(df.iterrows()):
    #         filename = row['filename']
    #         image = Image.open("224x224/" + filename.strip())
    #         image = convert_tensor(image)
    #         image = image.to(device=device).to(dtype=torch.float32)
    #         image = image.unsqueeze(0)
    #         output = model(image)
    #         softmax = torch.nn.Softmax(dim = 1)
    #         output = softmax(output)
    #         output = output.tolist()[0]
    #         train_outputs.append(output)
    # train_outputs_np = np.asarray(train_outputs)
    # np.savetxt('train_outputs.csv', train_outputs_np, delimiter=',')

    evaluate(model, test_loader, device, name='test')

    y_pred, y_true = get_labels(model, test_loader)
    cf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in range(30)],
                        columns = [i for i in range(30)])

    plt.figure(figsize = (24,14))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')

    
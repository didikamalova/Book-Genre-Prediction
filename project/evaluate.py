import torch


def evaluate(model, data_loader, device, name="val"):
    correct = 0  # number of correct predictions
    total = 0    # total number of examples in the data loader

    model.to(device=device)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device=device).to(dtype=torch.float32)
            labels = labels.type(torch.LongTensor)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} {name} images: {100 * correct / total}%')
    return correct


def evaluate2(model, data_set, device, name):
    correct = 0  # number of correct predictions
    total = 0    # total number of examples in the data loader

    if type(data_set) == torch.utils.data.dataset.Subset:
        images = data_set.dataset.get_all_images().to(device=device)
        labels = data_set.dataset.get_all_labels().to(device=device)
    elif type(data_set) == torch.utils.data.dataset:
        images = data_set.get_all_images().to(device=device)
        labels = data_set.get_all_labels().to(device=device)
    else:
        raise AssertionError("data_set is not expected type")

    model.to(device=device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu()
    total += len(data_set)
    correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} {name} images: {100 * correct / total}%')

    return correct

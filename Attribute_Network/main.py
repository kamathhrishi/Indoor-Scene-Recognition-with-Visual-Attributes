import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch
import time
from load_data import load_dataset

IMG_SIZE = 224


def evaluate(model, loader, label):

    model.eval()

    total = 0.0
    intensity_error = 0

    for data, target in loader:

        outputs = model(data.view([-1, 3, IMG_SIZE, IMG_SIZE]).double())
        loss = nn.BCEWithLogitsLoss()
        intensity_error += loss(outputs, target.double()).item()
        total += 1

    print("Error: ", float(intensity_error) / total)
    return float(intensity_error) / total


def save_model(model, optimizer, name="model.pth"):

    checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, name)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.eval()
    return model


def train(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader=None,
    epochs=100,
    name="model.pth",
    best_model="best_model.pth",
):

    model.train()
    best_accuracy = 1000000.0

    for epoch in range(epochs):  # loop over the dataset multiple times

        start = time.time()
        running_loss = 0.0
        index = 0

        for data, target in train_loader:

            optimizer.zero_grad()
            outputs = model(data.view([-1, 3, IMG_SIZE, IMG_SIZE]).double())
            loss1 = criterion(outputs, target.double())
            loss = loss1
            loss.backward()
            running_loss += loss
            optimizer.step()
            index += 1
            # print(loss)
            # print(index)
            # print(loss1)
            # print(loss2)

        print("\n")
        print("Epoch: ", epoch)
        print("Loss: ", running_loss / len(train_loader))
        test_accuracy = evaluate(model, test_loader, "Test Accuracy: ")

        if test_accuracy <= best_accuracy:

            save_model(model, optimizer, name=name)
            best_accuracy = test_accuracy

        print("Best accuracy: ", best_accuracy)

        # end=time.time()
        # evaluate(model,train_loader,"Training Accuracy: ")

    return loss


net = models.alexnet(pretrained=True)
net.classifier[6] = nn.Linear(4096, 102)

train_dataloader, test_dataloader = load_dataset()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)
train(net.double(), criterion, optimizer, train_dataloader, test_dataloader, epochs=200)

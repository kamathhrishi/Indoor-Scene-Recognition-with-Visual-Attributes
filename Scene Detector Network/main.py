# Import required libraries
import torch
import torch.optim as optim
import torch.nn as nn
import time
from load_prepare_data import load_dataset
from arguments import Arguments
from torchvision import transforms, datasets

args = Arguments()

torch.manual_seed(args.seed)


def save_model(model, optimizer, name="model.pth"):

    checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv9 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = nn.Linear(int(100352 / args.batch_size), 4000)
        self.fc2 = nn.Linear(4000, 4000)
        self.fc3 = nn.Linear(4102, 4102)
        self.fc4 = nn.Linear(4102, 4102)
        self.fc5 = nn.Linear(4102, 67)
        self.dropout = nn.Dropout(p=0.4)
        self.attribute_network = load_checkpoint(
            "./attribute_network/model.pth"
        ).eval()

    def forward(self, img, attr=None):

        x1 = self.pool(torch.relu(self.conv1(img)))
        x2 = self.pool(torch.relu(self.conv2(x1)))
        x3 = torch.relu(self.conv3(x2))
        x4 = torch.relu(self.conv4(x3) + x2)
        x5 = self.pool(torch.relu((self.conv5(x4))))
        x6 = torch.relu(self.conv6(x5))
        x7 = torch.relu(self.conv7(x6) + x5)
        x8 = torch.relu(self.conv8(x7) + x6)
        x9 = self.pool(torch.relu(self.conv9(x8)))
        x10 = torch.relu(self.conv10(x9))
        x11 = self.avgpool(torch.relu(self.conv11(x10) + x9))
        x = x11.view([-1, int(100352 / args.batch_size)])
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        attribute = self.attribute_network(img)

        x = self.fc3(torch.cat((x, attribute), axis=1))
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def evaluate(model, loader, label):

    model.eval()

    correct = 0.0
    total = 0.0

    for data, target in loader:

        outputs = model(data.double())
        correct += (torch.argmax(outputs, axis=1) == target).sum()
        total += args.batch_size
        # print(float(correct/total)*100)

    print(label, float(correct / total) * 100)
    return float(correct / total) * 100


def train(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs=100,
    model_name="best_model.pth",
):

    model.train()
    best_accuracy = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times

        start = time.time()
        running_loss = 0.0
        index = 0

        for data, target in train_loader:

            optimizer.zero_grad()
            outputs = model(data.double())
            loss = criterion(outputs, target)
            print(
                "Epoch ",
                epoch,
                " (Index ",
                str(index),
                "/",
                str(len(train_loader)),
                " Loss : ",
                loss.item(),
                ")",
            )
            running_loss += loss
            loss.backward()
            optimizer.step()
            # save_model(model,optimizer,name=name)
            index += 1
            # print(index)

        test_accuracy = evaluate(model, test_loader, "Test Accuracy: ")
        if test_accuracy >= best_accuracy:

            best_accuracy = test_accuracy
            print("Best accuracy: ", best_accuracy)
            save_model(model, optimizer, name=model_name)
            best_accuracy = test_accuracy

        end = time.time()
        # print("Epoch: ",epoch)
        print("Time for Evaluation: ", (end - start) / 60)
        print("Loss: ", running_loss / len(train_loader))
        # evaluate(model,train_loader,"Training Accuracy: ")

    return loss


train_loader, test_loader = load_dataset()

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
train(
    net.double(),
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs=1,
    model_name="model.pth",
)

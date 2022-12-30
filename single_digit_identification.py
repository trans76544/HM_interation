import torch
import torchvision
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(6*6*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        y = self.net1(x)
        y = torch.flatten(y, 1)
        y = nn.functional.relu(self.fc1(y))
        y = self.fc2(y)
        return y

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

Epoch = 20
train_batch = 200
test_batch = 200
lr=0.005
train_dataset = torchvision.datasets.MNIST("../../dataset/", train=True, download=False, transform=transforms)
test_dataset = torchvision.datasets.MNIST("../../dataset/", train=False, download=False, transform=transforms)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
Loss = nn.CrossEntropyLoss().to(device)
writer = SummaryWriter('G:/PythonProject/src/Human_machine_interaction/single_digit')

def train(model, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_dataloader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = Loss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print("Train Epoch:{} [{}/{}], Loss:{}".format(epoch, (batch_idx + 1) * train_batch, len(train_dataloader.dataset), train_loss / 50 / train_batch))
            writer.add_scalar('train_loss', train_loss / 50 / train_batch, (epoch - 1) * 6 + (batch_idx + 1) / 50)
            train_loss = 0

def test(model, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for datas, targets in test_dataloader:
            datas, targets = datas.to(device), targets.to(device)
            outputs = model(datas)
            test_loss += Loss(outputs, targets).item()
            preds = outputs.max(1, keepdim=True)[1]
            correct += preds.eq(targets.view_as(preds)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    correct /= len(test_dataloader.dataset)
    print("Test Epoch:{}, Loss:{}, Accuracy:{}".format(epoch, test_loss, correct))
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', correct, epoch)

if __name__ == '__main__':
    model = Model().to(device)
    print(model)
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.5)
    for i in range(1, Epoch + 1):
        train(model, optimizer, i)
        test(model, i)
    torch.save(model, 'single_digit.pth')
    writer.close()

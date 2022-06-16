import os

import torch
import torch.nn as nn

N_EPOCHS = 30
BATCH_SIZE = 128
LR = 0.01

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def instantiate_MNIST_model(name: str, device=None):
    assert name in ("linear", "conv")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)
    elif name == "conv":
        return nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(320, 10),
        ).to(device)


def instantiate_CIFAR10_model(name: str, device=None):
    assert name in ("linear", "conv")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).to(device)
    elif name == "conv":

        # adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        class CIFAR10_Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return CIFAR10_Net().to(device)
    else:
        assert False

def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: None,
):
    """Evaluate the model on the given data loader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        loss = 0.0
        correct = 0
        for input_, target in data_loader:
            input_ = input_.to(device)
            target = target.to(device)
            pred = model(input_)
            loss += criterion(pred, target).item()
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(data_loader.dataset)
        acc = correct / len(data_loader.dataset)
        return loss, acc

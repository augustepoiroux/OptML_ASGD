import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torchvision import datasets, transforms

N_EPOCHS = 30
BATCH_SIZE = 128
LR = 0.01

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NetworkModel(ABC):
    @classmethod
    @abstractmethod
    def num_epochs(cls):
        pass

    @classmethod
    @abstractmethod
    def batch_size(cls):
        pass

    @classmethod
    @abstractmethod
    def learning_rate(cls):
        pass

    @abstractmethod
    def fresh_model_instance(self, model_version: str, device: Any = None):
        pass

    @abstractmethod
    def dataset(self):
        pass


class MNIST_Model(NetworkModel):
    @classmethod
    def num_epochs(cls):
        return 30

    @classmethod
    def batch_size(cls):
        return 120

    @classmethod
    def learning_rate(cls):
        return 0.01

    def fresh_model_instance(self, model_version: str, device: Any = None):
        assert model_version in (
            "linear",
            "conv",
        ), f"No such model version: >>{model_version}<<"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_version == "linear":
            return nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)
        elif model_version == "conv":
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
        else:
            assert False, "No such model version"

    def dataset(self):
        return datasets.MNIST(
            os.path.join(ROOT_DIR, "data"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )


class CIFAR10_Model(NetworkModel):
    @classmethod
    def num_epochs(cls):
        return 30

    @classmethod
    def batch_size(cls):
        return 120

    @classmethod
    def learning_rate(cls):
        return 0.01

    def fresh_model_instance(self, model_version: str, device: Any = None):
        assert model_version in ("linear", "conv")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_version == "linear":
            return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).to(device)
        elif model_version == "conv":

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
            assert False, "No such model version"

    def dataset(self):
        return datasets.CIFAR10(
            os.path.join(ROOT_DIR, "data"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )


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

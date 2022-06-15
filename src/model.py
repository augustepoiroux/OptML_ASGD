import torch
import torch.nn as nn

N_EPOCHS = 30
BATCH_SIZE = 128
LR = 0.01


def instantiate_model(name: str, device="cpu"):
    if name == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)
    if name == "conv":
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


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
):
    """ Evaluate the model on the given data loader """
    with torch.no_grad():
        model.eval()
        loss = 0.0
        correct = 0
        for input, target in data_loader:
            input = input.to(device)
            target = target.to(device)
            pred = model(input)
            loss += criterion(pred, target).item()
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(data_loader.dataset)
        acc = correct / len(data_loader.dataset)
        return loss, acc
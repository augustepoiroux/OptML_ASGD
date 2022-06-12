import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from .data import DataPartitioner

N_EPOCHS = 30
BATCH_SIZE = 128

if __name__ == "__main__":
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a model
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)
    # model = nn.Sequential(
    #     nn.Conv2d(1, 10, kernel_size=5),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2),
    #     nn.Conv2d(10, 20, kernel_size=5),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2),
    #     nn.Flatten(),
    #     nn.Linear(320, 10),
    # ).to(device)

    # Load data
    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Partition the dataset into training, validation, and test sets
    partitioner = DataPartitioner(dataset, [0.7, 0.2, 0.1])
    train_set = partitioner.use(0)
    val_set = partitioner.use(1)
    test_set = partitioner.use(2)

    # Create a data loader for the training, validation, and test sets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True
    )

    # Create a loss and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Instantiates tensorboard
    writer = SummaryWriter(log_dir="./runs/sgd")

    # Train the model
    n_batch = 0
    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item() / len(data), n_batch)
            n_batch += 1
        train_loss /= len(train_loader.dataset)

        # Evaluate the model on the validation set
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            correct = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss /= len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)

        writer.add_scalar("Loss/val", val_loss, n_batch)
        writer.add_scalar("Accuracy/val", val_acc, n_batch)

        print(
            f"[Epoch {epoch + 1} / {N_EPOCHS}]"
            f"\ttrain loss: {train_loss:.4f}"
            f"\tval loss: {val_loss:.4f}"
            f"\tval accuracy: {val_acc * 100:.3f}%"
        )

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        test_acc = correct / len(test_loader.dataset)

        writer.add_scalar("Accuracy/test", test_acc, n_batch)

        print(f"Test accuracy: {test_acc*100:.2f}%")

    writer.flush()

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from .data import DataPartitioner

N_EPOCHS = 10
BATCH_SIZE = 128

if __name__ == "__main__":
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a model
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)

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

    # Train the model
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
        train_loss /= len(train_loader.dataset)

        # Evaluate the model on the validation set
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
        print(
            f"Test accuracy: {100 * correct / len(test_loader.dataset):.2f}%"
        )

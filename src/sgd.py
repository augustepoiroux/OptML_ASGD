import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from .data import DataPartitioner
from .model import (
    BATCH_SIZE,
    LR,
    N_EPOCHS,
    ROOT_DIR,
    evaluate,
    instantiate_model,
)

# fix seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    # Get torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a model
    model_name = "linear"
    model = instantiate_model(model_name, device)

    # Load data
    dataset = datasets.MNIST(
        os.path.join(ROOT_DIR, "data"),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Partition the dataset into training, validation, and test sets
    partitioner = DataPartitioner(dataset, [0.7, 0.2, 0.1], seed)
    train_set = partitioner.get_partition(0)
    val_set = partitioner.get_partition(1)
    test_set = partitioner.get_partition(2)

    # Create a data loader for the training, validation, and test sets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )

    # Create a loss and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Instantiates tensorboard
    writer = SummaryWriter(
        log_dir=os.path.join(ROOT_DIR, f"runs/sgd_{model_name}_{time.time()}")
    )

    # Train the model
    n_batch = 0
    nb_updates = N_EPOCHS * len(train_loader)
    print(f"Training for {N_EPOCHS} epochs (={nb_updates} updates)...")
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        writer.add_scalar("Loss/val", val_loss, n_batch)
        writer.add_scalar("Accuracy/val", val_acc, n_batch)
        print(
            f"[Batch {n_batch} / {nb_updates}]"
            f"\ttrain loss: {train_loss:.4f}"
            f"\tval loss: {val_loss:.4f}"
            f"\tval acc: {val_acc * 100:.3f}%"
        )

    # Evaluate the model on the test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    writer.add_scalar("Loss/test", test_loss, n_batch)
    writer.add_scalar("Accuracy/test", test_acc, n_batch)
    print(f"test loss {test_loss:.4f}\ttest acc {test_acc *100:.3f}%")

    writer.flush()

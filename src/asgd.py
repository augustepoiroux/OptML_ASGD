import argparse
import copy
import os
import random
import time
from dataclasses import dataclass, field
from functools import total_ordering
from queue import PriorityQueue
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from .data import partition_mnist
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


@total_ordering
@dataclass(eq=False)
class PrioritizedItem:
    time: float

    def __eq__(self, other):
        return self.time == other.time

    def __lt__(self, other):
        return self.time < other.time


@dataclass(eq=False)
class PrioritizedDevice(PrioritizedItem):
    device: int = field(compare=False)


@dataclass(eq=False)
class PrioritizedGradient(PrioritizedItem):
    gradient: Any = field(compare=False)
    loss: float = field(compare=False)


@dataclass(order=True)
class PrioritizedModelUpdate(PrioritizedItem):
    device: int = field(compare=False)


class ASGDTrainer:
    """Implement a fake ASGD trainer running on 1 device
    emulating the distributed training with a parameter server."""

    def __init__(
        self,
        algorithm: str,
        num_device: int,
        model_name: str,
        torch_device: torch.device = "cpu",
    ):
        self.algorithm = algorithm
        self.num_device = num_device
        self.model_name = model_name
        self.torch_device = torch_device
        self.queue: PriorityQueue[PrioritizedItem] = PriorityQueue()
        (
            self.train_partitions,
            self.val_set,
            self.test_set,
        ) = partition_mnist(num_device, seed=seed)

        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        model: nn.Module,
        nb_updates: int,
        val_every_n_updates: int,
        latency_dispersion: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        lr: float = LR,
        batch_size: int = BATCH_SIZE,
        log=True,
    ):
        """Train the model"""

        # Instantiate logger
        if log:
            writer = SummaryWriter(
                log_dir=os.path.join(
                    ROOT_DIR,
                    f"runs/asgd_{self.algorithm}-algo_{self.model_name}_{self.num_device}"
                    f"-devices_{latency_dispersion:.2f}-latency_{time.time()}",
                )
            )

        # Create a data loader for the training, validation, and test sets
        train_loaders = [
            torch.utils.data.DataLoader(
                partition, batch_size=batch_size, shuffle=True, pin_memory=True
            )
            for partition in self.train_partitions
        ]
        val_loader = torch.utils.data.DataLoader(
            self.val_set, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        # Instantiate an optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr / np.sqrt(self.num_device) if self.algorithm == "adjusted" else lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Define devices characteristics
        # latency is modeled using a LogNormal distribution
        mu_time_train = 0.0
        sigma_time_train = latency_dispersion

        mu_time_gradient = 0.0
        sigma_time_gradient = latency_dispersion

        mu_time_update = 0.0
        sigma_time_update = latency_dispersion

        # Global time
        t = 0

        # copy models on each device
        model.to(self.torch_device)
        models = [copy.deepcopy(model) for _ in range(self.num_device)]

        if self.algorithm == "dcasgd":
            models_backup = [copy.deepcopy(model) for _ in range(self.num_device)]

        # fill queue with orders to execute
        for device in range(self.num_device):
            device_time = t + np.random.lognormal(mu_time_train, sigma_time_train)
            self.queue.put(PrioritizedDevice(device_time, device))

        n_update = 0
        n_batch = [0 for _ in range(self.num_device)]
        while n_update < nb_updates:
            # get next item to process
            item = self.queue.get()

            t = item.time

            if isinstance(item, PrioritizedDevice):
                device = item.device

                # select a batch
                input, target = next(iter(train_loaders[device]))
                input = input.to(self.torch_device)
                target = target.to(self.torch_device)

                # compute gradient
                models[device].zero_grad()
                models[device].train()
                pred = models[device](input)
                loss = self.criterion(pred, target)
                loss.backward()
                train_loss = loss.item() / len(input)

                if log:
                    writer.add_scalar(
                        f"Loss_device/train_{device}",
                        train_loss,
                        n_batch[device],
                    )
                n_batch[device] += 1

                # add gradient to queue
                grad = []
                for param in models[device].parameters():
                    grad.append(param.grad.data.clone())
                grad_time = t + np.random.lognormal(
                    mu_time_gradient, sigma_time_gradient
                )
                self.queue.put(PrioritizedGradient(grad_time, grad, train_loss))

                # add model update to queue
                model_update_time = t + np.random.lognormal(
                    mu_time_update, sigma_time_update
                )
                self.queue.put(PrioritizedModelUpdate(model_update_time, device))

            elif isinstance(item, PrioritizedGradient):
                # update parameter server model with gradient
                model.zero_grad()
                grad = item.gradient
                if self.algorithm == "dcasgd":
                    for param, param_backup, grad_param in zip(
                        model.parameters(),
                        models_backup[device].parameters(),
                        grad,
                    ):
                        with torch.no_grad():
                            compensated_grad = (
                                grad_param
                                + VAR_CONTROL
                                * grad_param
                                * grad_param
                                * (param - param_backup)
                            )
                        param.grad = compensated_grad
                else:
                    for param, grad_param in zip(model.parameters(), grad):
                        param.grad = grad_param
                optimizer.step()

                if log:
                    writer.add_scalar("Loss/train", item.loss, n_update)
                n_update += 1

                # evaluate model on validation set
                if n_update % val_every_n_updates == 0:
                    val_loss, val_acc = self.evaluate(model, val_loader)
                    if log:
                        writer.add_scalar("Loss/val", val_loss, n_update)
                        writer.add_scalar("Accuracy/val", val_acc, n_update)
                    print(
                        f"[Batch {n_update} / {nb_updates}]"
                        f"\tval loss: {val_loss:.4f}"
                        f"\tval acc: {val_acc *100:.3f}%"
                    )

            elif isinstance(item, PrioritizedModelUpdate):
                # update model on device
                device = item.device
                models[device] = copy.deepcopy(model)
                if self.algorithm == "dcasgd":
                    models_backup[device] = copy.deepcopy(model)
                train_time = t + np.random.lognormal(mu_time_train, sigma_time_train)
                self.queue.put(PrioritizedDevice(train_time, device))

        # evaluate model on test set
        test_loss, test_acc = self.evaluate(model, test_loader)
        if log:
            writer.add_scalar("Loss/test", test_loss, n_update)
            writer.add_scalar("Accuracy/test", test_acc, n_update)
            writer.flush()
        print(f"test loss {test_loss:.4f}\ttest acc {test_acc *100:.3f}%")

    def evaluate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader):
        return evaluate(model, data_loader, self.criterion, self.torch_device)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description="ASGD for distributed training")
    parser.add_argument(
        "--algo",
        type=str,
        default="raw",
        help="algorithm to use",
        choices=["raw", "adjusted", "dcasgd"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="conv",
        help="Model to use",
        choices=["conv", "linear"],
    )
    parser.add_argument(
        "--num-device",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    parser.add_argument(
        "--latency-dispersion",
        type=float,
        default=0.7,
        help="Latency dispersion",
    )
    parser.add_argument(
        "--var-control", type=float, default=0.1, help="Variance control",
    )

    args = parser.parse_args()

    # Parse arguments
    NUM_DEVICE = args.num_device
    LATENCY_DISPERSION = args.latency_dispersion
    MODEL_NAME = args.model
    ALGORITHM = args.algo
    VAR_CONTROL = args.var_control

    # Get device
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a model
    MODEL = instantiate_model(MODEL_NAME).to(TORCH_DEVICE)

    trainer = ASGDTrainer(ALGORITHM, NUM_DEVICE, MODEL_NAME, TORCH_DEVICE)
    trainer.train(
        model=MODEL,
        nb_updates=N_EPOCHS * 329,
        val_every_n_updates=329,
        latency_dispersion=LATENCY_DISPERSION,
        # log=False,
    )

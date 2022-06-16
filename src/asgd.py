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

from .models import (
    ROOT_DIR,
    evaluate,
    NetworkModel,
    MNIST_Model,
    CIFAR10_Model,
)

# Fix the seed for reproducibility
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
    device: int = field(compare=False)


@dataclass(order=True)
class PrioritizedModelUpdate(PrioritizedItem):
    device: int = field(compare=False)
    model: Any = field(compare=False)


class ASGDTrainer:
    """This class simulates a network of devices performing asynchronous SGD.
    Topology is star-shaped, with one parameter server holding the authoritative
    knowledge on the model, and a number of worker devices, each communicating
    only with the server, computing gradients and sending them back to the
    parameter server, receiving an updated model parameters back."""

    def __init__(
        self,
        nn_model: NetworkModel,
        algorithm: str,
        num_device: int,
        model_name: str,
        torch_device: torch.device = None,
    ):
        assert algorithm in (
            "raw",
            "adjusted",
            "dcasgd",
        ), f"Unknown algorithm type >{algorithm}<"
        assert num_device >= 1
        assert model_name in ("linear", "conv")
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.neural_network_model = nn_model
        self.algorithm = algorithm
        self.num_device = num_device
        self.model_name = model_name
        self.torch_device = torch_device
        self.queue: PriorityQueue[PrioritizedItem] = PriorityQueue()
        (
            self.train_partitions,
            self.val_set,
            self.test_set,
        ) = nn_model.partition_dataset(num_device, seed=seed)

        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        model: nn.Module,
        nb_updates: int,
        val_every_n_updates: int,
        latency_dispersion: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        var_control: float = 0.1,  # DC-ASGD parameter
        log=True,
    ):
        """Train the model"""

        lr = self.neural_network_model.learning_rate()
        batch_size = self.neural_network_model.batch_size()

        # Instantiate the TensorBoard log writer
        if log:
            algo = (
                f"{self.algorithm}-{var_control:.2f}"
                if self.algorithm == "dcasgd"
                else self.algorithm
            )
            writer = SummaryWriter(
                log_dir=os.path.join(
                    ROOT_DIR,
                    f"runs/asgd_{algo}-algo_{self.model_name}_{self.num_device}"
                    f"-devices_{latency_dispersion:.2f}-latency_{time.time()}",
                )
            )

        # Create data loaders for the training, validation, and test sets
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

        # Instantiate the optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr / np.sqrt(self.num_device) if self.algorithm == "adjusted" else lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Training latency is modeled using a normal distribution
        mu_time_train = 1.0
        sigma_time_train = 0.1

        # Device latency is modeled using the log-normal distribution
        mu_time_gradient = 0.0
        sigma_time_gradient = latency_dispersion

        mu_time_update = 0.0
        sigma_time_update = latency_dispersion

        # Global timestamp, units are not important
        time_now = 0

        # Each device gets its own copy of the initial model ("dcasgd" needs two copies)
        model.to(self.torch_device)
        models = [copy.deepcopy(model) for _ in range(self.num_device)]
        if self.algorithm == "dcasgd":
            models_backup = [copy.deepcopy(model) for _ in range(self.num_device)]

        # Initially the queue contains events describing each device getting
        # its own training data and training on them.
        for emulated_device in range(self.num_device):
            device_time = time_now + np.random.lognormal(
                mu_time_train, sigma_time_train
            )
            self.queue.put(PrioritizedDevice(device_time, emulated_device))

        n_update = 0
        n_batch = [0 for _ in range(self.num_device)]
        while n_update < nb_updates:
            # Get the next item to process
            item = self.queue.get()

            time_now = item.time

            if isinstance(item, PrioritizedDevice):
                """This emulated device has just finished training on its datapoint,
                and now it sends the gradients it computed back to the main server
                (after some communication delay), and waits to receive the
                updated model parameter weights, also after some delay."""

                this_emulated_device = item.device

                # Get data to train on
                input_, target = next(iter(train_loaders[this_emulated_device]))
                input_ = input_.to(self.torch_device)
                target = target.to(self.torch_device)

                # Compute their gradient
                models[this_emulated_device].zero_grad()
                models[this_emulated_device].train()
                pred = models[this_emulated_device](input_)
                loss = self.criterion(pred, target)
                loss.backward()
                train_loss = loss.item() / len(input_)

                if log:
                    writer.add_scalar(
                        f"Loss_device/train_{this_emulated_device}",
                        train_loss,
                        n_batch[this_emulated_device],
                    )

                n_batch[this_emulated_device] += 1

                # Send the gradient back to the main server, but after some time
                grad = [
                    param.grad.data.clone()
                    for param in models[this_emulated_device].parameters()
                ]
                grad_time = time_now + np.random.lognormal(
                    mu_time_gradient, sigma_time_gradient
                )
                self.queue.put(
                    PrioritizedGradient(
                        grad_time, grad, train_loss, this_emulated_device
                    )
                )

            elif isinstance(item, PrioritizedGradient):
                """The global model's parameters get updated with the gradient from this message."""
                model.zero_grad()
                grad = item.gradient
                if self.algorithm == "dcasgd":
                    for param, param_backup, grad_param in zip(
                        model.parameters(),
                        models_backup[this_emulated_device].parameters(),
                        grad,
                    ):
                        with torch.no_grad():
                            compensated_grad = (
                                grad_param
                                + var_control
                                * grad_param
                                * grad_param
                                * (param - param_backup)
                            )
                        param.grad = compensated_grad
                elif self.algorithm in ("raw", "adjusted"):
                    for param, grad_param in zip(model.parameters(), grad):
                        param.grad = grad_param
                else:
                    assert False, f"Unknown algorithm type: {str(self.algorithm)}"

                optimizer.step()

                if log:
                    writer.add_scalar("Loss/train", item.loss, n_update)

                # Update model on the device, but after some time
                model_update_time = time_now + np.random.lognormal(
                    mu_time_update, sigma_time_update
                )
                self.queue.put(
                    PrioritizedModelUpdate(
                        model_update_time, item.device, copy.deepcopy(model)
                    )
                )

                n_update += 1

                # Evaluate the model on validation data according to `val_every_n_updates`
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
                """Model on this specific device gets replaced with the global model"""
                models[item.device] = copy.deepcopy(item.model)
                if self.algorithm == "dcasgd":
                    models_backup[item.device] = copy.deepcopy(item.model)
                train_time = time_now + max(
                    0, np.random.normal(mu_time_train, sigma_time_train)
                )
                self.queue.put(PrioritizedDevice(train_time, item.device))

            else:
                assert False, "Unknown message on the queue."

        # The final evaluation on test data
        test_loss, test_acc = self.evaluate(model, test_loader)
        if log:
            writer.add_scalar("Loss/test", test_loss, n_update)
            writer.add_scalar("Accuracy/test", test_acc, n_update)
            writer.flush()
        print(f"test loss {test_loss:.4f}\ttest acc {test_acc *100:.3f}%")

    def evaluate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader):
        return evaluate(model, data_loader, self.criterion, self.torch_device)


if __name__ == "__main__":
    # Argparse specification
    parser = argparse.ArgumentParser(description="ASGD for distributed training")
    parser.add_argument(
        "--algo",
        type=str,
        default="raw",
        help="Algorithm to use",
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
        "--var-control",
        type=float,
        default=0.1,
        help="Variance control",
    )

    # Parse arguments
    args = parser.parse_args()
    NUM_DEVICE = args.num_device
    LATENCY_DISPERSION = args.latency_dispersion
    MODEL_NAME = args.model
    ALGORITHM = args.algo
    VAR_CONTROL = args.var_control

    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neural_network_model = CIFAR10_Model()
    N_EPOCHS = neural_network_model.num_epochs()
    MODEL = neural_network_model.fresh_model_instance(MODEL_NAME, TORCH_DEVICE)

    trainer = ASGDTrainer(
        neural_network_model, ALGORITHM, NUM_DEVICE, MODEL_NAME, TORCH_DEVICE
    )
    trainer.train(
        model=MODEL,
        nb_updates=N_EPOCHS * 329,
        val_every_n_updates=329,
        latency_dispersion=LATENCY_DISPERSION,
        var_control=VAR_CONTROL,
        # log=False,
    )

import copy
from dataclasses import dataclass, field
from functools import total_ordering
from queue import PriorityQueue
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .data import partition_mnist_v2


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


@dataclass(order=True)
class PrioritizedModelUpdate(PrioritizedItem):
    device: int = field(compare=False)


class ASGDTrainer:
    """ Implement a fake ASGD trainer running on 1 device
    emulating the distributed training with a parameter server."""

    def __init__(self, num_device: int):
        self.num_device = num_device
        self.torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.queue: PriorityQueue[PrioritizedItem] = PriorityQueue()
        (
            self.train_partitions,
            self.val_set,
            self.test_set,
        ) = partition_mnist_v2(num_device)

        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        model: nn.Module,
        nb_updates: int,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        val_every_n_updates: int = 100,
    ):
        """ Train the model """

        # Create a data loader for the training, validation, and test sets
        train_loaders = [
            torch.utils.data.DataLoader(
                partition, batch_size=batch_size, shuffle=True
            )
            for partition in self.train_partitions
        ]
        val_loader = torch.utils.data.DataLoader(
            self.val_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=True
        )

        # Instantiate an optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Define devices characteristics
        mean_time_train = 1.0
        std_time_train = 0.5

        mean_time_gradient = 0.2
        std_time_gradient = 0.1

        mean_time_update = 0.1
        std_time_update = 0.05

        # Global time
        t = 0

        # copy models on each device
        model.to(self.torch_device)
        models = [copy.deepcopy(model) for _ in range(self.num_device)]

        # fill queue with orders to execute
        for device in range(self.num_device):
            device_time = np.random.normal(t + mean_time_train, std_time_train)
            self.queue.put(PrioritizedDevice(device_time, device))

        n_update = 0
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

                # add gradient to queue
                grad = []
                for param in models[device].parameters():
                    grad.append(param.grad.data.clone())
                grad_time = t + np.random.normal(
                    mean_time_gradient, std_time_gradient
                )
                self.queue.put(PrioritizedGradient(grad_time, grad))

                # add model update to queue
                model_update_time = t + np.random.normal(
                    mean_time_update, std_time_update
                )
                self.queue.put(
                    PrioritizedModelUpdate(model_update_time, device)
                )

            elif isinstance(item, PrioritizedGradient):
                # update parameter server model with gradient
                model.zero_grad()
                grad = item.gradient
                for param, grad_param in zip(model.parameters(), grad):
                    param.grad = grad_param
                optimizer.step()
                n_update += 1

                # evaluate model on validation set
                if n_update % val_every_n_updates == 0:
                    val_loss, val_acc = self.evaluate(model, val_loader)
                    print(
                        f"[{n_update}]"
                        f"\tval_loss {val_loss:.4f}"
                        + f"\tval_acc {val_acc *100:.4f}%"
                    )

            elif isinstance(item, PrioritizedModelUpdate):
                # update model on device
                device = item.device
                models[device] = copy.deepcopy(model)
                train_time = t + np.random.normal(
                    mean_time_train, std_time_train
                )
                self.queue.put(PrioritizedDevice(train_time, device))

        # evaluate model on test set
        test_loss, test_acc = self.evaluate(model, test_loader)
        print(f"test_loss {test_loss:.4f}\ttest_acc {test_acc *100:.4f}%")

    def evaluate(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader
    ):
        """ Evaluate the model on the given data loader """
        model.eval()
        loss = 0.0
        correct = 0
        for input, target in data_loader:
            input = input.to(self.torch_device)
            target = target.to(self.torch_device)
            pred = model(input)
            loss += self.criterion(pred, target).item()
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(data_loader)
        acc = correct / len(data_loader.dataset)
        return loss, acc


if __name__ == "__main__":
    trainer = ASGDTrainer(num_device=10)
    trainer.train(
        model=nn.Sequential(nn.Flatten(), nn.Linear(784, 10)),
        nb_updates=1000,
        lr=0.01,
        val_every_n_updates=10,
    )

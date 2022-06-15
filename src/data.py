import numpy as np

from torchvision import datasets, transforms


class Partition:
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner:
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]

        np.random.seed(seed)
        np.random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_mnist(
    num_train_partitions,
    train_size=0.7,
    val_size=0.2,
    test_size=0.1,
    seed=1234,
):
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    partition_sizes = [
        train_size / num_train_partitions for _ in range(num_train_partitions)
    ] + [val_size, test_size]
    partitioner = DataPartitioner(dataset, partition_sizes, seed)
    train_partitions = [
        partitioner.use(i) for i in range(num_train_partitions)
    ]
    return (
        train_partitions,
        partitioner.use(num_train_partitions),
        partitioner.use(num_train_partitions + 1),
    )

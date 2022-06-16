from typing import List
import numpy as np


class Partition:
    """
    Represents a subset of datapoints (selected by `indices`) from some larger dataset `data`.
    Partition behaves like a list that contains only the selected data points.
    """

    def __init__(self, data: list, indices: List[int]):
        assert all(0 <= i < len(data) for i in indices)
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]


class DataPartitioner:
    """
    DataPartitioner receives a list of datapoints and partitions them according to `sizes`,
    a list of fractions adding to 1 describing relative sized of the resulting partitions.
    Partitions are accessible by self.get_partition() and indexed by 0..len(sizes)-1.
    """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        assert all(0 <= fraction <= 1 for fraction in sizes)
        assert abs(sum(sizes) - 1) <= 0.001

        self.data = data
        self.partitions = []
        data_len = len(data)
        indices = list(range(0, data_len))

        np.random.seed(seed)
        np.random.shuffle(indices)

        for fraction in sizes:
            part_len = int(fraction * data_len)
            self.partitions.append(indices[0:part_len])
            indices = indices[part_len:]

    def get_partition(self, partition):
        return Partition(self.data, self.partitions[partition])

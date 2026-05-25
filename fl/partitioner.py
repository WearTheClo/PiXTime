"""Data partitioner for federated learning."""

import numpy as np
from torch.utils.data import Subset, Dataset
from typing import List


class DataPartitioner:
    """Data partitioner for federated learning - each client self-partitions using fixed seed."""

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        partition_method: str = "iid",
        alpha: float = 0.5,
        seed: int = 42,
    ):
        """Initialize DataPartitioner.

        Args:
            dataset: Full dataset to partition
            num_clients: Number of clients
            partition_method: 'iid' or 'dirichlet'
            alpha: Dirichlet alpha parameter (for non-IID)
            seed: Random seed (must be same for all clients)
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_method = partition_method
        self.alpha = alpha
        self.seed = seed

    def partition(self) -> List[List[int]]:
        """Partition dataset and return indices for each client.

        Returns:
            List of index lists, one per client
        """
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)

        np.random.seed(self.seed)

        if self.partition_method == "iid":
            # IID 场景：全局打乱，绝对均匀平分
            np.random.shuffle(indices)
            partition_sizes = np.array([n_samples // self.num_clients] * self.num_clients)
            for i in range(n_samples % self.num_clients):
                partition_sizes[i] += 1

            partitions = []
            start_idx = 0
            for size in partition_sizes:
                partitions.append(indices[start_idx:start_idx + size].tolist())
                start_idx += size
        else:
            # Non-IID 场景 (Dirichlet)
            # 1. 先给每个人保底 min_size
            min_size = 10
            if n_samples < min_size * self.num_clients:
                raise ValueError("Dataset is too small to distribute among clients.")

            # 2. 计算剩余可自由分配的数据量
            remaining_samples = n_samples - (min_size * self.num_clients)

            # 3. 使用 Dirichlet 生成比例，分配剩余数据
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            extra_alloc = (proportions * remaining_samples).astype(int)

            # 4. 把保底和额外分配加起来
            partition_sizes = np.array([min_size] * self.num_clients) + extra_alloc

            # 5. 把四舍五入丢掉的零头补给第一个客户端
            diff = n_samples - partition_sizes.sum()
            partition_sizes[0] += diff

            partitions = []
            start_idx = 0
            for size in partition_sizes:
                partitions.append(indices[start_idx:start_idx + size].tolist())
                start_idx += size

        return partitions

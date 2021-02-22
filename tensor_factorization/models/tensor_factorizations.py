from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

import common.utils.tensor as tensor_utils
from common.utils.tensor import convert_tensor_to_one_hot


class TensorFactorization(nn.Module, ABC):

    @abstractmethod
    def compute_tensor(self) -> torch.Tensor:
        """
        Computes the tensor.
        """
        raise NotImplementedError


class TensorCPFactorization(TensorFactorization):
    """
    Tensor CP factorization model.
    """

    def __init__(self, num_dim_per_mode: Sequence[int], rank: int = -1, init_mean: float = 0., init_std: float = 0.01):
        """
        :param num_dim_per_mode: Number of dimensions per tensor mode.
        :param rank: Number of tensor products sums of the factorization. If -1 the default is the max possible CP rank which is the product
        of all dimensions except the max.
        :param init_mean: mean of vectors gaussian init.
        :param init_std: std of vectors gaussian init.
        """
        super().__init__()
        self.num_dim_per_mode = num_dim_per_mode
        self.order = len(self.num_dim_per_mode)
        self.init_mean = init_mean
        self.init_std = init_std
        self.rank = rank if rank != -1 else self.__compute_max_possible_tensor_rank()

        self.factors = nn.ParameterList(self.__create_factors())

    def __compute_max_possible_tensor_rank(self):
        tensor_dims = list(self.num_dim_per_mode)
        max_index = tensor_dims.index(max(tensor_dims))
        tensor_dims.pop(max_index)

        return np.prod(tensor_dims)

    def __create_factors(self):
        factors = []
        for dim in self.num_dim_per_mode:
            factor = torch.randn(dim, self.rank, dtype=torch.float) * self.init_std + self.init_mean
            factors.append(factor)

        factors = [nn.Parameter(factor, requires_grad=True) for factor in factors]
        return factors

    def compute_tensor(self):
        return tensor_utils.reconstruct_parafac(self.factors)

    def forward(self, tensor_of_indices):
        tensor = self.compute_tensor()
        split_indices = [tensor_of_indices[:, i] for i in range(tensor_of_indices.size(1))]
        tensor_values = tensor[split_indices]
        return tensor_values


class LargeScaleTensorCPFactorization(nn.Module):
    """
    Tensor CP factorization model that supports large scale tensors which can't be computed entirely. Allows access to given tensor indices.
    Currently supports only factorizations with equal dimensions in each mode.
    """

    def __init__(self, modes_dim: int, order: int, rank: int = 1, init_mean: float = 0., init_std: float = 0.01, device=torch.device("cpu")):
        """
        :param modes_dim: Number of dimensions in the tensor modes.
        :param rank: Number of tensor products sums of the factorization. If -1 the default is the max possible CP rank which is the product
        of all dimensions except the max.
        :param init_mean: mean of vectors gaussian init.
        :param init_std: std of vectors gaussian init.
        :param device: device to initialize the parameters on.
        """
        super().__init__()
        self.modes_dim = modes_dim
        self.num_dim_per_mode = [modes_dim] * order
        self.order = order
        self.rank = rank
        self.init_std = init_std
        self.init_mean = init_mean

        self.factors = nn.Parameter(self.__create_factors(device=device), requires_grad=True)

    def __create_factors(self, device):
        return torch.randn(self.order, self.modes_dim, self.rank, device=device) * self.init_std + self.init_mean

    def compute_index_in_tensor(self, indices):
        return tensor_utils.reconstruct_index(self.factors, indices)

    def forward(self, indices_tensor, one_hot_input: bool = True):
        """
        :param indices_tensor: If one_hot_onput is False, Tensor of indices in shape (B, order), where B is the batch size and each column
        corresponds to a tensor mode. Otherwise (default), a one hot representation of the indices tensor of shape (B, order, mode_dim).
        :param one_hot_input: Flag that determines correct format for indices_tensor.
        """
        if not one_hot_input:
            indices_tensor = convert_tensor_to_one_hot(indices_tensor, num_options=self.modes_dim)

        a = torch.einsum('sab, abt -> sat', indices_tensor, self.factors).prod(dim=1).sum(dim=1)
        return a

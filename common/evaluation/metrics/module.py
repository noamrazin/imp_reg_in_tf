from typing import Sequence

import torch
import torch.nn as nn

from .metric import AveragedMetric
from ...utils import module as module_utils


class ParameterValueMean(AveragedMetric):
    """
    Mean of parameter values metric. Allows to compute mean only for specific types of layers.
    """

    def __init__(self, exclude: Sequence[type] = None, include_only: Sequence[type] = None, exclude_by_name_part: Sequence[str] = None):
        """
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__()
        self.exclude = exclude
        self.include_only = include_only
        self.exclude_by_name_part = exclude_by_name_part

    def _calc_metric(self, module: nn.Module):
        """
        :param module: PyTorch module
        :return: Mean value of parameters from relevant layers.
        """
        params = list(module_utils.get_parameters_iter(module, exclude=self.exclude,
                                                       include_only=self.include_only, exclude_by_name_part=self.exclude_by_name_part))
        flattened_params_vector = torch.cat([param.view(-1) for param in params])
        return flattened_params_vector.mean().item(), 1


class ParameterValueSTD(AveragedMetric):
    """
    Standard deviation of parameter values metric. Allows to compute mean only for specific types of layers.
    """

    def __init__(self, exclude: Sequence[type] = None, include_only: Sequence[type] = None, exclude_by_name_part: Sequence[str] = None):
        """
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__()
        self.exclude = exclude
        self.include_only = include_only
        self.exclude_by_name_part = exclude_by_name_part

    def _calc_metric(self, module: nn.Module):
        """
        :param module: PyTorch module
        :return: Standard deviation of parameters from relevant layers.
        """
        params = list(module_utils.get_parameters_iter(module, exclude=self.exclude,
                                                       include_only=self.include_only, exclude_by_name_part=self.exclude_by_name_part))
        flattened_params_vector = torch.cat([param.view(-1) for param in params])
        return flattened_params_vector.std().item(), 1


class ParameterQuantileValue(AveragedMetric):
    """
    Quantile parameter value metric. E.g. Allows to compute the median parameter value.
    """

    def __init__(self, quantile: float = 0.5, exclude: Sequence[type] = None, include_only: Sequence[type] = None,
                 exclude_by_name_part: Sequence[str] = None):
        """
        @param quantile: quantile value of parameters to return.
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__()
        self.quantile = quantile
        self.exclude = exclude
        self.include_only = include_only
        self.exclude_by_name_part = exclude_by_name_part

    def _calc_metric(self, module: nn.Module):
        """
        :param module: PyTorch module
        :return: Quantile value of parameters from relevant layers.
        """
        params = list(module_utils.get_parameters_iter(module, exclude=self.exclude,
                                                       include_only=self.include_only, exclude_by_name_part=self.exclude_by_name_part))
        flattened_params_vector = torch.cat([param.view(-1) for param in params])
        return torch.quantile(flattened_params_vector, q=self.quantile).item(), 1


class ParameterAbsoluteValueMean(AveragedMetric):
    """
    Mean of parameter absolute values metric. Allows to compute mean only for specific types of layers.
    """

    def __init__(self, exclude: Sequence[type] = None, include_only: Sequence[type] = None, exclude_by_name_part: Sequence[str] = None):
        """
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__()
        self.exclude = exclude
        self.include_only = include_only
        self.exclude_by_name_part = exclude_by_name_part

    def _calc_metric(self, module: nn.Module):
        """
        :param module: PyTorch module
        :return: Mean absolute value of parameters from relevant layers.
        """
        params = list(module_utils.get_parameters_iter(module, exclude=self.exclude,
                                                       include_only=self.include_only, exclude_by_name_part=self.exclude_by_name_part))
        flattened_abs_params_vector = torch.cat([torch.abs(param.view(-1)) for param in params])
        return flattened_abs_params_vector.mean().item(), 1


class ParameterAbsoluteValueSTD(AveragedMetric):
    """
    Standard deviation of parameter absolute values metric. Allows to compute mean only for specific types of layers.
    """

    def __init__(self, exclude: Sequence[type] = None, include_only: Sequence[type] = None, exclude_by_name_part: Sequence[str] = None):
        """
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__()
        self.exclude = exclude
        self.include_only = include_only
        self.exclude_by_name_part = exclude_by_name_part

    def _calc_metric(self, module: nn.Module):
        """
        :param module: PyTorch module
        :return: Standard deviation of absolute value of parameters from relevant layers.
        """
        params = list(module_utils.get_parameters_iter(module, exclude=self.exclude,
                                                       include_only=self.include_only, exclude_by_name_part=self.exclude_by_name_part))
        flattened_abs_params_vector = torch.cat([torch.abs(param.view(-1)) for param in params])
        return flattened_abs_params_vector.std().item(), 1

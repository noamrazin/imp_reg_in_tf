from typing import Dict

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator
from common.evaluation.evaluators import MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.train.tracked_value import TrackedValue
from tensor_factorization.evaluation.cp_factorization_metrics import CPFactorColNorm
from tensor_factorization.models.tensor_factorizations import LargeScaleTensorCPFactorization


class LargeScaleCPFactorizationEvaluator(Evaluator):
    FACTOR_COL_NORM_METRIC_NAME_TEMPLATE = "factor_{0}_col_{1}_fro_norm"

    def __init__(self, cp_model: LargeScaleTensorCPFactorization, track_factors_col_norm: bool = True, num_factors_to_track: int = 5,
                 device=module_utils.get_device()):
        self.cp_model = cp_model
        self.track_factors_col_norm = track_factors_col_norm
        self.num_factors_to_track = num_factors_to_track
        self.device = device

        self.metric_infos = {}
        if self.track_factors_col_norm:
            self.metric_infos.update(self.__create_factor_col_norms_metric_infos())

        self.metrics = {metric_info.name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_factor_col_norms_metric_infos(self):
        metric_infos = {}
        for factor_index in range(min(self.cp_model.order, self.num_factors_to_track)):
            for col_index in range(self.cp_model.rank):
                factor_col_norm_metric = CPFactorColNorm(factor_index=factor_index, col_index=col_index, norm="fro")
                metric_name = self.FACTOR_COL_NORM_METRIC_NAME_TEMPLATE.format(factor_index, col_index)
                metric_tag = f"factor {factor_index} cols Fro norm"
                metric_infos[metric_name] = MetricInfo(metric_name, factor_col_norm_metric, tag=metric_tag)

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.cp_model.to(self.device)

            metric_values = {}
            for name, metric in self.metrics.items():
                value = metric(self.cp_model)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            return metric_values

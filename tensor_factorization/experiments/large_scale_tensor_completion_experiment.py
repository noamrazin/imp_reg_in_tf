import argparse
import logging
from collections import OrderedDict
from typing import Tuple

import torch
import torch.utils
import torch.utils.data
from torch import nn as nn

from common.data.modules import DataModule
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import SupervisedTrainEvaluator, TrainEvaluator, \
    Evaluator, ComposeEvaluator
from common.evaluation.evaluators import SupervisedValidationEvaluator
from common.experiment import ExperimentBase
from common.experiment.experiment_base import ScoreInfo
from common.train import callbacks as callbacks
from common.train.callbacks import Callback
from common.train.optim import GroupRMSprop
from common.train.trainer import Trainer
from common.train.trainers import SupervisedTrainer
from tensor_factorization.datasets.large_scale_tensor_completion_datamodule import LargeScaleTensorCompletionDataModule
from tensor_factorization.evaluation.large_scale_cp_factorization_evaluator import LargeScaleCPFactorizationEvaluator
from tensor_factorization.models.tensor_factorizations import LargeScaleTensorCPFactorization
from tensor_factorization.trainers.weight_clamp_callback import WeightClampCallback


class LargeScaleTensorCompletionExperiment(ExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser: argparse.ArgumentParser):
        ExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, required=True, help="Path to the the dataset file")
        parser.add_argument("--load_dataset_to_gpu", action="store_true", help="Stores all dataset on the main GPU (if GPU device is given)")
        parser.add_argument("--num_samples", type=int, default=-1, help="Number of train samples to use. If < 0 will use the whole dataset")
        parser.add_argument("--batch_size", type=int, default=-1, help="Train batch size. If <= 0 will use the whole training set each batch")
        parser.add_argument("--loss", type=str, default="l2", help="Loss to use. Currently supports: 'l2', 'l1', 'huber'.")
        parser.add_argument("--lr", type=float, default=1e-2, help="Training learning rate")
        parser.add_argument("--optimizer", type=str, default="grouprmsprop", help="Optimizer to use. Supports: 'grouprmsprop', 'adam', 'sgd'")
        parser.add_argument("--momentum", type=float, default=0, help="Momentum for SGD")
        parser.add_argument("--stop_on_zero_loss_tol", type=float, default=1e-8, help="Stops when train loss reaches below this threshold")

        parser.add_argument("--num_cp_components", type=int, default=1, help="Number of components to use in the cp factorization")
        parser.add_argument("--init_mean", type=float, default=1, help="Init mean for gaussian init")
        parser.add_argument("--init_std", type=float, default=1e-3, help="Init std for gaussian init")
        parser.add_argument("--max_weights_value", type=float, default=-1, help="Weights magnitude constraint. (set -1 for no constraint)")
        parser.add_argument("--track_factor_norms", action="store_true",
                            help="Track cp factorization factors, components and factor column norms. Currently only supported for CP factorizaion")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None
        data_module = LargeScaleTensorCompletionDataModule(config["dataset_path"], config["num_samples"],
                                                           batch_size=config["batch_size"],
                                                           precompute_one_hot_repr=True,
                                                           load_dataset_to_device=load_dataset_to_device,
                                                           shuffle_train=True)
        data_module.setup()
        return data_module

    def create_model(self, datamodule: LargeScaleTensorCompletionDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        dataset = datamodule.dataset
        return LargeScaleTensorCPFactorization(modes_dim=dataset.target_tensor_modes_dim,
                                               order=dataset.tensor_indices.shape[1],
                                               rank=config["num_cp_components"],
                                               init_mean=config["init_mean"],
                                               init_std=config["init_std"])

    def create_train_and_validation_evaluators(self, model: LargeScaleTensorCPFactorization, datamodule: LargeScaleTensorCompletionDataModule, device,
                                               config: dict, state: dict, logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [metrics.MetricInfo("mse_loss", metrics.MSELoss(), tag="mse"),
                                 self.__get_train_loss_metric_info(config)]
        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        val_metric_info_seq = [metrics.MetricInfo("val_mse_loss", metrics.MSELoss(), tag="mse")]
        val_dataloader = datamodule.val_dataloader()

        val_evaluators = [SupervisedValidationEvaluator(model, val_dataloader, val_metric_info_seq, device=device)]

        if config["track_factor_norms"]:
            val_evaluators.append(LargeScaleCPFactorizationEvaluator(model, device=device))

        return train_evaluator, ComposeEvaluator(val_evaluators)

    def __get_train_loss_metric_info(self, config: dict):
        if config["loss"] == "l2":
            return metrics.MetricInfo("l2_loss", metrics.MSELoss())
        elif config["loss"] == "l1":
            return metrics.MetricInfo("l1_loss", metrics.L1Loss())
        elif config["loss"] == "huber":
            return metrics.MetricInfo("huber_loss", metrics.SmoothL1Loss())
        else:
            raise ValueError(f"Unsupported loss type: '{config['loss']}'")

    def __get_train_loss(self, config: dict):
        if config["loss"] == "l2":
            return nn.MSELoss()
        elif config["loss"] == "l1":
            return nn.L1Loss()
        elif config["loss"] == "huber":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: '{config['loss']}'")

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="mse_loss", is_train_metric=True, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: LargeScaleTensorCPFactorization, datamodule: LargeScaleTensorCompletionDataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        dataset = datamodule.dataset
        additional_metadata.update({
            "dataset size": len(dataset),
            "num train samples": datamodule.num_train_samples,
            "targets Fro norm": torch.norm(dataset.targets, p="fro").item(),
            "targets inf norm": torch.norm(dataset.targets, p=float('inf')).item()
        })

        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, config: dict, state: dict, logger: logging.Logger):
        train_loss_fn = lambda trainer: trainer.train_evaluator.get_tracked_values()["mse_loss"].current_value
        callbacks_dict["stop_on_zero_train_loss"] = callbacks.StopOnZeroTrainLoss(train_loss_fn=train_loss_fn,
                                                                                  tol=config["stop_on_zero_loss_tol"],
                                                                                  validate_every=config["validate_every"])
        callbacks_dict["terminate_on_nan"] = callbacks.TerminateOnNaN(verify_batches=False)

        if config["max_weights_value"] > 0:
            callbacks_dict["weight_clamp"] = WeightClampCallback(model, max_weights_value=config["max_weights_value"])

    def create_trainer(self, model: LargeScaleTensorCPFactorization, datamodule: DataModule, train_evaluator: TrainEvaluator,
                       val_evaluator: Evaluator, callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "grouprmsprop":
            optimizer = GroupRMSprop(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        loss = self.__get_train_loss(config)
        return SupervisedTrainer(model, optimizer, loss, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                 callback=callback, device=device)

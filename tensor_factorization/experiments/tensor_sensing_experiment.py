import argparse
import logging
import torch
import torch.utils
import torch.utils.data
from collections import OrderedDict
from torch import nn as nn
from typing import Tuple

from common.data.modules import DataModule
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import SupervisedTrainEvaluator, TrainEvaluator, \
    Evaluator, ComposeEvaluator
from common.experiment import ExperimentBase
from common.experiment.experiment_base import ScoreInfo
from common.train import callbacks as callbacks
from common.train.callbacks import Callback
from common.train.optim import GroupRMSprop
from common.train.trainer import Trainer
from tensor_factorization.datasets.tensor_sensing_datamodule import TensorSensingDataModule
from tensor_factorization.evaluation.cp_factorization_evaluator import CPFactorizationEvaluator
from tensor_factorization.evaluation.tensor_evaluator import TensorEvaluator
from tensor_factorization.models.tensor_factorizations import TensorCPFactorization, TensorFactorization
from tensor_factorization.trainers.tensor_sensing_trainer import TensorSensingTrainer


class TensorSensingExperiment(ExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser: argparse.ArgumentParser):
        ExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, required=True, help="Path to the the dataset file")
        parser.add_argument("--load_dataset_to_gpu", action="store_true", help="Stores all dataset on the main GPU (if GPU device is given)")
        parser.add_argument("--num_samples", type=int, default=500, help="Number of train samples to use")
        parser.add_argument("--loss", type=str, default="l2", help="Loss to use. Currently supports: 'l2', 'l1', 'huber'.")
        parser.add_argument("--huber_loss_thresh", type=float, default=1., help="Threshold to use for the Huber loss.")
        parser.add_argument("--lr", type=float, default=1e-2, help="Training learning rate")
        parser.add_argument("--optimizer", type=str, default="grouprmsprop", help="optimizer to use. Supports: 'grouprmsprop', 'adam', 'sgd'")
        parser.add_argument("--momentum", type=float, default=0, help="Momentum for SGD")
        parser.add_argument("--stop_on_zero_loss_tol", type=float, default=1e-8, help="Stops when train loss reaches below this threshold.")
        parser.add_argument("--stop_on_zero_loss_patience", type=int, default=50,
                            help="Number of validated epochs loss has to remain 0 before stopping.")

        parser.add_argument("--init_std", type=float, default=5e-3, help="Init std for gaussian init")
        parser.add_argument("--track_factor_norms", action="store_true",
                            help="Track cp factorization factors, components and factor column norms. Currently only supported for CP factorizaion")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None
        data_module = TensorSensingDataModule(config["dataset_path"], config["num_samples"], load_dataset_to_device=load_dataset_to_device)
        data_module.setup()
        return data_module

    def create_model(self, datamodule: TensorSensingDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        dataset = datamodule.dataset
        return TensorCPFactorization(num_dim_per_mode=list(dataset.target_tensor.size()), init_std=config["init_std"])

    def create_train_and_validation_evaluators(self, model: TensorFactorization, datamodule: TensorSensingDataModule, device,
                                               config: dict, state: dict, logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [metrics.MetricInfo("mse_loss", metrics.MSELoss()),
                                 self.__get_train_loss_metric_info(config)]
        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        dataset = datamodule.dataset
        tensor_evaluator = TensorEvaluator(model, dataset.target_tensor, device=device)

        val_evluators = [tensor_evaluator]

        if config["track_factor_norms"] and isinstance(model, TensorCPFactorization):
            compute_e2e_grad_fn = self.__create_compute_e2e_grad_fn(model, datamodule, device, config, state)
            val_evluators.append(CPFactorizationEvaluator(model, compute_e2e_grad_fn=compute_e2e_grad_fn, device=device))

        return train_evaluator, ComposeEvaluator(val_evluators)

    def __get_train_loss_metric_info(self, config: dict):
        if config["loss"] == "l2":
            return metrics.MetricInfo("l2_loss", metrics.MSELoss())
        elif config["loss"] == "l1":
            return metrics.MetricInfo("l1_loss", metrics.L1Loss())
        elif config["loss"] == "huber":
            return metrics.MetricInfo("huber_loss", metrics.SmoothL1Loss(beta=config["huber_loss_thresh"]))
        else:
            raise ValueError(f"Unsupported loss type: '{config['loss']}'")

    def __get_train_loss(self, config: dict):
        if config["loss"] == "l2":
            return nn.MSELoss()
        elif config["loss"] == "l1":
            return nn.L1Loss()
        elif config["loss"] == "huber":
            return nn.SmoothL1Loss(beta=config["huber_loss_thresh"])
        else:
            raise ValueError(f"Unsupported loss type: '{config['loss']}'")

    def __create_compute_e2e_grad_fn(self, model: TensorFactorization, datamodule: TensorSensingDataModule, device, config, state):
        loss_fn = self.__get_train_loss(config)
        dataset = datamodule.dataset
        X, y = dataset.sensing_tensors[datamodule.train_indices], dataset.targets[datamodule.train_indices]
        X = X.to(device)
        y = y.to(device)

        def compute_e2e_grad():
            with torch.enable_grad():
                end_to_end_tensor = model.compute_tensor()
                end_to_end_tensor.retain_grad()
                y_pred = (X * end_to_end_tensor.unsqueeze(0)).view(y.size(0), -1).sum(dim=1)

                loss = loss_fn(y_pred, y)
                loss.backward()

                e2e_grad = end_to_end_tensor.grad
                model.zero_grad()
                return e2e_grad

        return compute_e2e_grad

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="normalized_reconstruction_error", is_train_metric=False, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: TensorSensingDataModule, config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        dataset = datamodule.dataset
        additional_metadata.update({
            "target tensor num entries": len(dataset),
            "target tensor Fro norm": torch.norm(dataset.target_tensor, p="fro").item(),
            "target tensor inf norm": torch.norm(dataset.target_tensor, p=float('inf')).item()
        })

        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, config: dict, state: dict, logger: logging.Logger):
        train_loss_fn = lambda trainer: trainer.train_evaluator.get_tracked_values()["mse_loss"].current_value
        callbacks_dict["stop_on_zero_train_loss"] = callbacks.StopOnZeroTrainLoss(train_loss_fn=train_loss_fn,
                                                                                  tol=config["stop_on_zero_loss_tol"],
                                                                                  validate_every=config["validate_every"],
                                                                                  patience=config["stop_on_zero_loss_patience"])
        callbacks_dict["terminate_on_nan"] = callbacks.TerminateOnNaN(verify_batches=False)

    def create_trainer(self, model: TensorFactorization, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "grouprmsprop":
            optimizer = GroupRMSprop(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        loss = self.__get_train_loss(config)
        return TensorSensingTrainer(model, optimizer, loss, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                    callback=callback, device=device)

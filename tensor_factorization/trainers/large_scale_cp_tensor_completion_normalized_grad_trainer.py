import common.utils.module as module_utils
from common.evaluation.evaluators import VoidEvaluator
from common.train.trainer import Trainer
from tensor_factorization.models.tensor_factorizations import LargeScaleTensorCPFactorization


class LargeScaleCPTensorCompletionNormalizedGradTrainer(Trainer):
    """
    CP factorization tensor completion trainer which computes an adaptive learning rate calibration inside the gradient computation
    in order to improve numerical stability. Currently supports only factorizations with equal dimensions in each mode.
    """

    def __init__(self, model: LargeScaleTensorCPFactorization, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(),
                 callback=None, device=module_utils.get_device()):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn

    def batch_update(self, batch_num, batch):
        alpha = 3
        beta = 1e-2
        eps = 1e-6

        self.optimizer.zero_grad()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # save parameter's initial state
        initial_parameters = self.model.factors.clone().detach()

        # FIRST PASS

        # forward first time
        y_pred = self.model(x)
        correct_predictions = y_pred.detach()  # save aside because we modify y_pred in second pass

        # retain the predictions grad
        y_pred.retain_grad()

        loss = self.loss_fn(y_pred, y)
        loss.backward()

        # save the predictions grad
        saved_predictions_grad = y_pred.grad.clone().detach()

        # normalize parameters
        params_ = self.model.factors.permute(1, 0, 2)

        params_norms = params_.norm(dim=0)
        normalized_factors = (params_ / params_norms).permute(1, 0, 2) * alpha

        self.model.factors.data = normalized_factors

        ## SECOND PASS

        # forward again
        self.optimizer.zero_grad()
        y_pred = self.model(x)

        # backward again from predictions and specify gradient=...
        y_pred.backward(gradient=saved_predictions_grad)

        # restore original parameters
        self.model.factors.data = initial_parameters

        # rescale gradient result
        max_norm = params_norms.max().item()

        c = (max_norm * beta + eps)
        rescaled_params_norms = params_norms / c
        scale_factor = rescaled_params_norms.prod(dim=0) / rescaled_params_norms

        self.model.factors.grad.data = (self.model.factors.grad.data.permute(1, 0, 2) * scale_factor).permute(1, 0, 2)

        # take a step
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "y_pred": correct_predictions,
            "y": y
        }

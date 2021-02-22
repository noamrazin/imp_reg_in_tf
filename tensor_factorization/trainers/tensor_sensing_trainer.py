import common.utils.module as module_utils
from common.evaluation.evaluators import VoidEvaluator
from common.train.trainer import Trainer
from tensor_factorization.models.tensor_factorizations import TensorFactorization


class TensorSensingTrainer(Trainer):

    def __init__(self, model: TensorFactorization, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=module_utils.get_device()):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn

    def batch_update(self, batch_num, batch):
        self.optimizer.zero_grad()

        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        end_to_end_tensor = self.model.compute_tensor()
        y_pred = (X * end_to_end_tensor.unsqueeze(0)).view(y.size(0), -1).sum(dim=1)

        loss = self.loss_fn(y_pred, y)
        loss.backward()

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y
        }

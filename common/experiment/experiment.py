import json
from abc import ABC, abstractmethod


class ExperimentResult:
    """
    Result object of an Experiment. Contains a score for the fitted model and also additional metadata.
    """

    def __init__(self, score: float, score_name: str, score_epoch: int = -1, additional_metadata: dict = None):
        self.score = score
        self.score_name = score_name
        self.score_epoch = score_epoch
        self.additional_metadata = additional_metadata if additional_metadata is not None else {}
        self.additional_values = {}

    def __str__(self):
        exp_result_str = f"Score name: {self.score_name}\nScore value: {self.score:.3f}\n"
        if self.score_epoch != -1:
            exp_result_str += f"Score epoch: {self.score_epoch}\n"
        exp_result_str += f"Additional metadata: {json.dumps(self.additional_metadata, indent=2)}"
        return exp_result_str


class Experiment(ABC):
    """
    Abstract experiment class. Wraps a model and trainer to create an abstraction for experiment running.
    """

    @abstractmethod
    def run(self, config: dict, context: dict = None) -> ExperimentResult:
        """
        Runs the experiment with the given configuration. Usually fits a model and returns an ExperimentResult object with the score for the
        experiment/model, the larger the better, and any additional metadata. An example for a score is returning the negative validation loss.
        :param config: configurations dictionary for the experiment
        :param context: optional context dictionary with additional information (e.g. can contain an ExperimentsPlan configuration)
        """
        raise NotImplementedError

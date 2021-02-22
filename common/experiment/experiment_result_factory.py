from typing import Tuple, Dict

import numpy as np

from .experiment import ExperimentResult
from ..train.fit_output import FitOutput
from ..train.tracked_value import TrackedValue


class ExperimentResultFactory:
    """
    Static factory for creating ExperimentResult objects from the returned FitOutput of a Trainer.
    """

    @staticmethod
    def create_from_best_metric_score(metric_name: str, fit_output: FitOutput, largest=True, is_train_metric: bool = False,
                                      additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the best score value of the given metric from all epochs.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata to populate the experiment result with.
        :return: ExperimentResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values
        relevant_tracked_value = tracked_values[metric_name]

        best_score, best_score_epoch = ExperimentResultFactory.__get_best_score_and_epoch_of_tracked_value(relevant_tracked_value, largest=largest)

        experiment_result_metadata = ExperimentResultFactory.__create_additional_metadata(fit_output, best_score_epoch)
        if additional_metadata:
            experiment_result_metadata.update(additional_metadata)

        return ExperimentResult(best_score, metric_name, best_score_epoch, experiment_result_metadata)

    @staticmethod
    def create_from_last_metric_score(metric_name: str, fit_output: FitOutput, largest=True, is_train_metric: bool = False,
                                      additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the last score value of the given metric from the last training epoch.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata to populate the experiment result with.
        :return: ExperimentResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values

        relevant_tracked_value = tracked_values[metric_name]
        score = relevant_tracked_value.current_value
        if score is None:
            score = -np.inf if largest else np.inf

        score_epoch = relevant_tracked_value.epoch_last_updated

        experiment_result_metadata = ExperimentResultFactory.__create_additional_metadata(fit_output)
        if additional_metadata:
            experiment_result_metadata.update(additional_metadata)

        return ExperimentResult(score, metric_name, score_epoch, experiment_result_metadata)

    @staticmethod
    def create_from_best_metric_with_prefix_score(metric_name_prefix: str, fit_output: FitOutput, largest=True,
                                                  is_train_metric: bool = False, additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the best score value of the out of all the metrics that start with the given prefix
        from all of the training epochs.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata to populate the experiment result with.
        :return: ExperimentResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values
        best_score, metric_name, best_score_epoch = ExperimentResultFactory.__get_best_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                         tracked_values,
                                                                                                                         largest=largest)

        experiment_result_metadata = ExperimentResultFactory.__create_additional_metadata(fit_output, best_score_epoch)
        if additional_metadata:
            experiment_result_metadata.update(additional_metadata)

        return ExperimentResult(best_score, metric_name, best_score_epoch, experiment_result_metadata)

    @staticmethod
    def create_from_last_metric_with_prefix_score(metric_name_prefix: str, fit_output: FitOutput, largest=True,
                                                  is_train_metric: bool = False, additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the best last score value of the of the metrics that start with the given prefix.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata to populate the experiment result with.
        :return: ExperimentResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values
        score, metric_name, score_epoch = ExperimentResultFactory.__get_best_last_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                    tracked_values,
                                                                                                                    largest=largest)

        experiment_result_metadata = ExperimentResultFactory.__create_additional_metadata(fit_output)
        if additional_metadata:
            experiment_result_metadata.update(additional_metadata)

        return ExperimentResult(score, metric_name, score_epoch, experiment_result_metadata)

    @staticmethod
    def __create_additional_metadata(fit_output: FitOutput, score_epoch: int = -1) -> dict:
        train_tracked_values = fit_output.train_tracked_values
        val_tracked_values = fit_output.val_tracked_values
        other_tracked_values = fit_output.value_store.tracked_values

        additional_metadata = {}
        if score_epoch != -1:
            best_score_epoch_tracked_values = {}

            best_score_epoch_tracked_values.update({name: tracked_value.epoch_values[tracked_value.epochs_with_values.index(score_epoch)]
                                                    for name, tracked_value in train_tracked_values.items()})
            best_score_epoch_tracked_values.update({name: tracked_value.epoch_values[tracked_value.epochs_with_values.index(score_epoch)]
                                                    for name, tracked_value in val_tracked_values.items()})
            best_score_epoch_tracked_values.update({name: tracked_value.epoch_values[tracked_value.epochs_with_values.index(score_epoch)]
                                                    for name, tracked_value in other_tracked_values.items()})

            additional_metadata["Score epoch tracked values"] = best_score_epoch_tracked_values
        else:
            last_score_epoch_tracked_values = {}

            last_score_epoch_tracked_values.update({name: tracked_value.current_value for name, tracked_value in train_tracked_values.items()})
            last_score_epoch_tracked_values.update({name: tracked_value.current_value for name, tracked_value in val_tracked_values.items()})
            last_score_epoch_tracked_values.update({name: tracked_value.current_value for name, tracked_value in other_tracked_values.items()})

            additional_metadata["Last updated epoch tracked values"] = last_score_epoch_tracked_values

        if fit_output.exception_occured():
            additional_metadata["Exception"] = str(fit_output.exception)

        return additional_metadata

    @staticmethod
    def __get_best_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                     tracked_values: Dict[str, TrackedValue],
                                                     largest: bool = True) -> Tuple[float, str, int]:
        best_scores = []
        names = []
        best_score_epochs = []
        for name, tracked_value in tracked_values.items():
            if name.startswith(metric_name_prefix):
                best_score, best_score_epoch = ExperimentResultFactory.__get_best_score_and_epoch_of_tracked_value(tracked_value, largest=largest)

                best_scores.append(best_score)
                names.append(name)
                best_score_epochs.append(best_score_epoch)

        index_of_max = np.argmax(best_scores)
        return best_scores[index_of_max], names[index_of_max], best_score_epochs[index_of_max]

    @staticmethod
    def __get_best_last_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                          tracked_values: Dict[str, TrackedValue],
                                                          largest: bool = True) -> Tuple[float, str, int]:
        scores = []
        names = []
        score_epochs = []
        for name, tracked_value in tracked_values.items():
            if name.startswith(metric_name_prefix):
                score = tracked_value.current_value
                if score is None:
                    score = -np.inf if largest else np.inf

                score_epoch = tracked_value.epoch_last_updated

                scores.append(score)
                names.append(name)
                score_epochs.append(score_epoch)

        index_of_max = np.argmax(scores)
        return scores[index_of_max], names[index_of_max], score_epochs[index_of_max]

    @staticmethod
    def __get_best_score_and_epoch_of_tracked_value(tracked_value, largest: bool = True):
        if len(tracked_value.epoch_values) == 0:
            if tracked_value.current_value is None:
                best_score = -np.inf if largest else np.inf
                best_score_epoch = -1
            else:
                best_score = tracked_value.current_value
                best_score_epoch = tracked_value.epoch_last_updated

            return best_score, best_score_epoch

        get_best_score_index_fn = np.argmax if largest else np.argmin

        index_of_best_score = get_best_score_index_fn(tracked_value.epoch_values)
        best_score = tracked_value.epoch_values[index_of_best_score]
        best_score_epoch = tracked_value.epochs_with_values[index_of_best_score]
        return best_score, best_score_epoch

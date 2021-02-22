import itertools
import json
import logging

import numpy as np

from ...experiment import ExperimentResult, Experiment


def _create_is_better_fn(largest: bool = True):
    def is_better_fn(score1, score2):
        return score1 > score2 if largest else score1 < score2

    return is_better_fn


class TuneResult:
    """
    Result of a hyperparameter tuning process.
    """

    def __init__(self, largest: bool = True):
        """
        :param largest: whether larger score is better
        """
        self.largest = largest

        self.is_better_fn = _create_is_better_fn(largest)
        self.config_results = []
        self.best_config_result = ConfigResult({}, largest=largest)

    def add_config_result(self, config_result):
        self.config_results.append(config_result)
        if self.is_better_fn(config_result.get_score(), self.best_config_result.get_score()):
            self.best_config_result = config_result


class ConfigResult:
    """
    Result for a certain hyperparameter configuration. Wraps multiple ExperimentResults.
    """

    def __init__(self, config: dict, score_reduction: str = "mean", largest: bool = True):
        """
        :param config: configuration dictionary.
        :param score_reduction: determines score reduction method. Supports: 'mean', 'media', 'max', 'min'/
        :param largest: whether larger score is better.
        """
        self.config = config
        self.score_reduction = score_reduction.lower()
        self.score_reduce_fn = self.__get_score_reduce_fn(self.score_reduction)
        self.largest = largest
        self.is_better_fn = _create_is_better_fn(largest)
        self.worst_score = -np.inf if largest else np.inf

        self.experiment_results = []
        self.best_experiment_result = ExperimentResult(self.worst_score, "")

    @staticmethod
    def __get_score_reduce_fn(score_reduction: str):
        if score_reduction == "mean":
            return np.mean
        elif score_reduction == "median":
            return np.median
        elif score_reduction == "max":
            return np.max
        elif score_reduction == "min":
            return np.min

        raise ValueError(f"Unsupported score reduction type: {score_reduction}. Supported types are: 'mean', 'median', 'max', 'min'.")

    def add_experiment_result(self, experiment_result: ExperimentResult):
        self.experiment_results.append(experiment_result)
        if self.is_better_fn(experiment_result.score, self.best_experiment_result.score):
            self.best_experiment_result = experiment_result

    def get_score(self):
        return self.score_reduce_fn([experiment_result.score for experiment_result in self.experiment_results]) \
            if len(self.experiment_results) != 0 else self.worst_score

    def get_score_std(self):
        return np.std([experiment_result.score for experiment_result in self.experiment_results]) if len(self.experiment_results) != 0 else 0


class Tuner:
    """
    Tunes hyperparameters for a given experiment.
    """

    def __init__(self, experiment: Experiment, context: dict = None, largest: bool = True, logger=None):
        """
        :param experiment: Experiment to run
        :param context: optional context dictionary of the experiment (e.g. can contain an ExperimentsPlan configuration)
        :param largest: whether for the score returned in the ExperimentResult larger is better.
        :param logger: logger to use for logging tuning related logs.
        """
        self.experiment = experiment
        self.context = context
        self.largest = largest
        self.largest_log_str = "maximal" if largest else "minimal"
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def random_search(self, base_config, config_samplers, n_iter=10, repetitions=1, score_reduction="mean"):
        """
        Runs random search hyperparameter tuning.
        :param base_config: base dictionary of configurations for the experiments.
        :param config_samplers: dictionary of configuration name to a callable that samples a value.
        :param n_iter: number of configuration settings that are sampled.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        self.logger.info(f"Starting random search for {n_iter} iterations.\n"
                         f"Base config:\n{json.dumps(base_config, indent=2)}\n"
                         f"Config with samplers: {config_samplers.keys()}")
        config_seq = []
        for i in range(n_iter):
            config = base_config.copy()
            for config_name, sampler in config_samplers.items():
                config[config_name] = sampler()

            config_seq.append(config)

        tune_result = self.__run_search(config_seq, skip=0, repetitions=repetitions, score_reduction=score_reduction)
        self.logger.info(f"Finished random search.\n"
                         f"Best config:\n{json.dumps(tune_result.best_config_result.config, indent=2)}\n"
                         f"Best ({self.largest_log_str}) config score value: {tune_result.best_config_result.get_score()}\n"
                         f"Best config score std: {tune_result.best_config_result.get_score_std()}\n"
                         f"Best config best experiment result:\n{tune_result.best_config_result.best_experiment_result}")
        return tune_result

    def grid_search(self, base_config, config_options, skip=0, repetitions=1, score_reduction="mean"):
        """
        Runs grid search hyperparameter tuning.
        :param base_config: base configuration dictionary for the experiments.
        :param config_options: dictionary of configuration name to a sequence of values.
        :param skip: number of iterations from the start to skip.
        :param repetitions: number of times to repeat the experiment per parameter configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        config_names = config_options.keys()
        config_values = [config_options[config_name] for config_name in config_names]
        n_iter = np.prod([len(values) for values in config_values])

        self.logger.info(f"Starting grid search for {n_iter} iterations.\n"
                         f"Base config:\n{json.dumps(base_config, indent=2)}\n"
                         f"Config options: {json.dumps(config_options, indent=2)}")
        config_seq = []
        all_options_iterator = itertools.product(*config_values)
        for i, values in enumerate(all_options_iterator):
            config = base_config.copy()
            for name, value in zip(config_names, values):
                config[name] = value

            config_seq.append(config)

        tune_result = self.__run_search(config_seq, skip=skip, repetitions=repetitions, score_reduction=score_reduction)

        self.logger.info(f"Finished grid search.\n"
                         f"Best config:\n{json.dumps(tune_result.best_config_result.config, indent=2)}\n"
                         f"Best ({self.largest_log_str}) config score value: {tune_result.best_config_result.get_score()}\n"
                         f"Best config Score std: {tune_result.best_config_result.get_score_std()}\n"
                         f"Best config best experiment result:\n{tune_result.best_config_result.best_experiment_result}")

        return tune_result

    def preset_options_search(self, configs_seq, skip=0, repetitions=1, score_reduction="mean"):
        """
        Runs hyperparameter search for the given preset configuration.
        :param configs_seq: sequence of configuration dictionaries to try.
        :param skip: number of iterations from the start to skip.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        self.logger.info(f"Starting preset options search for {len(configs_seq)} options.")
        tune_result = self.__run_search(configs_seq, skip=skip, repetitions=repetitions, score_reduction=score_reduction)
        self.logger.info(f"Finished preset options search.\n"
                         f"Best config:\n{json.dumps(tune_result.best_config_result.config, indent=2)}\n"
                         f"Best ({self.largest_log_str}) config score value: {tune_result.best_config_result.get_score()}\n"
                         f"Best config score std: {tune_result.best_config_result.get_score_std()}\n"
                         f"Best config best experiment result:\n{tune_result.best_config_result.best_experiment_result}")
        return tune_result

    def __run_search(self, configs_seq, skip=0, repetitions=1, score_reduction="mean"):
        """
        Runs hyperparameter search on the given configurations.
        :param configs_seq: sequence of configuration dictionaries.
        :param skip: number of iterations from the start to skip.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        n_iter = len(configs_seq)
        tune_result = TuneResult(largest=self.largest)

        for i, config in enumerate(itertools.islice(configs_seq, skip, None)):
            config = config.copy()
            config_result = ConfigResult(config, score_reduction=score_reduction, largest=self.largest)

            self.logger.info(f"Starting experiment for config {skip + i + 1}/{n_iter}:\n{json.dumps(config, indent=2)}")
            for r in range(repetitions):
                self.logger.info(f"Starting repetition {r + 1}/{repetitions} for experiment {skip + i + 1}/{n_iter}.")
                experiment_result = self.experiment.run(config, context=self.context)
                config_result.add_experiment_result(experiment_result)
                self.logger.info(f"Finished repetition {r + 1}/{repetitions} for experiment {skip + i + 1}/{n_iter}:\n{experiment_result}")

            self.logger.info(f"Finished experiment for config {skip + i + 1}/{n_iter}\n"
                             f"Config score value: {config_result.get_score()}\n"
                             f"Config score std: {config_result.get_score_std()}\n"
                             f"Config experiment result with best ({self.largest_log_str}) score:\n{config_result.best_experiment_result}")
            tune_result.add_config_result(config_result)

        return tune_result

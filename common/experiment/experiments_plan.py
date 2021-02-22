from __future__ import annotations

import itertools
import json
from typing import List


class ExperimentsPlan:
    """
    Experiment plan configuration object. Allows loading and parsing of an experiments plan JSON configuration.
    """

    @staticmethod
    def load_from(plan_config_path: str) -> ExperimentsPlan:
        with open(plan_config_path) as f:
            raw_plan_config = json.load(f)
            return ExperimentsPlan(raw_plan_config)

    def __init__(self, raw_plan_config: dict):
        self.raw_plan_config = raw_plan_config

        self.name = self.raw_plan_config["name"] if "name" in self.raw_plan_config else ""
        self.description = self.raw_plan_config["description"] if "description" in self.raw_plan_config else ""
        self.repetitions = self.raw_plan_config["repetitions"] if "repetitions" in self.raw_plan_config else 1
        self.largest = self.raw_plan_config["largest"] if "largest" in self.raw_plan_config else True
        self.experiments_configurations_seq = self.__extract_experiments_configurations()

    def __extract_experiments_configurations(self) -> List[dict]:
        experiments_configurations_seq = []

        for configuration_def in self.raw_plan_config["configurations"]:
            base_config = configuration_def["base_config"]
            options = configuration_def["options"] if "options" in configuration_def else {}

            experiments_configurations = self.__create_experiment_configurations_for_base_config(base_config, options)
            experiments_configurations_seq.extend(experiments_configurations)

        return experiments_configurations_seq

    def __create_experiment_configurations_for_base_config(self, base_config: dict, options: dict) -> List[dict]:
        if len(options) == 0:
            config = base_config.copy()
            self.__format_experiment_name(config)
            return [config]

        field_names = options.keys()
        config_values = [options[field_name] for field_name in field_names]

        experiments_configurations = []
        all_options_iterator = itertools.product(*config_values)
        for values in all_options_iterator:
            config = base_config.copy()
            for field_name, config_value in zip(field_names, values):
                config[field_name] = config_value

            self.__format_experiment_name(config)
            experiments_configurations.append(config)

        return experiments_configurations

    def __format_experiment_name(self, config: dict):
        experiment_name = config.get("experiment_name")
        if not experiment_name:
            return

        formatted_experiment_name = experiment_name.format(**config)
        formatted_experiment_name = formatted_experiment_name.replace(".", "-")  # Replace periods with dash to avoid filename issues
        config["experiment_name"] = formatted_experiment_name

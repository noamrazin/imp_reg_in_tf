import argparse

import common.utils.logging as logging_utils
from tensor_factorization.experiments.tensor_sensing_experiment import TensorSensingExperiment


def main():
    parser = argparse.ArgumentParser()
    TensorSensingExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = TensorSensingExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()

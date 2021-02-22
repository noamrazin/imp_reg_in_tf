import argparse

import common.utils.logging as logging_utils
from tensor_factorization.experiments.large_scale_tensor_completion_experiment import LargeScaleTensorCompletionExperiment


def main():
    parser = argparse.ArgumentParser()
    LargeScaleTensorCompletionExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = LargeScaleTensorCompletionExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()

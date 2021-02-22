import argparse
import os
from datetime import datetime

import torch
import torch.utils.data
import torchvision

import common.utils.logging as logging_utils
from tensor_factorization.datasets.large_scale_tensor_completion_dataset import LargeScaleTensorCompletionDataset


class GrayscaleToFlattenedModeIndices:

    def __init__(self, num_opts: int = 2):
        """
        :param num_opts: Number of grayscale values
        """
        self.num_opts = num_opts

    def __call__(self, sample):
        sample *= (self.num_opts - 1)
        sample.round_()
        sample = sample.flatten()
        return sample.long()


def __get_dataset(dataset_name: str, train: bool, num_grayscale_opts: int):
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST("./data", train=train, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              GrayscaleToFlattenedModeIndices(num_opts=num_grayscale_opts)]))

    elif dataset_name == "fmnist":
        return torchvision.datasets.FashionMNIST("./data", train=train, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     GrayscaleToFlattenedModeIndices(num_opts=num_grayscale_opts)]))
    else:
        raise ValueError(f"Unsupported dataset name: '{dataset_name}'")


def import_dataset(dataset_name: str, num_grayscale_opts: int = 2):
    train_dataset = __get_dataset(dataset_name, train=True, num_grayscale_opts=num_grayscale_opts)
    test_dataset = __get_dataset(dataset_name, train=False, num_grayscale_opts=num_grayscale_opts)
    return train_dataset, test_dataset


def __create_rand_positive_and_negative_samples(num_opts: int, num_pos_samples: int, num_neg_samples: int, dim: int):
    positive_samples = torch.randint(0, num_opts, size=(num_pos_samples, dim)).long()
    negative_samples = torch.randint(0, num_opts, size=(num_neg_samples, dim)).long()
    return positive_samples, negative_samples


def __create_positive_and_negative_samples(arguments):
    train_dataset, test_dataset = import_dataset(arguments.dataset_name, arguments.num_grayscale_opts)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    train_negative_samples_seq, train_positive_samples_seq = select_positive_and_negative_samples(arguments, train_data_loader)
    test_negative_samples_seq, test_positive_samples_seq = select_positive_and_negative_samples(arguments, test_data_loader)

    train_positive_samples = torch.stack(train_positive_samples_seq)
    train_negative_samples = torch.stack(train_negative_samples_seq)

    test_positive_samples = torch.stack(test_positive_samples_seq)
    test_negative_samples = torch.stack(test_negative_samples_seq)

    return train_positive_samples, train_negative_samples, test_positive_samples, test_negative_samples


def select_positive_and_negative_samples(arguments, data_loader):
    positive_samples_seq = []
    negative_samples_seq = []

    for image, label in iter(data_loader):
        if label == arguments.positive_label:
            positive_samples_seq.append(image.squeeze())
        else:
            negative_samples_seq.append(image.squeeze())

    return negative_samples_seq, positive_samples_seq


def __create_tensor_indices_and_targets(train_positive_samples: torch.Tensor, train_negative_samples: torch.Tensor,
                                        test_positive_samples: torch.Tensor, test_negative_samples: torch.Tensor,
                                        arguments):
    if arguments.rand_data:
        train_positive_samples, train_negative_samples = __create_rand_positive_and_negative_samples(
            num_opts=arguments.num_grayscale_opts,
            num_pos_samples=len(train_positive_samples),
            num_neg_samples=len(train_negative_samples),
            dim=train_positive_samples.shape[1])

        test_positive_samples, test_negative_samples = __create_rand_positive_and_negative_samples(
            num_opts=arguments.num_grayscale_opts,
            num_pos_samples=len(test_positive_samples),
            num_neg_samples=len(test_negative_samples),
            dim=test_positive_samples.shape[1])

    train_tensor_indices = torch.cat([train_positive_samples, train_negative_samples])
    test_tensor_indices = torch.cat([test_positive_samples, test_negative_samples])

    train_shuffled_order = torch.randperm(train_tensor_indices.shape[0])
    train_tensor_indices = train_tensor_indices[train_shuffled_order]

    test_shuffled_order = torch.randperm(test_tensor_indices.shape[0])
    test_tensor_indices = test_tensor_indices[test_shuffled_order]

    train_positive_label_values = torch.full((train_positive_samples.shape[0],), fill_value=arguments.positive_label_value, dtype=torch.float)
    train_negative_label_values = torch.full((train_negative_samples.shape[0],), fill_value=arguments.negative_label_value, dtype=torch.float)
    test_positive_label_values = torch.full((test_positive_samples.shape[0],), fill_value=arguments.positive_label_value, dtype=torch.float)
    test_negative_label_values = torch.full((test_negative_samples.shape[0],), fill_value=arguments.negative_label_value, dtype=torch.float)

    train_label_values = torch.cat([train_positive_label_values, train_negative_label_values])
    test_label_values = torch.cat([test_positive_label_values, test_negative_label_values])

    train_label_values = train_label_values[train_shuffled_order]
    test_label_values = test_label_values[test_shuffled_order]

    if arguments.rand_labels:
        random_labels_order = torch.randperm(train_label_values.shape[0])
        train_label_values = train_label_values[random_labels_order]

        random_labels_order = torch.randperm(test_label_values.shape[0])
        test_label_values = test_label_values[random_labels_order]

    all_tensor_indices = torch.cat([train_tensor_indices, test_tensor_indices])
    all_label_values = torch.cat([train_label_values, test_label_values])

    return all_tensor_indices, all_label_values


def __create_rand_data_labels_str(rand_data: bool, rand_labels: bool):
    if not rand_data and not rand_labels:
        return "_"

    rand_data_labels_str = "_rd-"
    if rand_data:
        rand_data_labels_str += "d"

    if rand_labels:
        rand_data_labels_str += "l"

    return rand_data_labels_str + "_"


def create_and_save_dataset(arguments):
    train_positive_samples, train_negative_samples, test_positive_samples, test_negative_samples = __create_positive_and_negative_samples(arguments)

    tensor_indices, targets = __create_tensor_indices_and_targets(train_positive_samples, train_negative_samples,
                                                                  test_positive_samples, test_negative_samples, arguments)

    dataset = LargeScaleTensorCompletionDataset(tensor_indices, targets,
                                                target_tensor_modes_dim=arguments.num_grayscale_opts,
                                                additional_metadata={
                                                    "dataset_name": arguments.dataset_name,
                                                    "num_grayscale_opts": arguments.num_grayscale_opts,
                                                    "positive_label_index": arguments.positive_label,
                                                    "positive_label_value": arguments.positive_label_value,
                                                    "negative_label_value": arguments.negative_label_value,
                                                    "rand_data": arguments.rand_data,
                                                    "rand_labels": arguments.rand_labels
                                                })

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    if arguments.custom_file_name:
        file_name = f"{arguments.custom_file_name}_{now_utc_str}.pt"
    else:
        rand_data_labels_str = __create_rand_data_labels_str(arguments.rand_data, arguments.rand_labels)
        file_name = f"{arguments.dataset_name}{rand_data_labels_str}ord_{tensor_indices.shape[1]}_d_{arguments.num_grayscale_opts}_" \
                    f"s_{tensor_indices.shape[0]}_la_{arguments.positive_label}_" \
                    f"{now_utc_str}"
        file_name = file_name.replace(".", "-") + ".pt"

    output_path = os.path.join(arguments.output_dir, file_name)
    dataset.save(output_path)

    logging_utils.info(f"Created dataset at: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("--output_dir", type=str, default="data/nat", help="Path to the directory to save the target matrix and dataset at")
    p.add_argument("--custom_file_name", type=str, default="", help="Custom file name prefix for the dataset")

    p.add_argument("--dataset_name", type=str, default="mnist", help="Dataset to create a tensor completion dataset for. "
                                                                     "Currently supports: 'mnist' and 'fmnist'")
    p.add_argument("--rand_data", action="store_true", help="Randomize only the data (keep original labels to preserve"
                                                            " exact portions of positive and negative examples of "
                                                            "original data).")
    p.add_argument("--rand_labels", action="store_true", help="Randomize only the labels and keep the samples as is")
    p.add_argument("--positive_label", type=int, default=6, help="Index of positive label")
    p.add_argument("--positive_label_value", type=float, default=2, help="Target value corresponding to a positive label")
    p.add_argument("--negative_label_value", type=float, default=0, help="Target value corresponding to a negative label")
    p.add_argument("--num_grayscale_opts", type=int, default=2, help="Number of grayscale options. Determines the modes dimension size in "
                                                                     "the corresponding tensor.")

    args = p.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_and_save_dataset(args)

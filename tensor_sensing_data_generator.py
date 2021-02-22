import argparse
import itertools
import math
import os
import random
from datetime import datetime

import numpy as np
import torch

import common.utils.logging as logging_utils
import common.utils.tensor as tensor_utils
from tensor_factorization.datasets.tensor_sensing_dataset import TensorSensingDataset


def __set_initial_random_seed(random_seed: int):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def create_tensor_sensing_dataset(target_tensor_cp_rank, target_fro_norm, mode_dim_size, order, super_symmetric_target):
    target_tensor = tensor_utils.create_tensor_with_cp_rank([mode_dim_size] * order, target_tensor_cp_rank,
                                                            fro_norm=target_fro_norm, super_symmetric=super_symmetric_target)
    sensing_tensors = create_sensing_tensors(args, target_tensor)

    dataset = TensorSensingDataset(sensing_tensors, target_tensor, target_tensor_cp_rank)
    return dataset


def create_sensing_tensors(args, target_tensor):
    num_entries = target_tensor.numel()

    if args.task_type == "sensing":
        if args.sensing_tensors_cp_rank <= 0:
            X = torch.randn(args.num_samples, *target_tensor.size()) / math.sqrt(num_entries)
        else:
            X = [tensor_utils.create_tensor_with_cp_rank(target_tensor.size(), args.sensing_tensors_cp_rank, fro_norm=1)
                 for _ in range(args.num_samples)]
            X = torch.stack(X)

    elif args.task_type == "completion":
        num_samples = min(args.num_samples, num_entries)
        all_indices_tensors = __create_all_indices_tensor(target_tensor.size())

        chosen_observed_indices = torch.multinomial(torch.ones(num_entries), num_samples, replacement=False)
        shuffled_observed_indices_tensor = all_indices_tensors[chosen_observed_indices]

        X = torch.zeros(num_samples, *target_tensor.size())
        batch_observed_indices = [range(num_samples)] + [shuffled_observed_indices_tensor[:, i] for i in
                                                         range(shuffled_observed_indices_tensor.shape[1])]
        X[batch_observed_indices] = 1
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}.")

    return X


def __create_all_indices_tensor(mode_dims):
    indices = []
    per_mode_options = [range(dim) for dim in mode_dims]
    for tensor_index in itertools.product(*per_mode_options):
        indices.append(torch.tensor(tensor_index, dtype=torch.long))

    return torch.stack(indices)


def create_and_save_dataset(args):
    dataset = create_tensor_sensing_dataset(args.target_tensor_cp_rank, args.target_fro_norm,
                                            args.mode_dim_size, args.order, args.super_symmetric_target)

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    if args.custom_file_name:
        file_name = f"{args.custom_file_name}_{now_utc_str}.pt"
    else:
        sym_prefix = "" if not args.super_symmetric_target else "sym_"
        sensing_tensors_rank_str = "" if args.sensing_tensors_cp_rank <= 0 else f"_sr{args.sensing_tensors_cp_rank}"
        file_name = f"t{args.task_type[:4]}{sensing_tensors_rank_str}_{sym_prefix}tr_{args.target_tensor_cp_rank}" \
                    f"_fro_{int(args.target_fro_norm)}_order_{args.order}_dim_{args.mode_dim_size}_{now_utc_str}.pt"

    output_path = os.path.join(args.output_dir, file_name)
    dataset.save(output_path)

    logging_utils.info(f"Created {'symmetric ' if args.super_symmetric_target else ''}tensor {args.task_type} dataset at: {output_path}\n"
                       f"Order: {args.order}, Modes dimension: {args.mode_dim_size}, Target CP Rank: {args.target_tensor_cp_rank}, "
                       f"Target Fro Norm: {args.target_fro_norm}, # Samples: {args.num_samples}")

    if args.task_type == "sensing" and args.sensing_tensors_cp_rank > 0:
        logging_utils.info(f"Sensing tensors CP Rank: {args.sensing_tensors_cp_rank}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("--output_dir", type=str, default="data/syn", help="Path to the directory to save the target matrix and dataset at.")
    p.add_argument("--custom_file_name", type=str, default="", help="Custom file name prefix for the dataset.")

    p.add_argument("--task_type", type=str, default="completion", help="Task type. Can be either 'sensing' or 'completion'")
    p.add_argument("--num_samples", type=int, default=10000, help="Number of sensing samples to create.")
    p.add_argument("--target_tensor_cp_rank", type=int, default=5,
                   help="CP rank of the target tensor. If <= 0 then no rank constraint is used (tensor will be generated randomly)")
    p.add_argument("--target_fro_norm", type=float, default=1.0, help="Fro norm of the target tensor. If <=0 will not normalize")
    p.add_argument("--sensing_tensors_cp_rank", type=int, default=-1,
                   help="CP rank of the sensing tensors. If <= 0 then no rank constraint is used (tensors will be generated randomly)")

    p.add_argument("--mode_dim_size", type=int, default=10, help="Number of dimensions per each mode.")
    p.add_argument("--order", type=int, default=4, help="Order of the tensor (number of modes).")
    p.add_argument("--super_symmetric_target", action="store_true", help="Use super symmetric target tensor.")

    args = p.parse_args()

    logging_utils.init_console_logging()
    __set_initial_random_seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_and_save_dataset(args)

import torch

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.data.modules import DataModule
from common.utils.tensor import convert_tensor_to_one_hot
from tensor_factorization.datasets.large_scale_tensor_completion_dataset import LargeScaleTensorCompletionDataset


class LargeScaleTensorCompletionDataModule(DataModule):

    def __init__(self, dataset_path: str, num_train_samples: int = -1, batch_size: int = -1, shuffle_train: bool = False,
                 random_train_test_split: bool = False, precompute_one_hot_repr: bool = True, load_dataset_to_device=None):
        """
        :param dataset_path: path to LargeScaleTensorCompletionDataset file
        :param num_train_samples: number of samples to use for training, if < 0 will use the whole dataset
        :param batch_size: batch size, if <= 0 will use the size of the whole dataset
        :param shuffle_train: shuffle train samples each epoch
        :param random_train_test_split: randomize train test split
        :param precompute_one_hot_repr: If True will precompute all one hot representations of the train and test indices tensors.
        :param load_dataset_to_device: device to load dataset to (default is CPU)
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.random_train_test_split = random_train_test_split
        self.precompute_one_hot_repr = precompute_one_hot_repr
        self.load_dataset_to_device = load_dataset_to_device
        self.dataset = LargeScaleTensorCompletionDataset.load(dataset_path)
        self.num_train_samples = num_train_samples if num_train_samples > 0 else len(self.dataset)

        if self.load_dataset_to_device is not None:
            self.dataset.to_device(self.load_dataset_to_device)

        train_then_test_indices = torch.arange(len(self.dataset)) if not self.random_train_test_split else torch.randperm(len(self.dataset))
        self.train_indices = train_then_test_indices[:self.num_train_samples]
        self.test_indices = train_then_test_indices[self.num_train_samples:]

        self.train_tensor_indices = self.dataset.tensor_indices[self.train_indices]
        self.train_targets = self.dataset.targets[self.train_indices]

        self.test_tensor_indices = self.dataset.tensor_indices[self.test_indices]
        self.test_targets = self.dataset.targets[self.test_indices]

        if self.precompute_one_hot_repr:
            self.train_tensor_indices = convert_tensor_to_one_hot(self.train_tensor_indices, num_options=self.dataset.target_tensor_modes_dim)
            self.test_tensor_indices = convert_tensor_to_one_hot(self.test_tensor_indices, num_options=self.dataset.target_tensor_modes_dim)

    def setup(self):
        # No setup required
        pass

    def train_dataloader(self) -> FastTensorDataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_indices)
        return FastTensorDataLoader(self.train_tensor_indices, self.train_targets, batch_size=batch_size, shuffle=self.shuffle_train)

    def val_dataloader(self) -> FastTensorDataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.test_indices)
        return FastTensorDataLoader(self.test_tensor_indices, self.test_targets, batch_size=batch_size, shuffle=False)

    def test_dataloader(self) -> FastTensorDataLoader:
        return self.val_dataloader()

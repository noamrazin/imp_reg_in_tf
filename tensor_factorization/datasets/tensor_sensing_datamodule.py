import torch.utils.data

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.data.modules import DataModule
from tensor_factorization.datasets.tensor_sensing_dataset import TensorSensingDataset


class TensorSensingDataModule(DataModule):

    def __init__(self, dataset_path: str, num_train_samples: int, batch_size: int = -1, shuffle_train: bool = False,
                 random_train_test_split: bool = False, load_dataset_to_device=None):
        """
        :param dataset_path: path to LargeScaleTensorCompletionDataset file
        :param num_train_samples: number of samples to use for training
        :param batch_size: batch size, if <= 0 will use the size of the whole dataset
        :param shuffle_train: shuffle train samples each epoch
        :param random_train_test_split: randomize train test split
        :param load_dataset_to_device: device to load dataset to (default is CPU)
        """
        self.dataset_path = dataset_path
        self.num_samples = num_train_samples
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.random_train_test_split = random_train_test_split
        self.load_dataset_to_device = load_dataset_to_device
        self.dataset = TensorSensingDataset.load(dataset_path)

        if self.load_dataset_to_device is not None:
            self.dataset.to_device(self.load_dataset_to_device)

        train_then_test_indices = torch.arange(len(self.dataset)) if not self.random_train_test_split else torch.randperm(len(self.dataset))
        self.train_indices = train_then_test_indices[:num_train_samples]
        self.test_indices = train_then_test_indices[num_train_samples:]

    def setup(self):
        # No setup required
        pass

    def train_dataloader(self) -> FastTensorDataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_indices)
        return FastTensorDataLoader(self.dataset.sensing_tensors[self.train_indices], self.dataset.targets[self.train_indices],
                                    batch_size=batch_size, shuffle=self.shuffle_train)

    def val_dataloader(self) -> FastTensorDataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.test_indices)
        return FastTensorDataLoader(self.dataset.sensing_tensors[self.test_indices], self.dataset.targets[self.test_indices],
                                    batch_size=batch_size, shuffle=False)

    def test_dataloader(self) -> FastTensorDataLoader:
        return self.val_dataloader()

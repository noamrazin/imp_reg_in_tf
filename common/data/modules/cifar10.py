import torch.utils.data
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .datamodule import DataModule


class CIFAR10DataModule(DataModule):

    def __init__(self, data_dir: str = './data', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])

    def setup(self):
        self.cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform, download=True)
        self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform, download=True)

        self.input_dims = tuple(self.cifar10_train[0][0].shape)
        self.num_classes = len(self.cifar10_train.classes)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False)

import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST

from .datamodule import DataModule


class MNISTDataModule(DataModule):

    def __init__(self, data_dir: str = './data', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

    def setup(self):
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform, download=True)
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform, download=True)

        self.input_dims = tuple(self.mnist_train[0][0].shape)
        self.num_classes = len(self.mnist_train.classes)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

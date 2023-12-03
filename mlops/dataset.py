import os

import lightning.pytorch as pl
import torch
import torchvision
from torchvision import transforms


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        # Путь для папки с данными
        DATA_PATH = os.path.dirname(os.path.abspath(__name__))
        DATA_PATH = os.path.join(DATA_PATH, "data")

        # Папка с фотографиями для теста
        TEST_DIR = os.path.join(DATA_PATH, "test")

        transform = transforms.ToTensor()

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=True, download=False, transform=transform
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=False, download=False, transform=transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=False, download=False, transform=transform
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def teardown(self, stage: str) -> None:
        pass

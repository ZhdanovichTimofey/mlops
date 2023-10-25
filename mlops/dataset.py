import os

import lightning.pytorch as pl
import torch
import torchvision
from torchvision import transforms


class DataLoader:
    def __init__(self, batch_size):
        # Путь для папки с данными
        DATA_PATH = os.path.dirname(os.path.abspath(__name__))
        DATA_PATH = os.path.join(DATA_PATH, "data")

        # Папка с фотографиями для теста
        TEST_DIR = os.path.join(DATA_PATH, "test")

        transform = transforms.ToTensor()

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=True, download=True, transform=transform
        )

        # Загрузим тестовую часть данных
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=False, download=True, transform=transform
        )

        self.batch_size = batch_size

        self.train_batch_gen = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_batch_gen = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.Loader = DataLoader(batch_size)

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = self.Loader.train_dataset
        self.val_dataset = self.Loader.test_dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.Loader.train_batch_gen

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.Loader.test_batch_gen

    def teardown(self, stage: str) -> None:
        pass

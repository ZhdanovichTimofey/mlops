import os

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

        train_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=True, download=True, transform=transform
        )

        # Загрузим тестовую часть данных
        test_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DIR, train=False, download=True, transform=transform
        )

        self.batch_size = batch_size

        self.train_batch_gen = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_batch_gen = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

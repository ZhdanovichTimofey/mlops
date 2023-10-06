import os

import torch
import torchvision
from torchvision import transforms


# Путь для папки с данными
DATA_PATH = os.path.dirname(os.path.abspath(__name__))
DATA_PATH = os.path.join(DATA_PATH, "dataset/data")

# Папка со всеми фотографиями / папка с фотографиями для тренировки
TRAIN_DIR = os.path.join(DATA_PATH, "train")

# Папка с фотографиями для теста
TEST_DIR = os.path.join(DATA_PATH, "test")

dataset = torchvision.datasets.CIFAR10(
    root=TRAIN_DIR, train=True, download=True, transform=transforms.ToTensor()
)

train_dataset = torchvision.datasets.CIFAR10(
    root=TEST_DIR, train=True, download=True, transform=transforms.ToTensor()
)

# Загрузим тестеовую часть данных
test_dataset = torchvision.datasets.CIFAR10(
    root=TEST_DIR, train=False, download=True, transform=transforms.ToTensor()
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

batch_size = 64

train_batch_gen = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_batch_gen = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

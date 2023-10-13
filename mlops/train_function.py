import torch
from torch import nn

from .dataset import DataLoader as data
from .model import SimpleConvNet


def train(num_epochs, learning_rate, batch_size):
    device = "cpu"
    model = SimpleConvNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch = data(batch_size).train_batch_gen

    for epoch in range(num_epochs):

        model.train(True)

        # На каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in batch:
            # Обучаемся на батче (одна "итерация" обучения нейросети)

            # X_batch = transform_train(X_batch)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Логиты на выходе модели
            logits = model(X_batch)

            # Подсчитываем лосс
            loss = criterion(logits, y_batch.long().to(device))

            # Обратный проход
            loss.backward()
            # Шаг градиента
            optimizer.step()
            # Зануляем градиенты
            optimizer.zero_grad()

    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    import dataset.dataset as data
    import model
    import torch
    from torch import nn

    def train(model, criterion, optimizer, train_batch_gen, num_epochs=50):
        """
        Функция для обучения модели и вывода лосса и метрики во время обучения.

        :param model: обучаемая модель
        :param criterion: функция потерь
        :param optimizer: метод оптимизации
        :param train_batch_gen: генератор батчей для обучения
        :param num_epochs: количество эпох

        :return: обученная модель
        """

        for epoch in range(num_epochs):

            model.train(True)

            # На каждой "эпохе" делаем полный проход по данным
            for X_batch, y_batch in train_batch_gen:
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

        return model

    device = "cpu"
    model = model.SimpleConvNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch = data.train_batch_gen
    model = train(model, criterion, optimizer, batch, num_epochs=10)
    torch.save(model.state_dict(), "model.pt")

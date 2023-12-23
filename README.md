Решается задача классификации на датасете CIFAR10

Для запуска цикла обучения: python commands.py action=train
Для запуска теста: python commands.py action=infer

В версии c тэгом hw2 выполнено дз_1 и первые 3 пункта дз_2, infer реализован с помощью lightning, результаты логгируются на mlflow server

### Отчет по дз 3:
#### Системная конфигурация:

- OS: 13.6.1 (22G313)
- CPU: Apple M2
- Общее количество ядер (vCPU):	8 (4 производительности и 4 эффективности)
- Память (RAM):	24 ГБ

#### Описание решаемой задачи:

Решается задача классификации на датасете CIFAR10

#### Структура model_repository:

    model_repository
    └── simple-onnx-model
        ├── 1
        │   └── model.onnx
        └── config.pbtxt

#### Конвертация модели в ONNX:

Конвертация происходит после обучения модели, запускается командой: python commands.py action=train

#### Оптимизация параметров для config.pbtxt:

##### Метрики до оптимизации:

Inferences/Second vs. Client Average Batch Latency

Concurrency: 256, throughput: 3430.21 infer/sec, latency 74610 usec

##### Метрики после оптимизации dynamic batching(max_queue_delay_microseconds: 500):

Concurrency: 256, throughput: 7611.56 infer/sec, latency 33554 usec

Начинал поиск оптимума с 10000, при увеличении параметра результаты ухудшались. Далее стал уменьшать значение в 2 раза, каждый раз получал улучшение. При слишком маленьких значениях опять пошло ухудшение и latency и throughput, таким образом получил, что 500 - оптимальное значение.

##### Метрики после оптимизации dynamic batching и instance_group(count: 1):

Concurrency: 256, throughput: 7611.56 infer/sec, latency 33554 usec

При попытке увеличить число инстансов модели падал throughput и увеличивалось latency

##### Клиент и тесты:

Клиент написан в файле client.py, тесты захардкожены там же

##### Веса модели:

Веса модели лежат в dvc.

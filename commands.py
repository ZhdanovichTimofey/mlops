import fire
from mlops import infer_function, train_function


def train():
    train_function.train()


def infer():
    infer_function.infer()


if __name__ == "__main__":
    fire.Fire()

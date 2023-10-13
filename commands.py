import fire
from mlops import infer, train


def train_():
    train.train()


def infer_():
    infer.infer()


if __name__ == "__main__":
    fire.Fire()

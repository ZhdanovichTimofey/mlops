import hydra
from config import TrainParams
from hydra.core.config_store import ConfigStore
from mlops import infer, train


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainParams)


@hydra.main(config_path="conf", config_name="base_config", version_base="1.3")
def main(cfg: TrainParams):
    if cfg.action == "train":
        train_(cfg)
    elif cfg.action == "infer":
        infer_(cfg)


def train_(cfg: TrainParams):
    train.train(cfg.epoch_count, cfg.lr, cfg.batch_size)


def infer_(cfg: TrainParams):
    infer.infer(cfg.batch_size)


if __name__ == "__main__":
    # fire.Fire()
    main()

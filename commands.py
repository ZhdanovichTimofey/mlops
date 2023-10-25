import hydra
import lightning.pytorch as pl
import torch
from config import TrainParams
from hydra.core.config_store import ConfigStore
from mlops import dataset, infer, model


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainParams)


@hydra.main(config_path="conf", config_name="base_config", version_base="1.3")
def main(cfg: TrainParams):
    if cfg.action == "train":
        pl.seed_everything(42)
        torch.set_float32_matmul_precision("medium")

        dm = dataset.MyDataModule(
            batch_size=cfg.batch_size,
        )
        mymodel = model.SimpleConvNet(cfg.lr)

        loggers = [
            pl.loggers.CSVLogger(
                "./.logs/my-csv-logs", name=cfg.artifacts.experiment_name
            ),
            pl.loggers.MLFlowLogger(
                experiment_name=cfg.artifacts.experiment_name,
                tracking_uri="file:./.logs/my-mlflow-logs",
            ),
            pl.loggers.TensorBoardLogger(
                "./.logs/my-tb-logs", name=cfg.artifacts.experiment_name
            ),
        ]

        trainer = pl.Trainer(
            max_epochs=cfg.epoch_count,
            accumulate_grad_batches=cfg.grad_accum_steps,
            log_every_n_steps=cfg.log_every_n_steps,
            logger=loggers,
        )

        trainer.fit(mymodel, datamodule=dm)
    elif cfg.action == "infer":
        infer_(cfg)


def infer_(cfg: TrainParams):
    infer.infer(cfg.batch_size, cfg.lr)


if __name__ == "__main__":
    # fire.Fire()
    main()

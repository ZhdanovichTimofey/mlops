import os

import git
import hydra
import lightning.pytorch as pl
import mlflow
import onnx
import torch
from mlflow.models import infer_signature
from mlops import dataset
from mlops.model import SimpleConvNet
from omegaconf import DictConfig


def train(cfg: DictConfig, SAVE_PATH):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    dm = dataset.MyDataModule(
        batch_size=cfg.batch_size,
    )
    mymodel = SimpleConvNet(cfg.lr)

    mlflowlogger = pl.loggers.MLFlowLogger(
        experiment_name=mlflow.get_experiment(
            mlflow.active_run().info.experiment_id
        ).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epoch_count,
        accumulate_grad_batches=cfg.grad_accum_steps,
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        logger=mlflowlogger,
        devices=-1,
        enable_checkpointing=True,
    )

    trainer.fit(mymodel, datamodule=dm)

    imgs = [torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)]
    print(mymodel(imgs[0]))
    print(mymodel(imgs[1]))

    trainer.save_checkpoint(SAVE_PATH)

    X = torch.zeros((2, 3, 32, 32), dtype=torch.float32)

    torch.onnx.export(
        mymodel,
        X,
        "model.onnx",
        input_names=["INPUT"],
        output_names=["OUTPUT"],
        dynamic_axes={
            "INPUT": {0: "BATCH_SIZE"},
            "OUTPUT": {0: "BATCH_SIZE"},
        },
    )
    onnx_model = onnx.load_model("model.onnx")
    signature = infer_signature(X.numpy(), mymodel(X).detach().numpy())
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path="onnx-model",
        signature=signature,
        registered_model_name="simple-onnx-model",
    )


def infer(cfg: DictConfig, SAVE_PATH):
    model = SimpleConvNet.load_from_checkpoint(SAVE_PATH)

    mlflowlogger = pl.loggers.MLFlowLogger(
        experiment_name=mlflow.get_experiment(
            mlflow.active_run().info.experiment_id
        ).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    trainer = pl.Trainer(logger=mlflowlogger)

    dm = dataset.MyDataModule(
        batch_size=cfg.batch_size,
    )

    trainer.test(model, datamodule=dm)


@hydra.main(config_path="conf", config_name="base_config", version_base="1.3")
def main(cfg: DictConfig):

    os.system("dvc pull")

    mlflow.set_tracking_uri(cfg.mlflow.server_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.pytorch.autolog()
    mlflow.start_run()
    mlflow.log_params(cfg)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    code_version = {"commit id": sha}
    mlflow.log_params(code_version)

    SAVE_PATH = os.path.dirname(os.path.abspath(__name__))
    SAVE_PATH = os.path.join(SAVE_PATH, "saved_model/checkpoint.ckpt")

    if cfg.action == "train":
        train(cfg, SAVE_PATH)
    elif cfg.action == "infer":
        infer(cfg, SAVE_PATH)


if __name__ == "__main__":
    # fire.Fire()
    main()

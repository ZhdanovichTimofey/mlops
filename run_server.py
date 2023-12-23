import os

import hydra
import lightning.pytorch as pl
import mlflow
from mlops import dataset
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="base_config", version_base="1.3")
def run_server(cfg: DictConfig):
    os.system("dvc pull")

    mlflow.set_tracking_uri(cfg.mlflow.server_uri)

    onnx_pyfunc = mlflow.onnx.load_model(model_uri="models:/simple-onnx-model/7")
    print("2")
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.pytorch.autolog()
    mlflow.start_run()
    mlflow.log_params(cfg)

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

    trainer.test(onnx_pyfunc, datamodule=dm)


if __name__ == "__main__":
    run_server()

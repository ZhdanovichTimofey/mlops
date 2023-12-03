import lightning.pytorch as pl
import torch
import torcheval.metrics as metrics
from torch import nn


class SimpleConvNet(pl.LightningModule):
    def __init__(self, lr: float):
        super(SimpleConvNet, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.mp1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.droupout1 = nn.Dropout(0.3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.mp2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        self.droupout2 = nn.Dropout(0.3)
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(2304, 512)
        self.droupout3 = nn.Dropout(0.3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        layer1 = self.mp1(self.conv1(x))
        layer1 = self.relu1(self.droupout1(self.bn1(layer1)))

        layer2 = self.mp2(self.conv2(layer1))
        layer2 = self.relu2(self.droupout2(self.bn2(layer2)))

        out = self.flatten(layer2)
        out = self.relu3(self.droupout3(self.fc3(out)))
        out = self.fc4(out)
        return out

    def training_step(self, batch: any, batch_idx: int, dataloader_idx=0):
        X_batch, y_batch = batch
        y_preds = self(X_batch)
        loss = self.loss_fn(y_preds, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: any, batch_idx: int):
        X_batch, y_batch = batch
        y_preds = self(X_batch)
        loss = self.loss_fn(y_preds, y_batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        f1_metric = metrics.MulticlassF1Score(num_classes=10, average="macro")
        f1_metric.update(y_preds, y_batch)
        self.log(
            "val_f1", f1_metric.compute(), on_step=True, on_epoch=True, prog_bar=False
        )

        accuracy = metrics.MulticlassAccuracy(num_classes=10, average="macro")
        accuracy.update(y_preds, y_batch)
        self.log(
            "val_accuracy",
            accuracy.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        precision = metrics.MulticlassPrecision(num_classes=10, average="macro")
        precision.update(y_preds, y_batch)
        self.log(
            "val_precision",
            precision.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return {
            "val_loss": loss,
            "val_f1": f1_metric.compute(),
            "val_accuracy": accuracy.compute(),
            "val_precision": precision.compute(),
        }

    def test_step(self, batch: any, batch_idx: int, dataloader_idx: int = 0):
        X_batch, y_batch = batch
        y_preds = self(X_batch)
        loss = self.loss_fn(y_preds, y_batch)
        f1_metric = metrics.MulticlassF1Score(num_classes=10, average="macro")
        f1_metric.update(y_preds, y_batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "test_f1", f1_metric.compute(), on_step=True, on_epoch=True, prog_bar=False
        )

        accuracy = metrics.MulticlassAccuracy(num_classes=10, average="macro")
        accuracy.update(y_preds, y_batch)
        self.log(
            "test_accuracy",
            accuracy.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        precision = metrics.MulticlassPrecision(num_classes=10, average="macro")
        precision.update(y_preds, y_batch)
        self.log(
            "test_precision",
            precision.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return {
            "test_loss": loss,
            "test_f1": f1_metric.compute(),
            "test_accuracy": accuracy.compute(),
            "test_precision": precision.compute(),
        }

    def predict_step(self, batch: any, batch_idx: int) -> any:
        pass

    def configure_optimizers(self) -> any:
        param_optimizer = list(self.parameters())
        optimizer = torch.optim.Adam(param_optimizer, lr=self.lr)
        return [optimizer]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)

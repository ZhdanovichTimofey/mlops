import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .dataset import DataLoader
from .model import SimpleConvNet


def infer(batch_size):
    device = "cpu"

    data = DataLoader(batch_size)

    val_f1 = 0
    val_for_f1_b = []
    val_for_f1_p = []
    y_pred = np.array([])

    model = SimpleConvNet()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    model.train(False)

    for X_batch, y_batch in data.test_batch_gen:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)

        y_pred_ = logits.max(1)[1].detach().cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        y_pred = np.concatenate((y_pred, y_pred_))
        val_for_f1_b = np.append(val_for_f1_b, y_batch)
        val_for_f1_p = np.append(val_for_f1_p, y_pred_)

    val_f1 = f1_score(val_for_f1_b, val_for_f1_p, average="macro")
    print("val_f1", val_f1)

    pd.DataFrame(y_pred).to_csv("val_results.csv")

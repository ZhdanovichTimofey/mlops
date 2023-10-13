from dataclasses import dataclass


@dataclass
class TrainParams:
    epoch_count: int
    lr: float
    batch_size: int

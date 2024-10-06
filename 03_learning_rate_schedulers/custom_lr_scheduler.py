from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from typing import List, Dict
import math

class CustomizedLRScheduler(LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: int = 1,
                 epochs: int = 10,
                 min_lr: float = 1e-6,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)


    def get_lr(self) -> List[float]:
        epoch = self.last_epoch
        lrs = []
        for lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            if epoch < self.warmup_epochs:
                lr = lr * epoch / self.warmup_epochs
            else:
                lr = self.min_lr + (lr - self.min_lr) * 0.5 * \
                     (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
            if "lr_scale" in param_group:
                lr = lr * param_group["lr_scale"]
            lrs.append(lr)
        return lrs

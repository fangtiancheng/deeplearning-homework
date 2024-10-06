import torch
from torch.utils.data import Dataset

class SinxDataset(Dataset):
    def __init__(self,
                 x_left: float,
                 x_right: float,
                 step: int = 1000):
        self.x = torch.linspace(x_left, x_right, steps=step).unsqueeze(1)
        self.y = torch.sin(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    dataset = SinxDataset(-2, 2)

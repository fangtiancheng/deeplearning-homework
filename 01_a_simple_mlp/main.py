import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mlp import SimpleMLP
from sinx_dataset import SinxDataset
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # problem settings
    x_left = -5
    x_right = 5

    # training params
    epoches = 100
    batch_size = 64
    lr = 1e-2

    # setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mlp = SimpleMLP(1, 1, 256, 4).to(device)
    dataset = SinxDataset(x_left, x_right)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, sampler=None)
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=lr)

    # train
    mlp.train()
    for epoch in range(epoches):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = mlp(x)
            loss = loss_fn(y, y_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * batch_size + len(x)
                print(f"loss: {loss:>7f}  [{epoch:>2d}: {current:>4d}/{len(dataset):>4d}]")

    # eval
    mlp = mlp.to('cpu')
    mlp.eval()
    with torch.no_grad():
        x = torch.linspace(x_left, x_right, 1000).unsqueeze(1)
        y_pred = mlp(x)
        y = torch.sin(x)
        x = x.numpy()[:, 0]
        y_pred = y_pred.numpy()[:, 0]
        y = y.numpy()[:, 0]
        plt.plot(x, y, label='y = sin(x)')
        plt.plot(x, y_pred, label='y = mlp(x)')
        plt.legend()
        plt.show()

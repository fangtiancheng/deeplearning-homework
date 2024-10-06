import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from custom_lr_scheduler import CustomizedLRScheduler
from mlp import SimpleMLP
from sinx_dataset import SinxDataset
from matplotlib import pyplot as plt
import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training parameters for MLP.")
    parser.add_argument('--epoches', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Logging directory.')

    return parser.parse_args()

if __name__ == '__main__':
    # problem settings
    x_left = -5
    x_right = 5

    # training params
    args = parse_args()

    # setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mlp = SimpleMLP(1, 1, 256, 4).to(device)
    dataset = SinxDataset(x_left, x_right)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, sampler=None)
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=args.lr)

    # lr_scheduler init
    lr_scheduler = CustomizedLRScheduler(optimizer)
    print('lr_scheduler:', lr_scheduler.state_dict())

    # tensorboard
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # train
    mlp.train()
    for epoch in range(args.epoches):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = mlp(x)
            loss = loss_fn(y, y_pred)
            loss.backward()
            optimizer.step()

            # update learning rate
            lr_scheduler.step(epoch)

            optimizer.zero_grad()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + batch)

            # visualize learning rate
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            if batch % 10 == 0:
                loss, current = loss.item(), batch * args.batch_size + len(x)
                print(f"loss: {loss:>7f}  [{epoch:>2d}: {current:>4d}/{len(dataset):>4d}]")
    writer.close()

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

#!/bin/bash
# train
python main.py --epoches 100 --batch_siz 8 --lr 0.001 --log_dir ./log

# watch logging
tensorboard --logdir ./log

import os
import random
import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

from datetime import datetime
from datetime import timezone
from datetime import timedelta


def get_folder_name():
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S')
    save_path = f'./results/{folder_name}'
    os.makedirs(save_path)

    return folder_name, save_path


def get_data():
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/dev.csv')
    test_df = pd.read_csv('./data/test.csv')

    return train_df, val_df, test_df
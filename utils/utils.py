import os
import random
import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

from datetime import datetime


def get_folder_name():
    now = datetime.now()
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S')
    save_path = f'./results/{folder_name}'
    os.makedirs(save_path)

    return folder_name, save_path


def get_data():
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/dev.csv')
    test_df = pd.read_csv('./data/test.csv')

    return train_df, val_df, test_df


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True
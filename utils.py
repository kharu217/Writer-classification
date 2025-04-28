import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, random_split, DataLoader
import torchinfo

import string
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


data_address = [
    "Writer_classification\data\goethe.txt",
    "Writer_classification\data\hesse.txt",
    "Writer_classification\data\kafka.txt"
]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer_list = ["goethe","hesse","kafka"]

all_letters = string.ascii_letters + '.;: "\''
n_letters = len(all_letters)


def Line2tensor(lines : str) -> torch.Tensor:
    idx_list = []
    for text in lines :
        idx_list.append(all_letters.find(text))
    return idx_list

def text2tensor(address : str) -> torch.tensor :
    with open(address, encoding='utf-8') as file :
        raw_text = Line2tensor(''.join([text for text in file.read() if text in all_letters]))
    return raw_text


embd_n = 64
hidden_size = 64
layer_n = 32

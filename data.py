import torch
import string
import torch
import os
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.functional import one_hot


device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer_list = ['kafka', 'goethe', 'hesse']

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

class text_dataset(Dataset) :
    def __init__(self, addr_l, split_n):
        super().__init__()

        self.raw_x = []

        for addr_i in addr_l :
            self.raw_x.append(text2tensor(addr_i))

        self.process_x = []
        self.process_y = []

        for i_x in range(len(self.raw_x)) :
            for ii_x in range(len(self.raw_x[i_x]) // split_n - split_n) :
                self.process_x.append(self.raw_x[i_x][ii_x*split_n:ii_x*split_n + split_n])
                self.process_y.append(i_x)
        
        self.process_x = torch.tensor(self.process_x, dtype=torch.long).to(device)
        self.process_y = torch.tensor(self.process_y).to(device)
        
    def __len__(self) :
        return(len(self.process_x))

    def __getitem__(self, index):
        return self.process_x[index], self.process_y[index]

if __name__ == "__main__" :
    test_data = text_dataset([
    "Writer_classification\data\goethe.txt",
    "Writer_classification\data\hesse.txt",
    "Writer_classification\data\kafka.txt"
    ], 100)
    print(test_data.process_y.shape)


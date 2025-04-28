from utils import *

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
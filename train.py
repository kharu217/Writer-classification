from data import *
from model import *

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


data_address = [
    "Writer_classification\data\goethe.txt",
    "Writer_classification\data\hesse.txt",
    "Writer_classification\data\kafka.txt"
]

def Train(data_split : tuple, epoch=10, lr=1e-4, batch_size=5) :

    train_model = Writer_classifi(embd_n=64, hidden_size=64, layer_n=32).to(device)
  
    all_Dataset = text_dataset(data_address, 50,)
    print(len(all_Dataset))

    train_len = int(len(all_Dataset) * data_split[0])
    valid_len = int(len(all_Dataset) * data_split[1]) + 1

    train_dataset, valid_dataset = random_split(all_Dataset, (train_len, valid_len))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True)

    optimizer = optim.AdamW(train_model.parameters(), lr)
    schedular = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch : 0.95**epoch, verbose=False)
    
    loss_fn = nn.CrossEntropyLoss()

    total_loss = []
    total_accur = []

    min_loss = float('inf')

    for epch in range(1, epoch + 1) :
        epoch_loss = 0

        train_model.train()
        
        for x, y in tqdm(train_dataloader) :
            optimizer.zero_grad()

            y = F.one_hot(y, 3).to(torch.float16)

            pred = train_model(x)
            loss = loss_fn(pred, y)

            loss.backward()

            epoch_loss += loss.item()
            optimizer.step()
        schedular.step()

        loss_mean = epoch_loss/len(train_dataloader)
        total_loss.append(loss_mean)

        train_model.eval()

        valid_accur = 0
        for valid_x, vaild_y in tqdm(valid_dataloader) :
            pred = torch.argmax(train_model(valid_x))

            if pred == vaild_y :
                valid_accur += 1
        total_accur.append(valid_accur)
        print(f"epoch {epch}, accr : {valid_accur}/{len(valid_dataloader)}, loss : {loss_mean}")
        
        if epch % 10 == 0 :
            plt.subplot(1, 2, 1)
            plt.plot(total_loss)
            plt.xlabel('total_loss')
            plt.subplot(1, 2, 2)
            plt.plot(total_accur)
            plt.xlabel('total_accr')

            plt.savefig(f"Writer_classification\\train_result\\epoch{epch}_.png")

        if loss_mean < min_loss :
            min_loss = loss_mean
            torch.save(train_model.state_dict(), "Writer_classification\models\writer_model.h5")
            print(f"loss refreshed : {min_loss}")
    
        

if __name__ == "__main__" :
    Train(data_split=(0.95, 0.05), batch_size=32, epoch=150, lr=2e-3)


#히스플랜
#마커스
#제이어스
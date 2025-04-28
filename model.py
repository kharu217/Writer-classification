from utils import *

class Writer_classifi(nn.Module) :
    def __init__(self, *, embd_n, hidden_size=6, layer_n=1):
        super().__init__()
        self.embd_layer = nn.Embedding(len(all_letters), embd_n)

        self.lstm_layer = nn.LSTM(embd_n, hidden_size, num_layers=layer_n, batch_first=True, dropout=0.2)
        self.flatten = nn.Flatten(1)

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_size*layer_n, hidden_size*layer_n//2),
            nn.BatchNorm1d(num_features=(hidden_size*layer_n)//2),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear((hidden_size*layer_n)//2, 64),
            nn.BatchNorm1d(num_features=64),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(num_features=16),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(16, 16),
            nn.BatchNorm1d(num_features=16),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x:torch.Tensor) -> torch.tensor :
        embd_x = self.embd_layer(x)

        out, h_c = self.lstm_layer(embd_x)
        hidden, cell_state = h_c

        if hidden.ndim != 3 :
            hidden = hidden.unsqueeze(0)
        else :
            hidden = hidden.permute((1, 0, 2))
        
        flatten_x = self.flatten(hidden)
        linear_x = self.linear_layer(flatten_x)
        return linear_x
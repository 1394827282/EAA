import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.jit as jit
class LSTMNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.name = 'LSTM'
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_list = []
        h = x
        h_list.append(torch.mean(h, 0, True).detach())
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)

        activations = {}
        activations['lstm'] = lstm_out

        lstm_out = lstm_out[:, -1, :]

        h = lstm_out
        h_list.append(torch.mean(h, 0, True).detach())

        priority = self.fc(lstm_out)

        activations['fc'] = priority

        return priority, activations, h_list
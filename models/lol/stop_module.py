import torch
from torch import nn

from models.lol.lstm_1d import LSTM1D


class StopModule(nn.Module):

    def __init__(self, lstm_output_size=8):
        super(StopModule, self).__init__()
        # Linear layer at the end to get angle and sizing
        self.fully_connected = nn.Linear(512 + lstm_output_size, 1)
        self.fully_connected.bias.data[0] = -5  # base x
        self.lstm = LSTM1D(hidden_layer_size=64, output_size=lstm_output_size)

    def forward(self, input, angles):
        y = self.lstm(angles)
        y = self.fully_connected(torch.cat([input, y]))
        y = torch.sigmoid(y)
        return y

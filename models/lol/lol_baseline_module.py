import torch
from torch import nn

from models.lol.lstm_1d import LSTM1D

"""
Outputs next baseline point and next angle for patching
"""


class BaselineModule(nn.Module):

    def __init__(self, lstm_output_size=8):
        super(BaselineModule, self).__init__()
        # Linear layer at the end to get angle and sizing
        self.fully_connected = nn.Linear(512 + lstm_output_size, 3)
        self.fully_connected.bias.data[0] = 0  # base x
        self.fully_connected.bias.data[1] = 0  # base y
        self.fully_connected.bias.data[2] = 0  # angle
        self.lstm = LSTM1D(hidden_layer_size=64, output_size=8).cuda()

    def forward(self, input, angles):
        # y = self.l2(y)
        y = self.lstm(angles)
        y = self.fully_connected(torch.cat([input, y]))
        return y

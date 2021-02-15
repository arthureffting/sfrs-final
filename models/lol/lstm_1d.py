import torch
from torch import nn


class LSTM1D(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        torch.autograd.set_detect_anomaly(True)
        input_view = input_seq.view(len(input_seq), 1, -1)
        lstm_out, _ = self.lstm(input_view)
        formatted = lstm_out.view(len(input_seq), -1)
        predictions = self.linear(formatted)
        return predictions[-1]

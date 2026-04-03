import torch
from torch import nn

class NeuralDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = ConvLSTM(
            img_size=(70, 70),
            input_dim=1,
            hidden_dim=3,
            kernel_size=(3, 3),
            batch_first=True,
            bidirectional=False,
            return_sequence=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=31, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=13, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, last_state, _ = self.convlstm(x)
        x = self.conv1(last_state[0])
        x = self.conv2(x)
        return self.sigmoid(x)

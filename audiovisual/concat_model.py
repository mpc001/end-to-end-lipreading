# coding: utf-8
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        if self.every_frame:
            out = self.fc(out)  # predictions based on every time step
        else:
            out = self.fc(out[:, -1, :])  # predictions based on last time-step
        return out


def lipreading(mode, inputDim=2048, hiddenDim=2048, nLayers=2, nClasses=500, frameLen=29, every_frame=True):
    if mode == 'backendGRU' or mode == 'finetuneGRU':
        model = GRU(inputDim, hiddenDim, nLayers, nClasses, every_frame)
    print('\n'+mode+' model has been built')
    return model

from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, init


class TemporalAttention(nn.Module):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_channel, num_of_features)
    Output: (batch_size, num_of_timesteps)
    '''

    def __init__(self, input_shape):
        super(TemporalAttention, self).__init__()
        batch, num_of_channel, num_of_features = input_shape
        self.U_1 = nn.Parameter(torch.FloatTensor(num_of_channel, num_of_channel))
        self.b_e = nn.Parameter(torch.FloatTensor(num_of_features, num_of_features))
        self.V_e = nn.Parameter(torch.FloatTensor(num_of_features, num_of_features))
        self.reset_weigths()

    def forward(self, x):
        batch, num_of_channel, num_of_features = x.shape  # (B, C, F)

        temp = x.permute(0, 2, 1)  # (B, F, C) 64*124*64
        lhs = torch.matmul(temp, self.U_1)  # (B, F, 1) 64*124

        # shape of product is (batch_size, F, F)
        product = torch.matmul(lhs, x)

        product = torch.sigmoid(product + self.b_e)

        S = torch.matmul(self.V_e, product)

        # normalization
        S_normalized = F.softmax(S, dim=1)

        return S_normalized

    def reset_weigths(self):
        """reset weights
            """
        for weight in self.parameters():
            init.xavier_normal_(weight)

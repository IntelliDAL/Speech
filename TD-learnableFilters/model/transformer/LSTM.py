# -*- coding: utf-8 -*-
# file: dynamic_rnn.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.autograd.grad_mode import F


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM', input_resolution=(0, 0)):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.input_resolution = input_resolution
        self.embedding_dim = input_size
        self.reduce_dim = nn.Linear(input_size * 4, input_size)
        if self.rnn_type == 'LSTM':
            self.row_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
            self.col_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
            # 有问题
        # self.attention_size_row = Attention(self.input_size * 2, batch_first=True)  # 2 is bidrectional
        # self.attention_size_col = Attention(self.input_size * 2, batch_first=True)  # 2 is bidrectional

    def init_hidden(self, k):
        device = next(self.row_lstm.parameters()).device
        return torch.zeros(2, k, self.embedding_dim).to(device), torch.zeros(2, k, self.embedding_dim).to(device)

    def forward(self, features):
        # LSTM的输入特征格式 batch_size sequence_length input_size (特征维度)
        # 特征输入
        H, W = self.input_resolution
        B, L, C = features.shape
        assert L == H * W, "input feature has wrong size"

        # 准备行的数据
        features_row = features.reshape(-1, H, C)
        self.row_hidden = self.init_hidden(features_row.shape[1])
        lstm_out, self.row_hidden = self.row_lstm(features_row)
        # attn_output_row, _ = self.attention_size_row(lstm_out)
        # 行特征
        row_tensor = lstm_out.reshape(B, -1, 2 * C)

        # 准备列的数据
        features_col = features.reshape(-1, W, C)
        self.col_hidden = self.init_hidden(features_col.shape[1])
        lstm_out, self.col_hidden = self.col_lstm(features_col)
        # attn_output_col, _ = self.attention_size_col(lstm_out)
        # 列特征
        col_tensor = lstm_out.reshape(B, -1, 2 * C)

        # 特征拼接，
        table_tensor = torch.cat([row_tensor, col_tensor], dim=2)
        # reduce
        table_tensor = self.reduce_dim(table_tensor.view(-1, 4 * C)).view(B, L, self.embedding_dim)

        return table_tensor


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        stdv = 1.0 / np.sqrt(self.hidden_size)

        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )
        attentions = torch.softmax(torch.relu(weights.squeeze()), dim=-1)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        # representations = weighted.sum(1).squeeze()
        representations = weighted.squeeze()

        return representations, attentions

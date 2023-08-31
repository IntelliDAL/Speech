import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import model.Filter_files.impulse_responses as impulse_responses
from model.Filter_files import convolution
from model.Filter_files.utils import get_padding_value


class Adaptpooling(nn.Module):
    def __init__(self, in_channels, kernel_size, strides=1, padding="same", use_bias=False):
        super(Adaptpooling, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.in_channels = in_channels

        w = torch.ones((1, 1, in_channels, 1)) / (kernel_size*2)
        self.weights = nn.Parameter(w)
        # const init of 0.4 makes it approximate a Hanning window

        # w = torch.ones((1, 1, in_channels, 1))
        # self.weights = nn.Parameter(w)
        # nn.init.constant_(self.weights, 14.0)

        # --------------------------------------------
        # low_freq_mel = 40
        # fs = 16000
        # high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))
        # mel_points = np.linspace(low_freq_mel, high_freq_mel, in_channels)
        # f_cos = (700 * (10 ** (mel_points / 2595) - 1))
        # freq_scale = fs * 1.0
        # self.filt_b1 = nn.Parameter(torch.from_numpy(f_cos / freq_scale)).cuda()
        # --------------------------------------------
        # fscale = 400
        # points = torch.linspace(1 / fscale, 400 / fscale, 64)
        # self.rectangle = nn.Parameter(points, requires_grad=True).cuda()
        # -------------------------------------------
        if self.use_bias:
            self._bias = torch.nn.Parameter(torch.ones(in_channels, ))
        else:
            self._bias = None

        if self.padding.lower() == "same":
            self.pad_value = get_padding_value(kernel_size)
        else:
            self.pad_value = self.padding

        # self.Conv1d = nn.Conv1d(self.in_channels, self.in_channels, self.kernel_size, bias=False, stride=self.strides, padding=0, groups=self.in_channels)
        # self.Pooling = nn.MaxPool1d(self.kernel_size, stride=self.strides)
        # self.Pooling = nn.AvgPool1d(self.kernel_size, stride=self.strides)

    def forward(self, x):
        """fold and unfold"""
        # window = impulse_responses.Kaiser_windows(self.kernel_size, beta=self.weights).cuda()
        # b, c, l = x.shape
        # x = x.view(b * c, 1, 1, l)
        # unfold = torch.nn.functional.unfold(x, kernel_size=(1, self.kernel_size), stride=(1, self.strides))
        # x = unfold * window.view(self.kernel_size, 1)
        # unfold = torch.nn.functional.fold(x, output_size=l, kernel_size=(1, self.kernel_size), stride=(1, self.strides))
        # x = unfold.view(b, c, l)
        # ==========================================

        """standard"""
        # window = window.reshape(-1, self.kernel_size, self.in_channels)
        # window = window.permute(2, 0, 1)
        # window = torch.kaiser_window(self.kernel_size).cuda()
        # window = torch.hann_window(self.kernel_size).cuda()
        # window = torch.hann_window(self.kernel_size).cuda()
        # window = torch.hamming_window(self.kernel_size).cuda()
        # window = torch.bartlett_window(self.kernel_size).cuda(1)
        # window = impulse_responses.Sincwindows(N_filt=self.in_channels, Filt_dim=self.kernel_size, fs=16000, filt_b1=self.filt_b1)
        # --------------------------------------------------
        # outputs = F.conv1d(x, kernel, bias=self._bias, stride=self.strides, padding=pad_val, groups=self.in_channels)
        # window = impulse_responses.gaussian_lowpass(self.weights, self.kernel_size)
        # window = window.reshape(-1, self.kernel_size, self.in_channels)
        # window = window.permute(2, 0, 1)
        # kernel = self.Conv1d.weight.data
        # self.Conv1d.weight.data = window * kernel
        # outputs = self.Conv1d(x)
        # outputs = self.Pooling(x)
        # ------------------------------------------------------
        # kernel = impulse_responses.boxcarWindow_(N_filt=self.in_channels, Filt_dim=self.kernel_size, fs=16000, filt_b1=self.rectangle).cuda()
        # kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        # kernel = kernel.permute(2, 0, 1)
        # ------------------------------------------------------
        """gaussian"""
        kernel = impulse_responses.Kaiser_windows(self.kernel_size, self.weights)
        # print(self.weights)
        # kernel = impulse_responses.gaussian_lowpass(self.weights, self.kernel_size)
        kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        kernel = kernel.permute(2, 0, 1)
        outputs = F.conv1d(x, kernel, bias=self._bias, stride=self.strides, groups=self.in_channels)
        # ------------------------------------------------------
        """sinc"""
        # kernel = impulse_responses.Sincwindows(N_filt=self.in_channels, Filt_dim=self.kernel_size, fs=16000, filt_b1=self.filt_b1)
        # outputs = F.conv1d(x, kernel, bias=self._bias, stride=self.strides, padding=pad_val, groups=self.in_channels)
        return outputs

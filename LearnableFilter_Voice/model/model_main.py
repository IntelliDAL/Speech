import torch.nn
from torch import nn

from model.Classifler.models.RNN_FCN import LSTM_FCN
from model.Classifler.models.ResNetPlus import ResNetPlus
from model.Classifler.models.ResNet18 import resnet18
from model.Classifler.models.TST import TST
from model.Filter_files.LearnableFilters import LearnableFilters

from LearnableFilter_Voice.model.Classifler.models.ResNet import ResNet
from LearnableFilter_Voice.model.attention.attention_model import ChannelTimeSenseAttentionSELayer


class AudioSpectral(nn.Module):
    def __init__(self, args):
        super(AudioSpectral, self).__init__()
        self.kernel_size = args.kernel_size
        self.output_channels = args.output_channels
        self.fs = args.fs
        self.initializer = args.initializer
        self.LearnableFilters = LearnableFilters(n_filters=self.output_channels, sample_rate=self.fs, window_len=25., window_stride=10., init_min_freq=60.0, init_max_freq=7800.0,  initializer=self.initializer)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        output = self.LearnableFilters(inputs)
        return output


class InfoAttentionClassifier(nn.Module):
    def __init__(self, args):
        super(InfoAttentionClassifier, self).__init__()
        self.channels = args.output_channels
        self.AudioSpectral = AudioSpectral(args)
        self.AudioInfoCollect = AudioInfoCollect(args)
        # self.TE = TST(self.channels, 2, 596)
        # self.LSTM = LSTM_FCN(self.channels, 2, seq_len=595, rnn_layers=2, hidden_size=256, bidirectional=False)
        self.ResNet = ResNet(self.channels, 2)

    def forward(self, x):
        temporalFeatures = self.AudioSpectral(x)
        output_ = self.AudioInfoCollect(temporalFeatures)
        output_ = self.ResNet(output_)
        return output_


class AudioInfoCollect(torch.nn.Module):
    def __init__(self, args):
        """Inititalize variables."""
        super(AudioInfoCollect, self).__init__()
        self.output_channels = args.output_channels
        self.hidden_channels = args.hidden_channels
        self.skip_channels = args.skip_channels
        self.n_layers = args.n_layers
        self.n_blocks = args.n_blocks
        self.dilation = args.dilation
        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.relu = nn.ReLU()
        self.kernel_size = args.kernel_size
        self.conv, self.skip, self.resi = [], [], []
        for idx, d in enumerate(self.dilations):
            self.skip.append(torch.nn.Conv1d(in_channels=args.hidden_channels, out_channels=self.skip_channels, kernel_size=1, bias=False))
            self.resi.append(torch.nn.Conv1d(in_channels=args.hidden_channels, out_channels=args.hidden_channels, kernel_size=1, bias=False))
            self.conv.append(torch.nn.Conv1d(in_channels=args.hidden_channels, out_channels=args.hidden_channels, kernel_size=self.kernel_size, bias=False, dilation=d, padding=d * (2 - 1) // 2, groups=args.hidden_channels))
        self.conv = torch.nn.ModuleList(self.conv)
        self.skip = torch.nn.ModuleList(self.skip)
        self.resi = torch.nn.ModuleList(self.resi)
        self.CH_downsample = torch.nn.Conv1d(in_channels=args.hidden_channels * self.n_layers, out_channels=args.hidden_channels, kernel_size=1, bias=False)
        self.channel_att = ChannelTimeSenseAttentionSELayer(self.hidden_channels)

    def forward(self, inputs: torch.Tensor):
        """Returns embedding vectors. """
        output = inputs
        skip_connections = []
        for layer, conv, skip, resi in zip(self.dilations, self.conv, self.skip, self.resi):
            layer_in = output
            "dilated"
            output = conv(output)
            "skip-connectiuon"
            skip = skip(output)
            skip_connections.append(skip)
            "backbone"
            output = self.relu(output)
            output = resi(output)
            "resNet"
            output = output + layer_in[:, :, -output.size(2):]
            if layer == 2 ** (self.n_layers - 1):
                skip_connections = sum([s[:, :, -output.size(2):] for s in skip_connections])
                output, att_weight = self.channel_att(skip_connections)
                skip_connections = []
        return output


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.channels = args.output_channels
        self.resnet18 = resnet18()

    def forward(self, x):
        output_ = self.resnet18(x)
        return output_


class Regression(nn.Module):
    def __init__(self, args):
        super(Regression, self).__init__()
        self.channels = args.output_channels
        self.regression = ResNetPlus(self.channels, 1)

    def forward(self, x):
        scores_ = self.regression(x)
        return scores_.squeeze()

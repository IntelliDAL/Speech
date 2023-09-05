import torch.nn
from torch import nn
from torch.nn import init
from torch.nn.init import trunc_normal_

from model.Classifler.models.ConvNet import ConvNeXt
from model.Filter_files.FD_filters import GaussianShape
from model.Filter_files.postprocessing import TemporalBatchNorm
from model.PatchMerging import PatchMerging
from model.attention.attention_model import ChannelSELayer
from model.attention.self_attention import SelfAttention
from model.transformer.htsat import backbone
from model.Classifler.models.ResNet import ResNet, ResNet_sup
from model.Filter_files import postprocessing
import config_transformer as config


class AudioSpectral(nn.Module):
    def __init__(self, args):
        super(AudioSpectral, self).__init__()
        self.kernel_size = args.kernel_size
        self.output_channels = args.output_channels
        self.fs = args.fs
        # self.initializer = args.initializer
        # self.LearnableFilters = LearnableFilters(n_filters=self.output_channels, sample_rate=self.fs, window_len=25., window_stride=10., init_min_freq=60.0, init_max_freq=7800.0,  initializer=self.initializer)
        self.filter_size = args.filter_size
        self.filter_num = args.filter_num
        self.NFFT = args.NFFT
        self.sigma_coff = args.sigma_coff
        # 归一化方法
        # ---------------------------
        # 1. MCN
        # 2. Layernorm Batchnorm
        # self.norm = TemporalBatchNorm(args.filter_num, num_channels=1)
        # self.norm = torch.nn.LayerNorm(args.frame_num)
        # ---------------------------
        # 结果非线性变换
        self.compression_PCEN = postprocessing.PCENLayer(self.filter_num, alpha=0.98, smooth_coef=0.04, delta=2.0,
                                                         floor=1e-12, trainable=True, learn_smooth_coef=True,
                                                         per_channel_smooth_coef=True)
        self.FD_Learned_Filters = GaussianShape(args)

    def forward(self, inputs):
        """时域滤波器"""
        # inputs = inputs.unsqueeze(1)
        # output = self.LearnableFilters(inputs)
        """频域滤波器"""
        gauss, sort_Loss = self.FD_Learned_Filters(inputs)
        "Normalization+relu, 主要防止数值非负数"
        # gauss = self.norm(gauss)
        # gauss = torch.relu(gauss)
        # gauss = gauss.transpose(1, 2)
        "非线性变换"
        NL_gauss, loss_delta = self.compression_PCEN(gauss)
        # ---------------------
        sort_Loss = loss_delta + sort_Loss
        # ---------------------
        return NL_gauss.squeeze(), sort_Loss


class InfoAttentionClassifier(nn.Module):
    def __init__(self, args):
        super(InfoAttentionClassifier, self).__init__()
        self.channels = args.output_channels
        self.AudioSpectral = AudioSpectral(args)
        self.spectralFusion = AudioInfoCollect(args)
        self.backbone = ResNet(self.channels // 2 ** (args.n_blocks - 1), 2)
        """下面模型的主要研究目的就是为了能够从文字和频率的两个角度对构建好的音频特征进行两个维度的特征提取"""

    def forward(self, x):
        """特征提取"""
        inputs, sort_Loss = self.AudioSpectral(x)  # B Freq Frame
        # inputs = inputs.unsqueeze(2)  # B Freq C Frame  where C=1
        """分类模型"""
        output_ = self.spectralFusion(inputs)
        output_ = self.backbone(output_)
        return output_, sort_Loss


class supervisedBackbone(nn.Module):
    def __init__(self, args):
        super(supervisedBackbone, self).__init__()
        self.channels = args.output_channels
        self.AudioSpectral = AudioSpectral(args)

    def forward(self, x):
        """特征提取"""
        output, sort_Loss = self.AudioSpectral(x)  # B Freq Frame
        "这个地方需要注意特征构造后的大小.随着滤波器中心频率参数的移动,特征的数据会逐渐变小.-->"
        return output, sort_Loss


class supervisedBackbTwo(nn.Module):
    def __init__(self, args):
        super(supervisedBackbTwo, self).__init__()
        self.channels = args.output_channels
        self.AudioSpectral = AudioSpectral(args)

    def forward(self, x):
        """特征提取"""
        output, sort_Loss = self.AudioSpectral(x)  # B Freq Frame
        "这个地方需要注意特征构造后的大小.随着滤波器中心频率参数的移动,特征的数据会逐渐变小.-->"
        return output, sort_Loss


class RegionalExtraction1(nn.Module):
    def __init__(self, args):
        super(RegionalExtraction1, self).__init__()
        self.channels = args.output_channels
        self.spectralFusion = AudioInfoCollect(args)
        # 添加projector
        self.backbone = ResNet_sup(self.channels // 2 ** (args.n_blocks - 1), 2)

    def forward(self, inputs):
        """分类模型"""
        output_ = self.spectralFusion(inputs)
        output_ = self.backbone(output_)
        return output_


class RegionalExtraction2(nn.Module):
    def __init__(self, args):
        super(RegionalExtraction2, self).__init__()
        self.channels = args.output_channels
        self.spectralFusion = AudioInfoCollect(args)
        # 添加projector
        self.backbone = ResNet_sup(self.channels // 2 ** (args.n_blocks - 1), 2)

    def forward(self, inputs):
        """分类模型"""
        output_ = self.spectralFusion(inputs)
        output_ = self.backbone(output_)
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
        # 像素 --------------------
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num
        # 像素 --------------------

        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.kernel_size = args.kernel_size
        self.relu = nn.LeakyReLU(0.2)
        self.conv, self.skip, self.downsample, self.norm = [], [], [], []
        hidden_channels = self.hidden_channels
        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
                # nn.LayerNorm(self.frame_num // (2 ** (idx // self.n_layers)))
                )
            self.skip.append(skip_tmp)
            # self.resi.append(nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1,
            # bias=False))
            conv_tmp = torch.nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=self.kernel_size,
                          bias=False, dilation=d, padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
                # nn.BatchNorm1d(hidden_channels)
                )
            self.conv.append(conv_tmp)
            if (idx + 1) % self.n_layers == 0:  # 如果能被整除
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))
            # ============9==========================
        for block in range(self.n_blocks):
            input_resolution = (self.filter_num // (2 ** block), self.frame_num // (2 ** block))
            self.downsample.append(PatchMerging(input_resolution, dim=16, block=block))
            # self.norm.append(nn.LayerNorm(self.frame_num // (2 ** block)))
        self.conv = nn.ModuleList(self.conv)
        self.skip = nn.ModuleList(self.skip)
        self.downsample = nn.ModuleList(self.downsample)
        # self.norm = nn.ModuleList(self.norm)
        # self.resi = nn.ModuleList(self.resi)

    def forward(self, inputs: torch.Tensor):
        """Returns embedding vectors. """
        output = inputs
        skip_connections = []
        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            shortcut = output
            "dilated"
            output = conv(output)
            "skip-connectiuon"
            output = self.relu(output)
            skip_outputs = skip(output)
            skip_connections.append(skip_outputs)
            # output = resi(output) # 去掉，降低网络层数
            "resNet"
            output = output + shortcut[:, :, -output.size(2):]
            if dilation == 2 ** (self.n_layers - 1):  # 每个block单独进行特征汇聚
                # 定义一个权重,来动态融合每个block中的多尺度特征, 每个block都提取了不同尺度的信息. 但是融合的时候要有策略.
                sum_output = sum([s[:, :, -output.size(2):] for s in skip_connections])
                # output = self.norm[((idx + 1) // self.n_layers)-1](output)
                """
                要充分挖掘时间T 和 频率 F 上的信息  
                1. 多尺度汇聚
                2. 降低维度， 在每个block中， TCN中时间序列的维度并未改变。B F T
                # PatchMerging的优势：可以完整保留相应区域中的信息。
                """
                if ((idx + 1) // self.n_layers) <= self.n_blocks - 1: output = self.downsample[((idx + 1) // self.n_layers) - 1](sum_output)
                # 清空每个block的多尺度信息, 每次清空
                skip_connections = []

        return output

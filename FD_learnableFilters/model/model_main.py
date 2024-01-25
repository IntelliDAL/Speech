import torch.nn
from torch import nn

from model.Filter_files import postprocessing
from model.Filter_files.LFB import GaussianShape, GaussianShape_2
from model.MCCA import PatchMerging as MCCA
from model.Classifler.models.ResNet import ResNet, ResNet_sup


class Learnable_FD_filters(nn.Module):
    def __init__(self, args):
        super(Learnable_FD_filters, self).__init__()
        self.kernel_size = args.kernel_size
        self.output_channels = args.output_channels
        self.fs = args.fs
        self.filter_size = args.filter_size
        self.filter_num = args.filter_num
        self.NFFT = args.NFFT
        self.sigma_coff = args.sigma_coff
        # ---------------------------
        self.compression_PCEN = postprocessing.PCENLayer(self.filter_num, alpha=0.98, smooth_coef=0.04, delta=2.0, floor=1e-6, trainable=True, learn_smooth_coef=True, per_channel_smooth_coef=True)
        self.FD_Filters = GaussianShape(args)

    def forward(self, inputs):
        """频域滤波器"""
        gauss, sort_Loss = self.FD_Filters(inputs)
        "非线性变换"
        NL_gauss, loss_delta = self.compression_PCEN(gauss)
        # ---------------------
        sort_Loss = loss_delta + sort_Loss
        # ---------------------
        return NL_gauss.squeeze(), sort_Loss


class AudioSpectral_2(nn.Module):
    def __init__(self, args):
        super(AudioSpectral_2, self).__init__()
        self.kernel_size = args.kernel_size
        self.output_channels = args.output_channels
        self.fs = args.fs
        self.filter_size = args.filter_size
        self.filter_num = args.filter_num
        self.NFFT = args.NFFT
        self.sigma_coff = args.sigma_coff
        # ---------------------------
        self.FD_Learned_Filters = GaussianShape_2(args)

    def forward(self, inputs):
        """频域滤波器"""
        gauss, sort_Loss = self.FD_Learned_Filters(inputs)
        return gauss.squeeze(), sort_Loss


class InfoAttentionClassifier(nn.Module):
    def __init__(self, args):
        super(InfoAttentionClassifier, self).__init__()
        self.channels = args.output_channels
        self.LFB = Learnable_FD_filters(args)
        self.MCCAs = MCCAs(args)
        num_classes = 2
        self.out_dim = (args.filter_num // (2 ** (args.n_blocks-1))) * (args.frame_num // (2 ** (args.n_blocks-1)))
        # self.FC = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.Conv = ResNet(self.channels // 2 ** (args.n_blocks - 1),   c_out=num_classes)

    def forward(self, x):
        """特征提取"""
        inputs, sort_Loss = self.LFB(x)  # B Freq Frame
        """分类模型"""
        output_ = self.MCCAs(inputs)
        # output_ = output_.reshape(output_.size(0), -1)
        # output_ = self.FC(output_)
        output_ = self.Conv(output_)
        return output_, sort_Loss


class supervisedBackbone(nn.Module):
    def __init__(self, args):
        super(supervisedBackbone, self).__init__()
        self.channels = args.output_channels
        self.LFB = Learnable_FD_filters(args)

    def forward(self, x):
        """特征提取"""
        output, sort_Loss = self.LFB(x)  # B Freq Frame
        return output, sort_Loss


class supervisedBackbTwo(nn.Module):
    def __init__(self, args):
        super(supervisedBackbTwo, self).__init__()
        self.channels = args.output_channels
        self.AudioSpectral = AudioSpectral_2(args)

    def forward(self, x):
        """特征提取"""
        output, sort_Loss = self.AudioSpectral(x)  # B Freq Frame
        return output, sort_Loss


class RegionalExtraction1(nn.Module):
    def __init__(self, args):
        super(RegionalExtraction1, self).__init__()
        self.channels = args.output_channels
        self.spectralFusion = MCCAs(args)
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
        self.spectralFusion = MCCAs(args)
        # 添加projector
        self.backbone = ResNet_sup(self.channels // 2 ** (args.n_blocks - 1), 2)

    def forward(self, inputs):
        """分类模型"""
        output_ = self.spectralFusion(inputs)
        output_ = self.backbone(output_)
        return output_


class MCCAs(torch.nn.Module):
    def __init__(self, args):
        """Inititalize variables."""
        super(MCCAs, self).__init__()
        self.output_channels = args.output_channels
        self.hidden_channels = args.hidden_channels
        self.skip_channels = args.skip_channels
        self.n_layers = args.n_layers
        self.n_blocks = args.n_blocks
        self.dilation = args.dilation
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num
        self.dilations = [args.dilation ** i for i in range(args.n_layers)] * args.n_blocks
        self.input_dim = int(args.input_dim)
        self.kernel_size = args.kernel_size
        self.conv, self.skip, self.RepExtractorModule, self.norm1, self.norm2 = [], [], [], [], []
        hidden_channels = self.hidden_channels
        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False, groups=hidden_channels))
            self.skip.append(skip_tmp)
            conv_tmp = torch.nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=self.kernel_size,
                          bias=False, dilation=d, padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
                nn.LeakyReLU(0.2))
            self.conv.append(conv_tmp)

            if (idx + 1) % self.n_layers == 0:
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))
        for block in range(self.n_blocks):
            input_resolution = (self.filter_num // (2 ** block), self.frame_num // (2 ** block))
            self.RepExtractorModule.append(MCCA(input_resolution, dim=192, block=block, flag=args.n_blocks))

        self.conv = nn.ModuleList(self.conv)
        self.skip = nn.ModuleList(self.skip)
        self.RepExtractorModule = nn.ModuleList(self.RepExtractorModule)

    def forward(self, inputs):
        output = inputs
        skip_connections = []
        ori_x = inputs
        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            shortcut = output
            output = conv(output)
            skip_outputs = skip(output)
            skip_connections.append(skip_outputs)
            output = output + shortcut[:, :, -output.size(2):]
            if dilation == 2 ** (self.n_layers - 1):
                sum_output = sum([s[:, :, -output.size(2):] for s in skip_connections])
                if ((idx + 1) // self.n_layers) <= self.n_blocks - 1:
                    output = self.RepExtractorModule[((idx + 1) // self.n_layers) - 1](sum_output, ori_x)
                skip_connections = []

        return output



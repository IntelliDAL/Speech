from torch import nn
from model.Filter_files.tdfilters import GaussianShape
from model.Filter_files import postprocessing


class FreqLearnableFilters(nn.Module):
    def __init__(self, args):
        super(FreqLearnableFilters, self).__init__()
        self.kernel_size = args.kernel_size
        self.output_channels = args.output_channels
        self.fs = args.fs
        self.filter_size = args.filter_size
        self.filter_num = args.filter_num
        self.NFFT = args.NFFT

        self.FD_Learned_Filters = GaussianShape(self.filter_size, self.filter_num, self.sigma_coff, self.NFFT)
        self.compression = postprocessing.PCENLayer(self.filter_num, alpha=0.96, smooth_coef=0.025, delta=2.0,
                                                    floor=1e-12, trainable=True, learn_smooth_coef=True,
                                                    per_channel_smooth_coef=True)
        # 结果非线性变换

    def forward(self, inputs):
        gauss = self.FD_Learned_Filters(inputs)
        res = self.compression(gauss)
        return res

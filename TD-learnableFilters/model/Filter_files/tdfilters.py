import torch
from torch import nn
from model.Filter_files import convolution
from model.Filter_files import initializers
from model.Filter_files import pooling
from model.Filter_files import postprocessing
import matplotlib.pyplot as plt
import numpy as np


class GaussianShape(nn.Module):
    def __init__(self, filter_size, filter_num, sigma_coff, NFFT):
        super(GaussianShape, self).__init__()
        self.sample_rate = 16000
        self.w = torch.nn.Parameter(self.guass_paramset(filter_num, sigma_coff))
        self.filter_size = filter_size
        self.filter_number = filter_num
        self.NFFT = NFFT

    def forward(self, inputs):
        cn = self.w[:, 0]
        sigma = self.w[:, 1]
        filters = gaussian_lowpass(cn, sigma, self.filter_size, self.filter_number, self.NFFT)
        filters = filters.transpose(1, 0)  # [Length,filters_numb]
        res = torch.matmul(inputs.permute(0, 2, 1), filters).permute(0, 2, 1)
        return res

    def guass_paramset(self, filter_num, sigma_coff):
        """参数初始化阶段"""

        """1. 梅尔初始化"""
        low_freq_mel = 30
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        nfilt = 64
        mel_points = torch.linspace(low_freq_mel, high_freq_mel, nfilt)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        cn = (hz_points / (self.sample_rate / 2)).unsqueeze(1)  # 范围 [0, 1]
        sigma = torch.ones((filter_num, 1)) * sigma_coff
        """2. 随机初始化"""
        # cn = nn.init.uniform_(torch.empty(filter_num, 1), 0, 1).cuda()
        # sigma = nn.init.uniform_(torch.empty(filter_num, 1), sigma_low, sigma_top).cuda()

        """拼接"""
        output = torch.cat([cn, sigma], dim=1)
        return output


"""若干个滤波器参数构造和搭建"""
def gaussian_lowpass(cn, sigma, filter_size, nfilt, NFFT):
    """
    cn和sigma 参数限制在一定范围内
    Role:约束滤波器的参数大小值，保持滤波器的形状处在合理范围内。
    """
    sigma = torch.clamp(sigma, min=0.00001, max=0.5)
    cn = torch.clamp(cn, min=30/80000, max=1)

    """
    cn和sigma 参数重新排序
    Role:重新排序滤波器参数,保证滤波器中心频率能够正确/合理分布。
    """
    # 计算滤波器参数的排序索引
    sorted_indices = torch.argsort(cn)
    # 重新排列滤波器参数
    cn = torch.index_select(cn, 0, sorted_indices)
    sigma = torch.index_select(sigma, 0, sorted_indices)
    # 计算滤波器参数
    fbanks = torch.zeros((nfilt, int(NFFT / 2 + 1))).cuda()
    half_filterLen = 4
    for i in range(len(cn)):
        center_Location = cn[i] * filter_size  # 中心频率位置
        band_low = int(center_Location) - half_filterLen if int(center_Location) - half_filterLen >= 0 else 0
        band_high = (int(center_Location) + half_filterLen+1) if int(center_Location) + half_filterLen < filter_size else filter_size-1
        t = torch.arange(band_low, band_high, dtype=sigma.dtype, device=sigma.device) / filter_size
        """高斯滤波器"""
        # ===================================================
        numerator = t - cn[i]
        temp = torch.exp(-0.5 * (numerator / sigma[i]) ** 2)
        # ====================================================
        """三角滤波器"""
        # =====================================================
        # numerator = torch.abs(t - cn[i])
        # temp = torch.relu(1-2 * numerator / sigma[i])
        # =====================================================
        fbanks[i][band_low:band_high] = temp

    return fbanks


def gaussian_lowpassYWJ(cn, sigma, filter_size, flag):
    nfilt = 64
    NFFT = 512
    sample_rate = 16000
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    """中心频率位置分布"""
    hz_points = np.linspace(0, 1, nfilt + 2)
    bin = (hz_points / (sample_rate / 2))

    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])

    t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device) / filter_size
    numerator = t.unsqueeze(0) - cn.unsqueeze(1)
    temp = torch.exp(-0.5 * (numerator / sigma.unsqueeze(1)) ** 2)
    '目前是全部长度，要根据中心频率的位置进行动态的调整。其中这个长度 1. 可以是定长，2. 也可以是动态调整的。'

    return temp.cuda()


def show_gaussian_function(x, y, cn, sigma):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决pythonmatplotlib绘图无法显示中文的问题
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(x, y, 'b-', linewidth=2)
    strcn = str(cn)[7:13]
    strsigma = str(sigma)[7:13]
    plt.title("cn=" + strcn + " sigma=" + strsigma)
    plt.xlim(0, 1.5)
    plt.ylim(0, 1.5)
    # plt.legend(labels=['$\mu = '+strcn+'$'+', \sigma='+strsigma+'$'])
    plt.savefig('./gaussPic/' + "cn=" + strcn + " sigma=" + strsigma + '.png')
    # plt.show()
    plt.close()

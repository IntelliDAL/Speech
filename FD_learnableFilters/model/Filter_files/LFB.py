import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


def hz_to_erb(frequency_hz):
    return 21.4 * np.log10(4.37 * frequency_hz / 1000 + 1)


def erb_to_hz(erb_value):
    return 1000 * ((10 ** (erb_value / 21.4)) - 1) / 4.37


# 定义 Hz 到 Bark 的转换函数
def hz_to_bark(hz):
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2.0)


# 定义 Bark 到 Hz 的转换函数
def bark_to_hz(bark):
    return 600.0 * np.sinh(bark / 7.0)

class GaussianShape_2(nn.Module):
    def __init__(self, args):
        super(GaussianShape_2, self).__init__()
        self.sample_rate = 16000
        self.initializer = args.initializer
        self.w_2 = torch.nn.Parameter(self.guass_paramset(args.filter_num, args.sigma_coff, self.initializer))
        self.filter_size = args.filter_size
        self.filter_number = args.filter_num
        self.NFFT = args.NFFT

    def forward(self, inputs):
        cn = self.w_2[:, 0]
        sigma = self.w_2[:, 1]

        filters = gaussianFreqFilters(cn, sigma, self.filter_size, self.filter_number, self.NFFT)
        filters = filters.transpose(1, 0)  # [Length,filters_numb]
        res = torch.matmul(inputs.permute(0, 2, 1), filters).permute(0, 2, 1)
        # sort_Loss = ParamIsIncreasedLoss(cn)*50000 + ParamIsLessZero(cn) + ParamIsMoreOne(cn) + ParamIsSparse(cn)*50
        sort_Loss = ParamIsIncreasedLoss(cn) + ParamIsLessZero(cn) + ParamIsMoreOne(cn)
        sigma_Loss = ParamIsLessZero(sigma)*999999
        return res, sort_Loss + sigma_Loss

    def guass_paramset(self, filter_num, sigma_coff, flag):
        """参数初始化阶段"""
        if flag == 'random':
            # cn = nn.init.uniform_(torch.empty(filter_num), 0, 1).cuda()
            cn = torch.linspace(0, 1, filter_num).cuda()
            sigma = torch.full((filter_num,), sigma_coff).cuda()
            sigma = torch.clamp(sigma, min=0.00001, max=0.5)
            cn = torch.clamp(cn, min=30 / 8000, max=1)
            sorted_indices = torch.argsort(cn)
            cn = torch.index_select(cn, 0, sorted_indices).unsqueeze(1)
            sigma = torch.index_select(sigma, 0, sorted_indices).unsqueeze(1)
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'mel_scale':
            """1. 梅尔初始化"""
            # High_Hz = self.sample_rate  # 限制其在高频内的范围
            High_Hz = 8000  # 限制其在高频内的范围
            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(1 + (High_Hz / 2) / 700)
            # high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
            mel_points = torch.linspace(low_freq_mel, high_freq_mel, filter_num)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            cn = (hz_points / (self.sample_rate / 2)).unsqueeze(1)  # 范围 [0, 1]
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'ERB':
            print("ERB Initialization")
            lowest_freq = 0  # Hz
            highest_freq = 8000  # Hz
            lowest_erb = hz_to_erb(lowest_freq)
            highest_erb = hz_to_erb(highest_freq)
            erb_values = torch.linspace(lowest_erb, highest_erb, filter_num)
            frequency_values = erb_to_hz(erb_values).unsqueeze(1)
            cn = (frequency_values / (self.sample_rate / 2))
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'Bark':
            print("Bark Initialization")
            lowest_freq = 0  # Hz
            highest_freq = 8000  # Hz
            bark_low = hz_to_bark(lowest_freq)
            bark_high = hz_to_bark(highest_freq)
            bark_values = torch.linspace(bark_low, bark_high, filter_num)
            frequency_values = bark_to_hz(bark_values)
            cn = (frequency_values / (self.sample_rate / 2)).unsqueeze(1)
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'Truncated Mel':
            print("Truncated Mel Initialization")
            High_Hz = 3000  # 限制其在高频内的范围
            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(1 + (High_Hz / 2) / 700)
            # high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
            mel_points = torch.linspace(low_freq_mel, high_freq_mel, filter_num)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            cn = (hz_points / (self.sample_rate / 2)).unsqueeze(1)  # 范围 [0, 1]
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'Truncated U-Hz':
            print("Truncated U-Hz Initialization")
            # cn = nn.init.uniform_(torch.empty(filter_num), 0, 1).cuda()
            cn = torch.linspace(0, 0.5, filter_num).cuda()
            sigma = torch.full((filter_num,), sigma_coff).cuda()
            sigma = torch.clamp(sigma, min=0.00001, max=0.5)
            cn = torch.clamp(cn, min=30 / 8000, max=1)
            sorted_indices = torch.argsort(cn)
            cn = torch.index_select(cn, 0, sorted_indices).unsqueeze(1)
            sigma = torch.index_select(sigma, 0, sorted_indices).unsqueeze(1)
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        return output


class GaussianShape(nn.Module):
    def __init__(self, args):
        super(GaussianShape, self).__init__()
        self.sample_rate = 16000
        self.initializer = args.initializer
        self.w = torch.nn.Parameter(self.guass_paramset(args.filter_num, args.sigma_coff, self.initializer))
        self.filter_size = args.filter_size
        self.filter_number = args.filter_num
        self.NFFT = args.NFFT

    def forward(self, inputs):
        cn = self.w[:, 0]
        sigma = self.w[:, 1]

        filters = gaussianFreqFilters(cn, sigma, self.filter_size, self.filter_number, self.NFFT)
        filters = filters.transpose(1, 0)  # [Length,filters_numb]
        res = torch.matmul(inputs.permute(0, 2, 1), filters).permute(0, 2, 1)
        # sort_Loss = ParamIsIncreasedLoss(cn)*50000 + ParamIsLessZero(cn) + ParamIsMoreOne(cn) + ParamIsSparse(cn)*50
        sort_Loss = ParamIsIncreasedLoss(cn) + ParamIsLessZero(cn) + ParamIsMoreOne(cn)
        sigma_Loss = ParamIsLessZero(sigma)*999999
        return res, sort_Loss + sigma_Loss

    def guass_paramset(self, filter_num, sigma_coff, flag):
        """参数初始化阶段"""
        if flag == 'random':
            # cn = nn.init.uniform_(torch.empty(filter_num), 0, 1).cuda()
            cn = torch.linspace(0, 1, filter_num).cuda()
            sigma = torch.full((filter_num,), sigma_coff).cuda()
            sigma = torch.clamp(sigma, min=0.00001, max=0.5)
            cn = torch.clamp(cn, min=30 / 8000, max=1)
            sorted_indices = torch.argsort(cn)
            cn = torch.index_select(cn, 0, sorted_indices).unsqueeze(1)
            sigma = torch.index_select(sigma, 0, sorted_indices).unsqueeze(1)
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'mel_scale':
            """1. 梅尔初始化"""
            # High_Hz = self.sample_rate  # 限制其在高频内的范围
            High_Hz = 8000  # 限制其在高频内的范围
            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(1 + (High_Hz / 2) / 700)
            # high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
            mel_points = torch.linspace(low_freq_mel, high_freq_mel, filter_num)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            cn = (hz_points / (self.sample_rate / 2)).unsqueeze(1)  # 范围 [0, 1]
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'ERB':
            print("ERB Initialization")
            lowest_freq = 0  # Hz
            highest_freq = 8000  # Hz
            lowest_erb = hz_to_erb(lowest_freq)
            highest_erb = hz_to_erb(highest_freq)
            erb_values = torch.linspace(lowest_erb, highest_erb, filter_num)
            frequency_values = erb_to_hz(erb_values).unsqueeze(1)
            cn = (frequency_values / (self.sample_rate / 2))
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'Bark':
            print("Bark Initialization")
            lowest_freq = 0  # Hz
            highest_freq = 8000  # Hz
            bark_low = hz_to_bark(lowest_freq)
            bark_high = hz_to_bark(highest_freq)
            bark_values = torch.linspace(bark_low, bark_high, filter_num)
            frequency_values = bark_to_hz(bark_values)
            cn = (frequency_values / (self.sample_rate / 2)).unsqueeze(1)
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'Truncated Mel':
            print("Truncated Mel Initialization")
            High_Hz = 3000  # 限制其在高频内的范围
            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(1 + (High_Hz / 2) / 700)
            # high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
            mel_points = torch.linspace(low_freq_mel, high_freq_mel, filter_num)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            cn = (hz_points / (self.sample_rate / 2)).unsqueeze(1)  # 范围 [0, 1]
            sigma = torch.ones((filter_num, 1)) * sigma_coff
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        elif flag == 'Truncated U-Hz':
            print("Truncated U-Hz Initialization")
            # cn = nn.init.uniform_(torch.empty(filter_num), 0, 1).cuda()
            cn = torch.linspace(0, 0.5, filter_num).cuda()
            sigma = torch.full((filter_num,), sigma_coff).cuda()
            sigma = torch.clamp(sigma, min=0.00001, max=0.5)
            cn = torch.clamp(cn, min=30 / 8000, max=1)
            sorted_indices = torch.argsort(cn)
            cn = torch.index_select(cn, 0, sorted_indices).unsqueeze(1)
            sigma = torch.index_select(sigma, 0, sorted_indices).unsqueeze(1)
            """拼接"""
            output = torch.cat([cn, sigma], dim=1)
        return output


"""若干个滤波器参数构造和搭建"""


def gaussianFreqFilters(cn, sigma, filter_size, nfilt, NFFT):
    """
    cn和sigma 参数限制在一定范围内
    Role:约束滤波器的参数大小值，保持滤波器的形状处在合理范围内。
    """
    sigma = torch.clamp(sigma, min=0.00001, max=1)
    cn = torch.clamp(cn, min=30 / 80000, max=1)
    """
    cn和sigma 参数重新排序
    Role:重新排序滤波器参数,保证滤波器中心频率能够正确/合理分布。
    """
    # 计算滤波器参数的排序索引
    # sorted_indices = torch.argsort(cn)
    # 重新排列滤波器参数
    # cn = torch.index_select(cn, 0, sorted_indices)
    # sigma = torch.index_select(sigma, 0, sorted_indices)
    # 计算滤波器参数
    fbanks = torch.zeros((nfilt, int(NFFT / 2 + 1))).cuda()
    # 长度约束
    half_filterLen = (((NFFT // 2) + 1) // nfilt) // 4
    for i in range(len(cn)):
        center_Location = cn[i] * filter_size  # 中心频率位置
        band_low = int(center_Location) - half_filterLen if int(center_Location) - half_filterLen >= 0 else 0
        band_high = (int(center_Location) + half_filterLen + 1) if int(center_Location) + half_filterLen < filter_size else filter_size - 1
        t = torch.arange(band_low, band_high, dtype=sigma.dtype, device=sigma.device) / filter_size
        """高斯滤波器"""
        # ===================================================
        numerator = t - cn[i]
        temp = 1 * (torch.exp(-0.5 * (numerator / sigma[i]) ** 2))  # 2 等于A 提升高斯的高度  sigma 要小。
        # ====================================================
        """三角滤波器"""
        # =====================================================
        # numerator = torch.abs(t - cn[i])
        # temp = torch.relu(1-2 * numerator / sigma[i])
        # =====================================================
        fbanks[i][band_low:band_high] = temp
        # fbanks[(nfilt-1)-i][band_low:band_high] = temp

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


# 约束增序
def ParamIsIncreasedLoss(x):
    diff = x[1:] - x[:-1]
    # 相邻元素都非递减
    if torch.all(diff >= 0):
        return 0
    loss = torch.abs(diff[diff < 0].sum())
    return loss


# 约束大于0
def ParamIsLessZero(x):
    if torch.all(x >= 0):
        return 0
    loss = torch.abs(x[x < 0].sum())
    return loss * 100


# 约束小于1
def ParamIsMoreOne(x):
    if torch.all(x <= 1):
        return 0
    loss = torch.abs(x[x > 1].sum())
    return loss


# L1 loss 约束稀疏

# 我们认为并不是全部参数都有用，因此我们约束滤波器的个数，
# 或者滤波器之间的距离近一些。
# 那是不是可以改用滤波器之间的距离作为可学习参数。
# 这个地方在差分处处理，差分大与0，动态参数w，用于控制滤波器之间的距离。
# 0<w<0.1//fft
# 初始化一个岂止最小频率。然后每个滤波器之间的间隔逐渐相加。

def ParamIsSparse(x):
    loss_l1 = torch.norm(x, p=1)
    return loss_l1


def freqResIsSparse(x):
    loss_l1 = torch.norm(x, p=1)
    return loss_l1

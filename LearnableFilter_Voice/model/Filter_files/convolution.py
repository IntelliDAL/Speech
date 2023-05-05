import numpy as np
import torch
import math
from typing import Tuple, Callable
from torch import nn
from torch.autograd import Variable
from model.Filter_files.impulse_responses import gabor_filters
from model.Filter_files.utils import get_padding_value


class GaborConstraint(nn.Module):
    def __init__(self, kernel_size):
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, kernel_data):
        mu_lower = 0.
        mu_upper = math.pi

        sigma_lower = 4 * torch.sqrt(2. * torch.log(torch.tensor(2., device=kernel_data.device))) / math.pi
        sigma_upper = self._kernel_size * torch.sqrt(
            2. * torch.log(torch.tensor(2., device=kernel_data.device))) / math.pi

        clipped_mu = torch.clamp(kernel_data[:, 0], mu_lower, mu_upper).unsqueeze(1)
        clipped_sigma = torch.clamp(kernel_data[:, 1], sigma_lower, sigma_upper).unsqueeze(1)

        return torch.cat([clipped_mu, clipped_sigma], dim=-1)


class freqNorm_Const(nn.Module):
    def __init__(self, kernel_size, sample_rate):
        super(freqNorm_Const, self).__init__()
        self._kernel_size = kernel_size
        self._sample_rate = sample_rate
        self.min_band_hz = 50 / (sample_rate / 2)
        self.min_freq = 30 / (sample_rate / 2)

    def forward(self, kernel_data):
        """
        我们约束 f_l > 0 and f_w > f_l
        中心频率的限制范围[0, 1/2]
        """
        # ================================================= 裁减
        # 裁减范围
        Lower, Upper = self.min_freq, 0.5

        Low, High = kernel_data[:, 0], kernel_data[:, 1]

        clipped_Lfreq = torch.clamp(Low, Lower, Upper)
        """最小和f_low 相等, 
        最高要限制其在 乃奎斯特采样率的一半
        必须要保证高频信息比低频信息大
        """

        band_hz_ = torch.abs(High - Low)

        """对低频和高频信息添加约束"""
        # clipped_Hfreq = torch.clamp(High, Lower, Upper).unsqueeze(1)
        clipped_Hfreq = (torch.clamp(Low + self.min_band_hz + band_hz_, Lower, Upper))
        # =================================== Normalization
        # theta 和 bandwidth之间的关系
        coeff = torch.sqrt(2. * torch.log(torch.tensor(2.)))
        bandwidth = (clipped_Hfreq - clipped_Lfreq)
        # bandwidth = (High - Low)[:, 0]
        Thetas = coeff / (bandwidth * torch.pi)
        Center_frequencies = (((clipped_Hfreq + clipped_Lfreq) / 2) * (2 * torch.pi))  # 每个频点的频率

        """# parameters in radians 将弧度化的中心频率和参数进行返回"""
        output = torch.cat([Center_frequencies.unsqueeze(1), Thetas.unsqueeze(1)], dim=1)
        return output


class GaborConv1d(nn.Module):
    def __init__(self, filters, kernel_size, strides, sample_rate, padding, initializer=None, use_bias=False, sort_filters=False, use_legacy_complex=False):
        super(GaborConv1d, self).__init__()
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._sort_filters = sort_filters
        self._sample_rate = sample_rate

        if isinstance(initializer, Callable):
            init_weights = initializer((self._filters, 2))
        elif initializer == "random":
            init_weights = torch.randn(self._filters, 2)
        elif initializer == "xavier_normal":
            print("Using xavier_normal init scheme..")
            init_weights = torch.randn(self._filters, 2)
            init_weights = torch.nn.init.xavier_normal_(init_weights)
        elif initializer == "kaiming_normal":
            print("Using kaiming_normal init scheme..")
            init_weights = torch.randn(self._filters, 2)
            init_weights = torch.nn.init.kaiming_normal_(init_weights)
        else:
            raise ValueError("unsupported initializer")

        self.kernel = nn.Parameter(init_weights)  # 初始化采参数
        # self.constraint = GaborConstraint(self._kernel_size)
        self.freqNorm_Const = freqNorm_Const(self._kernel_size, self._sample_rate)
        # ------------------------------------------------
        # ------------------------------------------------
        if self._padding.lower() == "same":
            self._pad_value = get_padding_value(self._kernel_size)
        else:
            self._pad_value = self._padding
        if self._use_bias:
            self._bias = torch.nn.Parameter(torch.ones(self._filters * 2, ))
        else:
            self._bias = None

        self.use_legacy_complex = use_legacy_complex
        if self.use_legacy_complex:
            print("ATTENTION: Using legacy_complex format for gabor filter estimation.")

    def forward(self, x):
        """每次初始化参数, 然后施加约束, 重新赋值给Gabor"""
        """# apply Gabor constraint"""
        # kernel = self.constraint(self.kernel)
        kernel = self.freqNorm_Const(self.kernel)

        if self._sort_filters:
            raise NotImplementedError("sort filter functionality not yet implemented")
        filters = gabor_filters(kernel, self._kernel_size, legacy_complex=self.use_legacy_complex)

        if not self.use_legacy_complex:
            temp = torch.view_as_real(filters)
            real_filters = temp[:, :, 0]
            img_filters = temp[:, :, 1]
        else:
            real_filters = filters[:, :, 0]
            img_filters = filters[:, :, 1]

        stacked_filters = torch.cat([real_filters.unsqueeze(1), img_filters.unsqueeze(1)], dim=1)
        stacked_filters = torch.reshape(stacked_filters, (2 * self._filters, self._kernel_size))
        stacked_filters = stacked_filters.unsqueeze(1)
        if self._padding.lower() == "same":
            x = nn.functional.pad(x, self._pad_value, mode='constant', value=0)
            pad_val = 0
        else:
            pad_val = self._pad_value

        output = nn.functional.conv1d(x, stacked_filters, bias=self._bias, stride=self._strides, padding=pad_val)
        return output


class SincConv1d(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs, strides, groups):
        super(SincConv1d, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.strides = strides
        self.groups = groups
        # self._bias = torch.nn.Parameter(torch.ones(self.N_filt, ))

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)
        n = torch.linspace(0, N, steps=N)
        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N)
        window = Variable(window.float().cuda())
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / torch.max(band_pass)
            filters[i, :] = band_pass.cuda() * window
        # out = nn.functional.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), stride=self.strides, groups=self.groups, bias=self._bias)
        out = nn.functional.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), stride=self.strides, groups=self.groups)

        return out


def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)
    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
    return y


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class GammaToneConv1d(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(GammaToneConv1d, self).__init__()

        # Initialization of the filterbanks
        nb_filters = N_filt
        PARAMETER_NUMBER = 3
        b = np.arange(0.03, 0.1, 0.01)
        f = np.linspace(0, 0.5, nb_filters, endpoint=False)
        x = np.zeros((PARAMETER_NUMBER, nb_filters))

        for index in range(nb_filters):
            x[0, index] = f[index]
            x[1, index] = float(1 / nb_filters / 4)
            x[2, index] = 4.0  # order starts at 2
        self.weight = x.transpose()

        self.freq_scale = fs * 1.0
        self.f = nn.Parameter(torch.from_numpy(self.weight[:, 0]))
        self.b = nn.Parameter(torch.from_numpy((self.weight[:, 1])))
        self.order = nn.Parameter(torch.from_numpy((self.weight[:, 2])))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()

        N = self.Filt_dim
        t = Variable(torch.linspace(0, N, steps=int(N)).cuda())
        t = t / self.fs

        for i in range(self.N_filt):
            f = self.f[i]
            b = self.b[i]
            order = self.order[i]
            alpha = order - 1
            gtone = t ** alpha * torch.exp(-2 * math.pi * b * t) * torch.cos(2 * math.pi * f * t)
            # gnorm = (4 * math.pi * b) ** ((2 * alpha + 1) / 2) / torch.sqrt(torch.exp(torch.lgamma(2 * alpha + 1))) * np.sqrt(2)
            kernel = gtone * 1
            kernel = kernel / torch.norm(kernel)
            filters[i, :] = kernel

        out = torch.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), stride=1)

        return out


### GAMMATONE IMPULSE RESPONSE ###

def gammatone_impulse_response(samplerate_hz, length_in_seconds, center_freq_hz):
    # Generate single parametrized gammatone filter
    p = 2  # filter order
    erb = 24.7 + 0.108 * center_freq_hz  # equivalent rectangular bandwidth
    divisor = (np.pi * np.math.factorial(2 * p - 2) * np.power(2, float(-(2 * p - 2)))) / np.square(
        np.math.factorial(p - 1))
    b = erb / divisor  # bandwidth parameter
    a = 1.0  # amplitude. This is varied later by the normalization process.
    L = int(np.floor(samplerate_hz * length_in_seconds))
    t = np.linspace(1. / samplerate_hz, length_in_seconds, L)
    gammatone_ir = a * np.power(t, p - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * center_freq_hz * t)
    return gammatone_ir


class WaveletConv1d(nn.Module):
    """Wavelet filter.
    From the paper "Learning filter widths of spectral decompositions with wavelets." NeurIPS 2018.
    cf https://github.com/haidark/WaveletDeconv
    """
    # parameters: s - scale
    PARAMETER_NUMBER = 1
    PARAMETER_NAMES = ['Scale']
    COSINE_AND_SINE_FILTER = False

    def __init__(self, N_filt, Filt_dim, fs):
        super(WaveletConv1d, self).__init__()
        self.fs = fs
        self.filter_length = Filt_dim
        self.type = 'Wavelet'
        nb_filters = N_filt
        x = torch.logspace(0, 1.4, nb_filters).cuda()
        weight = torch.reshape(x, (len(x), 1))
        self.weight = nn.Parameter(weight)
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t = Variable(torch.linspace(0, N, steps=int(N)) / self.fs).cuda()

        for i in range(self.N_filt):
            t2 = t ** 2
            scale = self.weight[i]
            scale2 = scale ** 2
            B = (3 * scale) ** 0.5
            A = (2 / (B * (math.pi ** 0.25)))
            mod = (1 - t2 / scale2)
            gauss = torch.exp(-t2 / (2 * scale2))
            kern = A * mod * gauss
            kernel = torch.reshape(kern, (len(t), 1))
            kernel = kernel / torch.norm(kernel)
            filters[i, :] = kernel.squeeze()
        out = torch.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), stride=1)
        return out

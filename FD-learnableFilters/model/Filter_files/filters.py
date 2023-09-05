import math
import torch
import numpy as np
import torchaudio

import model.Filter_files.impulse_responses as impulse_responses
from torch import nn


class GaborFilter:
    def __init__(self, n_filters: int = 40, min_freq: float = 0., max_freq: float = 8000., sample_rate: int = 16000,
                 window_len: int = 401, n_fft: int = 512, normalize_energy: bool = False):
        super(GaborFilter, self).__init__()
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        min_band_hz = 50
        self.min_band_hz = min_band_hz
        self.window_len = window_len
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy
        self.constraint = FreqConstraint(self.window_len, sample_rate)
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def gabor_params_from_mels(self):
        """
        nfft==512
        初始化方式主要是 调用Mel-filter 方法. 具体shape 为[40,257 ]
        理解:
        在整个频率的采样范围0-8000内. fft变换后,具体频率被等分成257 (n_fft/2+1) 份
        :return:
        """

        # parameters in radians
        coeff = torch.sqrt(2. * torch.log(torch.tensor(2.))) * self.n_fft

        Mel_filters = self.mel_filters()  # [40, 257]
        sqrt_filters = torch.sqrt(Mel_filters)

        center_frequencies = torch.argmax(sqrt_filters, dim=1)

        peaks, _ = torch.max(sqrt_filters, dim=1, keepdim=True)
        half_magnitudes = peaks / 2.  # fwhm 表示的是半高宽度,因此取值一半.

        demo = sqrt_filters >= half_magnitudes.float()
        """!!!!!!----此处的带宽和中心频率并未转换为频率"""
        width = torch.sum(demo, dim=1)  # 选取所有滤波器中超过fwhm的部分,即为带宽

        # 使用美尔滤波器的fwhm初始化

        # parameters in radians
        # width and center
        output = torch.cat([(center_frequencies * 2 * np.pi / self.n_fft).unsqueeze(1), (coeff / (np.pi * width)).unsqueeze(1)], dim=-1)
        # 通过参数化中心频率和带宽进行优化。
        print('Mel initializers')
        return output

    def _mel_filters_areas(self, filters):
        peaks, _ = torch.max(filters, dim=1, keepdim=True)
        return peaks * (torch.sum((filters > 0).float(), dim=1, keepdim=True) + 2) * np.pi / self.n_fft

    def gabor_filters(self):
        gabor_filters = impulse_responses.gabor_filters(self.gabor_params_from_mels, size=self.window_len)
        output = gabor_filters * torch.sqrt(self._mel_filters_areas(self.mel_filters) * 2 * math.sqrt(math.pi) * self.gabor_params_from_mels[:, 1:2]).type(torch.complex64)
        return output

    def mel_filters(self):
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.n_filters,
            sample_rate=self.sample_rate
        )
        mel_filters = mel_filters.transpose(1, 0)
        if self.normalize_energy:
            mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
        return mel_filters

    def gabor_params_from_LHFreq(self):
        """
        min_freq = 0.0
        max_freq = 8000.0
        n_fft = self.n_fft
        我们根据采样定理.音频的范围就是在 : min_freq = 0.0-max_freq = 8000.0
         max_freq = (self.sample_rate/2) ==8000
        """
        """# 在频率空间内初始化filters数量的频率个数"""
        # Mel = np.linspace(self.to_mel(self.min_freq), self.to_mel(self.max_freq), self.n_filters + 1)
        # Hz = self.to_hz(Mel)
        Hz = np.linspace(self.min_freq, self.max_freq, self.n_filters + 1)

        """# 低截止频率和高截止频率"""
        Low_hz_ = torch.Tensor(Hz[:-1]).view(-1, 1)

        Low = ((self.min_freq + torch.abs(Low_hz_)) / self.sample_rate).cuda()

        """  filter frequency band (out_channels, 1) 
        主要目: a+(b-a) = b
        # 差分运算
        """
        # 初始化频率之间的间隔 L和H频率之间的覆盖
        self.band_hz_ = torch.Tensor(np.diff(Hz)).view(-1, 1)

        """a+ (b-a) = b"""
        High = (torch.clamp(Low_hz_ + self.min_band_hz + torch.abs(self.band_hz_), self.min_freq, self.sample_rate / 2) / self.sample_rate).cuda()

        output = torch.cat([Low, High], dim=1)
        print('Low_and_High_initializers')
        return output


class FreqConstraint(nn.Module):
    def __init__(self, kernel_size, sample_rate):
        super(FreqConstraint, self).__init__()
        self._kernel_size = kernel_size
        self._sample_rate = sample_rate

    def forward(self, Low, High):
        """
        我们约束 f_l > 0 and f_w > f_l
        中心频率的限制范围[0, 1/2]
        """
        Lower, Upper = 0., 0.5
        clipped_Lfreq = torch.clamp(Low, Lower, Upper).unsqueeze(1)

        """最小和f_low 相等, 最高要限制其在 乃奎斯特采样率的一半"""
        clipped_sigma = torch.clamp(High, Lower, Upper).unsqueeze(1)

        return clipped_Lfreq, clipped_sigma
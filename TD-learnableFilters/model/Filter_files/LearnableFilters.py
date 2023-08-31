import torch
from torch import nn
from model.Filter_files import convolution
from model.Filter_files import initializers
from model.Filter_files import pooling
from model.Filter_files import postprocessing
import matplotlib.pyplot as plt
import numpy as np


class gaussian_lowpassaHY(nn.Module):
    def __init__(self):
        super(gaussian_lowpass, self).__init__()
        w1 = nn.Parameter(torch.ones(64,1))*0.5
        w2 = nn.Parameter(torch.ones(64, 1))
        self.sigma = w1.cuda()
        self.cn = w2.cuda() # 这个地方应该使用插值  lin
        self.filter_size = 257

    def gausfilter(self):
        #sigma = torch.clamp(sigma, min=(2. / filter_size), max=0.5)
        # t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device)
        # t = torch.reshape(t, (1, filter_size, 1, 1))
        t = torch.arange(0, self.filter_size, dtype=self.sigma.dtype, device=self.sigma.device)
        # numerator = t - 0.5 * (filter_size - 1)
        # denominator = sigma * 0.5 * (filter_size - 1)
        numerator = t - self.cn
        #denominator = self.sigma
        return torch.exp(-0.5 * (numerator / self.sigma) ** 2)

    def forward(self,x):
        gauss = self.gausfilter()
        temp = torch.unsqueeze(gauss,0).permute(0,2,1)
        res = torch.matmul(x,temp)
        return res.permute(0,2,1)


class SquaredModulus(nn.Module):
    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = 2 * self._pool(x ** 2.)
        output = output.transpose(1, 2)
        return output


class Modulus(nn.Module):
    def __init__(self):
        super(Modulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=1, stride=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = self._pool(x ** 2.)
        output = output.transpose(1, 2)
        return output


def gaussian_lowpass(cn, sigma, filter_size,flag):
    res = torch.empty(0,filter_size).cuda()
    for i in range(len(cn)):
        t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device) / filter_size
        numerator = t - cn[i]
        temp = torch.exp(-0.5 * (numerator / sigma[i]) ** 2)
        band_low = int(cn[i]*filter_size)-25 if int(cn[i]*filter_size)-25 >= 0 else 0
        band_high = int(cn[i]*filter_size)+25 if int(cn[i]*filter_size)+25 < 257 else 256
        value_seg = temp[band_low:band_high]
        t1 = torch.zeros(band_low).cuda()
        t2 = torch.zeros(filter_size-band_high).cuda()
        value = torch.cat((torch.cat((t1,value_seg),dim=0),t2),dim = 0)
        if flag is 'test':
            show_gaussian_function(t,value,cn[i],sigma[i])
        res = torch.cat((res,value.unsqueeze(0)),dim = 0)
    return res

#高斯函数矩阵运算
# def gaussian_lowpass(cn, sigma, filter_size,flag):
#     t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device) / filter_size
#     temp = torch.Tensor(len(cn),filter_size).copy_(t).cuda()
#     numerator = temp - cn.reshape(-1,1)
#     res = torch.exp(-0.5 * (numerator / sigma.reshape(-1,1)) ** 2)
#     if flag is 'test':
#         for i in range(len(cn)):
#             show_gaussian_function(temp[i], res[i], cn[i], sigma[i])
#     return res

def triangle_filter(alpha, beta, filter_size,flag):
    res = torch.empty(0, filter_size).cuda()
    for i in range(len(alpha)):
        t = torch.arange(0, filter_size, dtype=beta.dtype, device=beta.device) / filter_size  #这里是否还需要归一化？？？
        numerator = torch.abs(t - alpha[i])
        temp = torch.relu(1-2 * numerator / beta[i])
        if flag is 'test':
            show_gaussian_function(t, temp, alpha[i], beta[i])
        res = torch.cat((res, temp.unsqueeze(0)), dim=0)
    return res


def show_gaussian_function(x,y,cn,sigma):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决pythonmatplotlib绘图无法显示中文的问题
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(x, y, 'b-', linewidth=2)
    strcn = str(cn)[7:13]
    strsigma = str(sigma)[7:13]
    plt.title("cn="+strcn+" sigma="+strsigma)
    plt.xlim(0, 1.5)
    plt.ylim(0, 1.5)
    #plt.legend(labels=['$\mu = '+strcn+'$'+', \sigma='+strsigma+'$'])
    plt.savefig('./gaussPic/'+"cn="+strcn+" sigma="+strsigma+'.png')
    #plt.show()
    plt.close()


class LearnableFilters(nn.Module):
    def __init__(self,  n_filters: int = 40, sample_rate: int = 16000, window_len: float = 25.,  window_stride: float = 10., preemp: bool = False,  init_min_freq=60.0, init_max_freq=7800.0, mean_var_norm: bool = False, pcen_compression: bool = True, use_legacy_complex=False, initializer="kaiming_normal"):
        super(LearnableFilters, self).__init__()
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)
        if preemp:
            raise NotImplementedError("Pre-emp functionality not implemented yet..")
        else:
            self._preemp = None
        if initializer == "default":
            initializer = initializers.GaborInit(default_window_len=window_size, sample_rate=sample_rate,  min_freq=init_min_freq, max_freq=init_max_freq)
        elif initializer == "LowAndHighFreq":
            initializer = initializers.LowAndHighFreq(default_window_len=window_size, sample_rate=sample_rate,  min_freq=init_min_freq, max_freq=init_max_freq)
        # --------------------------------------------------
        self.complex_conv = convolution.GaborConv1d(filters=2 * n_filters, kernel_size=window_size, strides=1, sample_rate=sample_rate, padding="same", use_bias=False, initializer=initializer, use_legacy_complex=use_legacy_complex)
        self.activation = SquaredModulus()
        self.pooling = pooling.Adaptpooling(n_filters, kernel_size=window_size, strides=window_stride, padding="same")
        self._instance_norm = None
        if mean_var_norm:
            raise NotImplementedError("Instance Norm functionality not added yet..")
        if pcen_compression:
            self.compression = postprocessing.PCENLayer(n_filters, alpha=0.96, smooth_coef=0.025, delta=2.0, floor=1e-12, trainable=True, learn_smooth_coef=True, per_channel_smooth_coef=True)
        else:
            self.compression = postprocessing.LogTBN(n_filters, a=5, trainable=True, per_band=True, median_filter=False, append_filtered=False)
        self._maximum_val = torch.tensor(1e-5)

    def forward(self, x):
        if self._preemp:
            x = self._preemp(x)
        outputs = self.complex_conv(x)
        outputs = self.activation(outputs)
        outputs = self.pooling(outputs)
        outputs = torch.maximum(outputs, torch.tensor(1e-5, device=outputs.device))
        outputs = self.compression(outputs)
        if self._instance_norm is not None:
            outputs = self._instance_norm(outputs)
        return outputs

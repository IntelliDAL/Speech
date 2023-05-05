import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def sinc(self, x):
    sinc = torch.sin(x) / x
    sinc[:, self.kernel_size // 2] = 1.
    return sinc


def SincConv(out_channels, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50):


    # Hamming window
    window_ = torch.hamming_window(kernel_size)

    # (kernel_size, 1)
    n = (kernel_size - 1) / 2
    n_ = torch.range(-n, n).view(1, -1) / sample_rate

    low = min_low_hz / sample_rate + torch.abs(low_hz_)

    f_times_t = torch.matmul(low, n_)
    low_pass1 = 2 * low * sinc(2 * math.pi * f_times_t * sample_rate)

    max_, _ = torch.max(low_pass1, dim=1, keepdim=True)
    low_pass1 = low_pass1 / max_

    outputs = (low_pass1 * window_).view(out_channels, 1, kernel_size)

    return outputs

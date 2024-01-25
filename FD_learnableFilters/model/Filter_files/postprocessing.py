import numpy as np
import torch
from torch import nn
from typing import Optional, Union


class ExponentialMovingAverage(nn.Module):
    def __init__(self, in_channels, coeff_init, per_channel: bool = False):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        weights = torch.ones(in_channels, ) if self._per_channel else torch.ones(1, )
        self._weights = nn.Parameter(weights * self._coeff_init)

    def forward(self, x):
        w = torch.clamp(self._weights, min=0.00001, max=1.)
        initial_state = x[:, :, 0]

        def scan(init_state, x, w):
            x = x.permute(2, 0, 1)
            acc = init_state
            results = []
            for ix in range(len(x)):
                acc = (w * x[ix]) + ((1.0 - w) * acc)
                results.append(acc.unsqueeze(0))
            results = torch.cat(results, dim=0)
            results = results.permute(1, 2, 0)
            return results

        return scan(initial_state, x, w)


class PCENLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 alpha: float = 0.98,
                 smooth_coef: float = 0.04,
                 delta: float = 2.0,
                 root: float = 2.0,
                 floor: float = 1e-6,
                 trainable: bool = False,
                 learn_smooth_coef: bool = False,
                 per_channel_smooth_coef: bool = False):
        super(PCENLayer, self).__init__()
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable
        self._learn_smooth_coef = learn_smooth_coef
        self._per_channel_smooth_coef = per_channel_smooth_coef

        self.alpha = nn.Parameter(torch.ones(in_channels) * self._alpha_init)
        self.delta = nn.Parameter(torch.ones(in_channels) * self._delta_init)
        self.root = nn.Parameter(torch.ones(in_channels) * self._root_init)

        if self._learn_smooth_coef:
            self.ema = ExponentialMovingAverage(in_channels, coeff_init=self._smooth_coef,  per_channel=self._per_channel_smooth_coef)
        else:
            raise ValueError("SimpleRNN based ema not implemented.")

    def forward(self, x):
        """这个地方不能直接切断，或出现梯度问题  依旧是使用loss 进行约束"""
        # [0, 1] models the degree  of compression
        alpha = torch.max(torch.min(self.alpha, torch.tensor(1.0, dtype=x.dtype, device=x.device)), torch.tensor(0.0, dtype=x.dtype, device=x.device))
        # root (0, 1] are the main control parameters
        root = torch.max(self.root, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        # positive bias term delta>1
        delta = torch.max(self.delta, torch.tensor(1.0,  dtype=x.dtype, device=x.device))
        # delta = self.delta
        """Loss delta"""
        loss_delta = ParamIsMoreOne(delta)
        # pre-set smoothing coefficient
        ema_smoother = self.ema(x)
        one_over_root = 1. / root
        output = (
                (x / (self._floor + ema_smoother) ** alpha.view(1, -1, 1) + delta.view(1, -1, 1)) ** one_over_root.view(1, -1, 1)
                - delta.view(1, -1, 1) ** one_over_root.view(1, -1, 1)
        )
        return output, loss_delta


# 约束小于1
def ParamIsMoreOne(x):
    if torch.all(x >= 1):
        return 0
    loss = torch.abs(x[x < 1].sum())
    return loss


# Log1p + Median filter + TBN (temporal batch normalization) compression function
class LogTBN(nn.Module):
    """
    Calculates the Log1p of the input signal, optionally subtracts the median
    over time, and finally applies batch normalization over time.
    :param num_bands: number of filters
    :param affine: learnable affine parameters for TBN
    :param a: value for 'a' for Log1p
    :param trainable: sets 'a' trainable for Log1p
    :param per_band: separate 'a' per band for Log1p
    :param median_filter: subtract the median of the signal over time
    :param append_filtered: if true-ish, append the median-filtered signal as an additional channel instead of subtracting the median in place
    """

    def __init__(self, num_bands: int, affine: bool = True, a: float = 0, trainable: bool = False,
                 per_band: bool = False, median_filter: bool = False, append_filtered: bool = False):
        super(LogTBN, self).__init__()
        self.log1p = Log1p(a=a, trainable=trainable, per_band=per_band, num_bands=num_bands)
        self.TBN = TemporalBatchNorm(num_bands=num_bands, affine=affine, per_channel=append_filtered,
                                     num_channels=2 if append_filtered else 1)
        self.median_filter = median_filter
        self.append_filtered = append_filtered

    def forward(self, x):
        x = self.log1p(x)
        if self.median_filter:
            if self.append_filtered and x.ndim == 3:
                x = x[:, np.newaxis]  # add channel dimension
            m = x.median(-1, keepdim=True).values
            if self.append_filtered:
                x = torch.cat((x, x - m), dim=1)
            else:
                x = x - m
        x = self.TBN(x)
        return x


class Log1p(nn.Module):
    """
    Applies `log(1 + 10**a * x)`, with `a` fixed or trainable.
    If `per_band` and `num_bands` are given, learn `a` separately per band.
    :param a: value for 'a'
    :param trainable: sets 'a' trainable
    :param per_band: separate 'a' per band
    :param num_bands: number of filters
    """

    def __init__(self, a=0, trainable=False, per_band=False, num_bands=None):
        super(Log1p, self).__init__()
        if trainable:
            dtype = torch.get_default_dtype()
            if not per_band:
                a = torch.tensor(a, dtype=dtype)
            else:
                a = torch.full((num_bands,), a, dtype=dtype)
            a = nn.Parameter(a)
        self.a = a
        self.trainable = trainable
        self.per_band = per_band

    def forward(self, x):
        if self.trainable or self.a != 0:
            a = self.a[:, np.newaxis] if self.per_band else self.a
            x = 10 ** a * x
        return torch.log1p(x)

    def extra_repr(self):
        return 'trainable={}, per_band={}'.format(repr(self.trainable), repr(self.per_band))


class TemporalBatchNorm(nn.Module):
    """
    Batch normalization of a (batch, channels, bands, time) tensor over all but
    the previous to last dimension (the frequency bands). If per_channel is
    true-ish, normalize each channel separately instead of joining them.
    :param num_bands: number of filters
    :param affine: learnable affine parameters
    :param per_channel: normalize each channel separately
    :param num_channels: number of input channels
    """

    def __init__(self, num_bands: int, affine: bool = True, per_channel: bool = True,
                 num_channels: Optional[int] = None):
        super(TemporalBatchNorm, self).__init__()
        num_features = num_bands * num_channels if per_channel else num_bands
        self.bn = nn.BatchNorm1d(num_features, affine=affine)
        self.per_channel = per_channel

    def forward(self, x):
        shape = x.shape
        if self.per_channel:
            # squash channels into the bands dimension
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        else:
            # squash channels into the batch dimension
            x = x.reshape((-1,) + x.shape[-2:])
        # pass through 1D batch normalization
        x = self.bn(x)
        # restore squashed dimensions
        return x.reshape(shape)

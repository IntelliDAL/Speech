import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    # 这个地方有两个窗口, 一个是patch内像素分割的窗口, 另一个是块间的窗口，用于计算注意力。

    def __init__(self, input_resolution, dim, block, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.block = block
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.window_size = 4
        self.patch_window = 4
        self.num_heads = 4
        # dim 表示的是输入通道的维度?????????  在本工作中,我们考虑的应该是当前窗口内的所有特征点.
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads, qkv_bias=True, qk_scale=True, attn_drop=0.1, proj_drop=0.2)
        # 关于这个窗口的重点内容应该聚焦于如何探讨窗口内的useful windows
        self.Unfold = nn.Unfold(kernel_size=(self.patch_window, self.patch_window), stride=(self.patch_window, self.patch_window))
        self.Fold = nn.Fold(output_size=(input_resolution[0] // 2, input_resolution[1] // 2), kernel_size=(self.patch_window, self.patch_window), stride=(self.patch_window, self.patch_window))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, _, _ = x.shape
        # 分块
        x = x.unsqueeze(1)  # B C Freq  Frame  where C=1
        x = self.Unfold(x).contiguous()  # 输出格式：batch, 展开后的维度, block number
        x = x.reshape(B, -1, self.input_resolution[0] // self.patch_window, self.input_resolution[1] // self.patch_window)
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, F = x.shape
        # -------------------------------------------------
        "块"
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, F)  # nW*B, window_size*window_size, C
        # 注意力机制针对的是分块后的每个块 就是哪一部分的块（patch-->frequency and frame）更重要。
        x, attn = self.attn(x)
        # merge windows
        x = x.view(-1, self.window_size, self.window_size, F)
        x = window_reverse(x, self.window_size, H, W)
        # -------------------------------------------------
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # 所有行
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # 所有列
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        """
        目标： 选择🈶的特征进行保留。
        按照块去选择，上下左右四个块 根据网络选择其中useful的特征
        在特征中进行拼接，
        按照汉和列单独选择
        """
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*F  # 特征维度的拼接
        x = self.norm(x)
        x = self.reduction(x)  # B, H//2, W//2, F
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, self.patch_window ** 2, -1).contiguous()
        x = self.Fold(x)
        return x.squeeze()

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# 按照窗口分区
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # 坐标coordinates
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='xy'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch-script happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        """位置编码作用提升2-3%"""
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
from itertools import repeat
import collections.abc
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    # 这个地方有两个窗口, 一个是patch内像素分割的窗口, 另一个是块间的窗口，用于计算注意力。

    def __init__(self, input_resolution, dim, block, flag, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=True)
        self.norm = norm_layer(4 * dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(input_resolution[1] // 2)
        self.window_size = 4
        self.patch_window = 4
        self.num_heads = 4
        # dim 表示的是输入通道的维度  在本工作中,我们考虑的应该是当前窗口内的所有特征点.
        # =========================================================
        self.branch_num = 2
        self.split_size = 8
        head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        qk_scale = head_dim ** -0.5
        if block == flag - 1:
            self.attn_crossTF = nn.ModuleList([
                CSWAttention(dim, resolution=(
                input_resolution[0] // self.patch_window, input_resolution[1] // self.patch_window),
                             idx=-1, split_size=self.split_size, num_heads=self.num_heads, dim_out=dim,
                             qk_scale=qk_scale, attn_drop=0.1)])
        else:
            self.attn_crossTF = nn.ModuleList([
                CSWAttention(dim // 2, resolution=(
                input_resolution[0] // self.patch_window, input_resolution[1] // self.patch_window),
                             idx=i, split_size=self.split_size, num_heads=self.num_heads // 2, dim_out=dim // 2,
                             qk_scale=qk_scale, attn_drop=0.1)
                for i in range(self.branch_num)])
        # ==============================================================================================================
        self.conv_embed = nn.Sequential(
            nn.Conv2d(1, self.dim, kernel_size=(self.patch_window, self.patch_window),
                      stride=(self.patch_window, self.patch_window)),
            nn.GELU(),  # 先应用ReLU激活函数
            nn.BatchNorm2d(self.dim),
            Rearrange('b c h w -> b (h w) c'),
        )
        # 关于这个窗口的重点内容应该聚焦于如何探讨窗口内的useful windows
        self.deconv_embed = nn.Sequential(
            nn.ConvTranspose2d(self.dim * 2, 1, kernel_size=(self.patch_window, self.patch_window),
                               stride=(self.patch_window, self.patch_window)),
            nn.GELU(),  # 先应用ReLU激活函数
            nn.BatchNorm2d(1)  # 然后是批归一化
        )
        mlp_ratio = 4
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=nn.GELU)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, ori_x):
        """
        x: B, H*W, C=1
        """
        B, h_featmap, w_featmap = x.shape  # batch_size, filter_number, frame_length
        # 分块
        x0 = x.unsqueeze(1)  # B C filter_number  frame_length  where C=1
        x1 = self.conv_embed(x0).contiguous()  # 输出格式：batch, 展开后的维度, block total number
        x2 = x1.view(B, -1, self.input_resolution[0] // self.patch_window,
                     self.input_resolution[1] // self.patch_window)  # 输出格式：batch, 展开后的维度, width_num, length_num
        x3 = x2.permute(0, 2, 3, 1).contiguous()  # 输出格式：batch, width_num, length_num, 展开后的维度
        B, H, W, C = x3.shape  # windows size
        # ------------------此处以上，为输入特征图的embed------------------------------------
        qkv = self.qkv(x3.view(B, -1, C)).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # Q K V according  to the dim
        x6_t = self.attn_crossTF[0](qkv[:, :, :, :C // 2])  # 水平
        x6_f = self.attn_crossTF[1](qkv[:, :, :, C // 2:])  # 垂直
        x6 = torch.cat([x6_t, x6_f], dim=2)
        # ---------------------------------------------------------------=====================================
        x6 = self.proj(x6)
        x6 = self.proj_drop(x6)
        x6 = self.mlp(self.norm1(x6))  # 合并后整合
        # ---------------------------------------------------------------=====================================
        # merge windows
        x7 = x6.view(-1, self.window_size, self.window_size, C)
        x8 = window_reverse(x7, self.window_size, H, W)
        # -------------------------------------------------
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # 所有行
        x_0 = x8[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x_1 = x8[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # 所有列
        x_2 = x8[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x_3 = x8[:, 1::2, 1::2, :]  # B H/2 W/2 C
        """
        目标： 选择🈶的特征进行保留。
        按照块去选择，上下左右四个块 根据网络选择其中useful的特征
        在特征中进行拼接，
        按照行和列单独选择
        """
        x9 = torch.cat([x_0, x_1, x_2, x_3], -1)  # B H/2 W/2 4*F  # 特征维度的拼接
        x10 = self.norm(x9)
        x11 = self.reduction(x10)  # B, H//2, W//2, F
        # -------------------------------------------------
        x12 = x11.permute(0, 3, 1, 2)
        # 还原为image
        x13 = self.deconv_embed(x12)

        return x13.squeeze()

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CSWAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Spec2Window
        H, W = self.resolution[0], self.resolution[1]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q, H, W)
        k = self.im2cswin(k, H, W)
        v, lepe = self.get_lepe(v, self.get_v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

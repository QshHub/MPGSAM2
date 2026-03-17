import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple

from toolbox.models.CDNet.code.config import device


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def build_rope(seq_len, dim, device=None):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    dim_t = torch.arange(dim//2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000**(dim_t/(dim//2)))
    freqs = position * inv_freq.unsqueeze(0)

    sin = torch.sin(freqs)
    cos = torch.cos(freqs)

    sin = sin.repeat(1, 2)
    cos = cos.repeat(1, 2)

    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    return sin,cos

class MALAAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()


    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        seq_len = H*W
        sin, cos = build_rope(seq_len, q.shape[-1], device=q.device)

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H*W)) ** 0.5) @ (v * (self.scale / (H*W)) ** 0.5)

        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)


class RSLAAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()

        self.dim = dim

        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.qkvo = nn.Conv2d(dim, dim * 4, 1)  # 4倍通道用于q,k,v,o

        self.long_range_conv = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)  # 长距离依赖

        self.structural_conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)  # 结构感知

        self.proj = nn.Conv2d(dim, dim, 1)

        self.scale = self.head_dim ** -0.5

        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor):
        '''

        x: (b c h w)

        '''

        B, C, H, W = x.shape

        qkvo = self.qkvo(x)  # (b 4*c h w)

        qkv = qkvo[:, :3 * self.dim, :, :]  # (b 3*c h w)

        o = qkvo[:, 3 * self.dim:, :, :]  # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        seq_len = H * W

        sin, cos = build_rope(seq_len, q.shape[-1], device=q.device)
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H * W)) ** 0.5) @ (v * (self.scale / (H * W)) ** 0.5)

        res = q @ kv * (1 + 1 / (z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)

        long_range_feature = self.long_range_conv(x)
        structural_feature = self.structural_conv(x)
        multi_scale_feature = (long_range_feature + structural_feature) * 0.5

        # res = res + self.long_range_weight * long_range_feature + self.structural_weight * structural_feature
        res = res + long_range_feature + structural_feature + multi_scale_feature

        return self.proj(res * o)
if __name__ == "__main__":
    x = torch.randn(4,64,32,32)
    MALA = RSLAAttention(64, 8)
    out = MALA(x)
    print(out.shape)

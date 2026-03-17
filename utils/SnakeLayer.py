import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from toolbox.models.TestNet.models.utils.DSConv import DSConv_pro
import torch.nn.functional as F
import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F


def generate_serpentine_mask(in_features, out_features):

    mask = torch.zeros(out_features, in_features)
    k = max(1, in_features // 4)

    for i in range(out_features):
        start = (i * k) % in_features
        if (i // (in_features // k)) % 2 == 1:
            start = in_features - start - k
            start = max(0, start)
        end = min(start + k, in_features)
        mask[i, start:end] = 1.0
    return mask


class SerpentineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        mask = generate_serpentine_mask(in_features, out_features)
        self.register_buffer('serpentine_mask', mask)

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):

        masked_weight = self.weight * self.serpentine_mask

        return F.linear(x, masked_weight, self.bias)




import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.models.SAM2_UNet.utils.MALA import MALAAttention,RSLAAttention






class FuseBlock(nn.Module):
    def __init__(self, inchannels):
        super(FuseBlock, self).__init__()
        self.mala = MALAAttention(inchannels, 8)
        self.rsla = RSLAAttention(inchannels, 8)
        self.conv = nn.Conv2d(2 * inchannels, inchannels, 1)
    def forward(self, rgb, ndsm):

        out = self.conv(torch.cat([rgb, ndsm], dim=1))
        # out = self.mala(out) + rgb
        out = self.rsla(out) + rgb
        return out
if __name__ == "__main__":
    x = torch.randn(1,64,64,64)
    y = torch.randn(1,64,64,64)
    fuse = FuseBlock(64)
    out = fuse(x,y)
    print(out.shape)
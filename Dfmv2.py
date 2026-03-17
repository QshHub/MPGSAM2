# from toolbox.backbone.DFormer.DFormer import DFormer_Tiny
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models

from toolbox.backbone.DFormer.DFormer import DFormer_Tiny

class TransBasicConv2d(nn.Module):
    def   __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Dfmv2(nn.Module):
    def __init__(self):
        super(Dfmv2, self).__init__()
        self.Dfmv2 = DFormer_Tiny(pretrained=True)
        self.rdconv1 = nn.Conv2d(32,64,1)
        self.rdconv2 = nn.Conv2d(64, 64,1)
        self.rdconv3 = nn.Conv2d(128, 64,1)
        self.rdconv4 = nn.Conv2d(256, 64,1)

        self.transconv1 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv2 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv3 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                             padding=0, dilation=1, bias=False)

        self.conv1_1 = nn.Conv2d(128, 64, 1)
        self.conv2_1 = nn.Conv2d(128, 64, 1)
        self.conv3_1 = nn.Conv2d(128, 64, 1)

        self.conv_out_1 = nn.Conv2d(64, 64, 1)
        self.conv_out_2 = nn.Conv2d(64, 64, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.convout = nn.Conv2d(64, 6, 1)


    def forward(self, ndsm):
        outs = []


        x1, x2, x3, x4 = self.Dfmv2(ndsm, None)
        x1, x2, x3, x4 = self.rdconv1(x1), self.rdconv2(x2), self.rdconv3(x3), self.rdconv4(x4)
        outs.append(x1)
        outs.append(x2)
        outs.append(x3)
        outs.append(x4)

        out = self.conv1_1(torch.cat([x3, self.transconv1(x4)], dim=1))
        out = self.conv2_1(torch.cat([x2, self.transconv2(out)], dim=1))
        out = self.conv3_1(torch.cat([x1, self.transconv3(out)], dim=1))
        out = self.up2(self.conv_out_1(out))
        out1 = self.up2(self.conv_out_2(out))

        out2 = self.up2(self.convout(out))



        return outs,out1,out2


if __name__ == "__main__":
    with torch.no_grad():
        net = Dfmv2().cuda()
        x = torch.randn(4, 3, 256, 256).cuda()
        outs, out, out2 = net(x)
        for x in out:
            print(x.shape)
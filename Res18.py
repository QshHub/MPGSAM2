import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models

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

class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()

        self.RE_conv1 = models.resnet18(pretrained=True).conv1
        self.DE_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.DE_conv1.weight.data = torch.unsqueeze(
            torch.mean(models.resnet18(pretrained=True).conv1.weight.data, dim=1),
            dim=1)
        self.DE_bn1 = models.resnet18(pretrained=True).bn1
        self.DE_relu = models.resnet18(pretrained=True).relu
        self.DE_maxpool = models.resnet18(pretrained=True).maxpool
        self.DE_layer1 = models.resnet18(pretrained=True).layer1
        self.DE_layer2 = models.resnet18(pretrained=True).layer2
        self.DE_layer3 = models.resnet18(pretrained=True).layer3
        self.DE_layer4 = models.resnet18(pretrained=True).layer4

        self.rdconv1 = nn.Conv2d(64,64,1)
        self.rdconv2 = nn.Conv2d(128, 64, 1)
        self.rdconv3 = nn.Conv2d(256, 64, 1)
        self.rdconv4 = nn.Conv2d(512, 64, 1)

        self.transconv1 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv2 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv3 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv4 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
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

        ndsm = self.RE_conv1(ndsm)
        # ndsm = self.DE_conv1(ndsm)
        ndsm = self.DE_bn1(ndsm)
        ndsm = self.DE_relu(ndsm)

        ndsm = self.DE_maxpool(ndsm)
        ndsm = self.DE_layer1(ndsm)
        d1 = ndsm
        d1 = self.rdconv1(d1)
        outs.append(d1)

        ndsm = self.DE_layer2(ndsm)
        d2 = ndsm
        d2 = self.rdconv2(d2)
        outs.append(d2)

        ndsm = self.DE_layer3(ndsm)
        d3 = ndsm
        d3 = self.rdconv3(d3)
        outs.append(d3)

        ndsm = self.DE_layer4(ndsm)
        d4 = ndsm
        d4 = self.rdconv4(d4)
        outs.append(d4)

        out = self.conv1_1(torch.cat([d3, self.transconv1(d4)], dim=1))
        out = self.conv2_1(torch.cat([d2, self.transconv2(out)], dim=1))
        out = self.conv3_1(torch.cat([d1, self.transconv3(out)], dim=1))

        out = self.up2(self.conv_out_1(out))
        out2 = self.up2(self.convout(out))
        out = self.up2(self.conv_out_2(out))

        return outs,out,out2
if __name__ == "__main__":
    with torch.no_grad():
        net = Res18().cuda()
        x = torch.randn(1, 1, 256, 256).cuda()
        outs,out = net(x)
        print(out.shape)
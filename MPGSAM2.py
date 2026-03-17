import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.models.SAM2_UNet.utils.Clip import CLIPS
from toolbox.models.SAM2_UNet.sam2.build_sam import build_sam2
from toolbox.models.SAM2_UNet.utils.SnakeLayer import SerpentineLinear
from toolbox.models.SAM2_UNet.utils.ACMI import FuseBlock
from toolbox.models.SAM2_UNet.utils.Selector import PointPromptEvaluator
from toolbox.models.SAM2_UNet.utils.Selector import BoxPromptEvaluator
from toolbox.models.SAM2_UNet.utils.BoundaryPM import TransBoundary, BoundaryOptimizer
from toolbox.models.SAM2_UNet.utils.PBME import PointGuidedRefiner, BoxGuidedRefiner


class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super(PermuteLayer, self).__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=8, dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        result = self.linear(x)
        lora_result = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_result

class Adapter(nn.Module):
    def __init__(self, blk, lora_rank=4, lora_alpha=8, lora_dropout=0.1) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        in_channels = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            SerpentineLinear(in_channels, 32),
            nn.GELU(),
            nn.Linear(32, in_channels),
            nn.GELU()
        )
        self.lora_qkv = LoRALinear(
            blk.attn.qkv,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )

    def forward(self, x):
        original_qkv = self.block.attn.qkv
        self.block.attn.qkv = self.lora_qkv
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        self.block.attn.qkv = original_qkv
        return net

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

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


class CosineSimilarityFusion(nn.Module):
    def __init__(self, img_dim=64, text_dim=512, fusion_dim=64):
        super(CosineSimilarityFusion, self).__init__()

        self.img_projector = nn.Conv2d(fusion_dim, text_dim,1)
        self.conv = nn.Conv2d(text_dim, img_dim,1)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(img_dim + 5, img_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(img_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feat, text_feat):

        B, C, H, W = img_feat.shape
        K, D = text_feat.shape

        proj_img = self.img_projector(img_feat)

        img_norm = F.normalize(proj_img, p=2, dim=1)
        text_norm = F.normalize(text_feat, p=2, dim=1)

        similarity = torch.einsum('bchw, kc -> bkhw', img_norm, text_norm)

        weights = torch.softmax(similarity, dim=1)
        weights = weights.unsqueeze(2)

        img_norm_reshaped = img_norm.unsqueeze(1)
        weighted_img = weights * img_norm_reshaped
        weighted_img = weighted_img.sum(dim=1)

        proj_img = self.conv(weighted_img)
        output = img_feat + proj_img

        return output

class total_Net(nn.Module):
    def __init__(self, checkpoint_path='/media/xyx/shuju/xj_pytorch_segementation_Remote_Sensing/toolbox/models/SAM2_UNet/Sam_prepth2.0/sam2_hiera_large.pt') -> None:
        super(total_Net, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.CLIPMODEL = CLIPS()
        self.CLIPMODEL.build_subtype_features()

        self.i_t_align1 = CosineSimilarityFusion()
        self.i_t_align2 = CosineSimilarityFusion()
        self.i_t_align3 = CosineSimilarityFusion()
        self.i_t_align4 = CosineSimilarityFusion()

        self.evaluator_p = PointPromptEvaluator()
        self.evaluator_b = BoxPromptEvaluator()

        self.prompt_RF = PointGuidedRefiner()
        self.box_RF = BoxGuidedRefiner()

        self.transboundary = TransBoundary().cuda()
        self.boundary_RF = BoundaryOptimizer(feat_channels=64, prompt_dim=128)

        self.rdconv1 = nn.Conv2d(144, 64, 1)
        self.rdconv2 = nn.Conv2d(288, 64, 1)
        self.rdconv3 = nn.Conv2d(576, 64, 1)
        self.rdconv4 = nn.Conv2d(1152, 64, 1)

        self.transconv1 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.transconv2 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv3 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.transconv4 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)

        self.fusion1 = FuseBlock(64)
        self.fusion2 = FuseBlock(64)
        self.fusion3 = FuseBlock(64)
        self.fusion4 = FuseBlock(64)

        self.conv_p_b = nn.Conv2d(64 * 2, 64, 1)
        self.conv_x_pb = nn.Conv2d(64 * 2, 64, 1)

        self.conv1_b = nn.Conv2d(128, 64, 1)
        self.conv2_b = nn.Conv2d(128, 64, 1)
        self.conv3_b = nn.Conv2d(128, 64, 1)

        self.conv1_1 = nn.Conv2d(128, 64, 1)
        self.conv2_1 = nn.Conv2d(128, 64, 1)
        self.conv3_1 = nn.Conv2d(128, 64, 1)
        self.convbound = nn.Conv2d(128, 64, 1)


        self.conv_out_1 = nn.Conv2d(64, 32, 1)
        self.conv_out_2 = nn.Conv2d(32, 6, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x, p_l1, p_l2, b_l1, b_l2, outs1, outs2):
        x1, x2, x3, x4 = self.encoder(x)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        x1, x2, x3, x4 = self.rdconv1(x1), self.rdconv2(x2), self.rdconv3(x3), self.rdconv4(x4)
        out_S = x4
        # print(x1.shape, x2.shape, x3.shape, x4.shape)


        accp1, accp2 = self.evaluator_p((p_l1[0], p_l1[1]), (p_l2[0], p_l2[1]), out_S)

        scoreb1, scoreb2 = self.evaluator_b(out_S, b_l1[0], b_l2[0])
        accb1 = torch.mean(scoreb1)
        accb2 = torch.mean(scoreb2)
        acc1 = accp1 + accb1
        acc2 = accp2 + accb2

        nlist = outs1 if acc1 > acc2 else outs2

        points = p_l1[0] if acc1 > acc2 else p_l2[0]
        boxes = b_l1[0] if acc1 > acc2 else b_l2[0]
        mask = b_l1[2] if acc1 > acc2 else b_l2[2]

        x1, x2, x3, x4 = (self.fusion1(x1, nlist[0]), self.fusion1(x2, nlist[1])
                              , self.fusion2(x3, nlist[2]), self.fusion2(x4, nlist[3]))

        out_bound = self.conv1_1(torch.cat([x3, self.transconv1(x4)], dim=1))
        out_bound = self.conv2_1(torch.cat([x2, self.transconv2(out_bound)], dim=1))
        out_bound = self.conv3_1(torch.cat([x1, self.transconv3(out_bound)], dim=1))
        bound_p = self.transboundary(out_bound)
        out_bound = self.boundary_RF(out_bound, bound_p)


        x1 = self.i_t_align1(x1, self.CLIPMODEL.text_subtype)
        x2 = self.i_t_align2(x2, self.CLIPMODEL.text_subtype)
        x3 = self.i_t_align3(x3, self.CLIPMODEL.text_subtype)
        x4 = self.i_t_align4(x4, self.CLIPMODEL.text_subtype)

        out_p = self.prompt_RF(x4, points)
        out_b = self.box_RF(x4, boxes, mask)

        out = self.conv_p_b(torch.cat([out_p, out_b], dim=1))
        out = self.conv_x_pb(torch.cat([x4, out], dim=1))

        out = self.conv1_1(torch.cat([x3, self.transconv1(out)], dim=1))
        out = self.conv2_1(torch.cat([x2, self.transconv2(out)], dim=1))

        x1_bound = self.convbound(torch.cat([x1, out_bound], dim=1))

        out = self.conv3_1(torch.cat([x1_bound, self.transconv3(out)], dim=1))


        out = self.up2(self.conv_out_1(out))
        out = self.up2(self.conv_out_2(out))
        return out,out_S


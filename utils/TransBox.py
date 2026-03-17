import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureToBoxPrompt(nn.Module):

    def __init__(self, in_channels=64, num_classes=1):
        super(FeatureToBoxPrompt, self).__init__()

        self.center_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )

    def forward(self, x):
        with torch.no_grad():
            center_logit = self.center_head(x)
            center_map = torch.sigmoid(center_logit)

            size_map = self.size_head(x)
            offset_map = self.offset_head(x)

        return center_map, size_map, offset_map

    def get_box_prompts(self, x, K=10, score_threshold=0.3):
        """
        从特征图直接生成边界框提示 (Prediction Phase)。

        参数:
            x: 输入特征图 (B, 64, 256, 256)
            K: 每张图取前 K 个最高分的物体
            score_threshold: 过滤低分框的阈值

        返回:
            boxes: (B, K, 4) 格式为 [x1, y1, x2, y2]
            scores: (B, K)
            valid_mask: (B, K) 标记哪些box是有效的(超过阈值)
        """
        B, _, H, W = x.shape
        center_map, size_map, offset_map = self.forward(x)

        # 1. 寻找热力图中的峰值 (简单版 NMS: Max Pooling)
        # CenterNet 标准做法：如果在 3x3 区域内不是最大值，则抑制该点
        hmax = F.max_pool2d(center_map, kernel_size=3, padding=1, stride=1)
        keep = (hmax == center_map).float()
        center_map = center_map * keep

        # 2. Flatten 并提取 Top-K
        # view: (B, C, H, W) -> (B, H*W) 假设单类别
        center_map_flat = center_map.view(B, -1)

        topk_scores, topk_inds = torch.topk(center_map_flat, K)

        # 将 1D 索引转换回 2D 坐标 (y, x)
        topk_ys = torch.div(topk_inds, W, rounding_mode='floor').float()
        topk_xs = (topk_inds % W).float()

        # 3. 获取对应的 Offset 和 Size
        # 需要从 map 中根据 index 取值。
        # size_map: (B, 2, H, W) -> (B, 2, H*W)
        size_flat = size_map.view(B, 2, -1)
        offset_flat = offset_map.view(B, 2, -1)

        # gather 索引扩展: (B, 2, K)
        inds_expanded = topk_inds.unsqueeze(1).expand(B, 2, K)

        # 提取对应的 size (w, h) 和 offset (x_off, y_off)
        dec_size = torch.gather(size_flat, 2, inds_expanded)  # (B, 2, K)
        dec_offset = torch.gather(offset_flat, 2, inds_expanded)  # (B, 2, K)

        # 4. 计算边界框坐标

        # topk_xs 是网格坐标，加上预测的 offset
        center_xs = topk_xs + dec_offset[:, 0, :]
        center_ys = topk_ys + dec_offset[:, 1, :]

        # 宽和高
        ws = dec_size[:, 0, :]
        hs = dec_size[:, 1, :]

        # 计算左上角 (x1, y1) 和 右下角 (x2, y2)
        x1 = center_xs - ws / 2
        y1 = center_ys - hs / 2
        x2 = center_xs + ws / 2
        y2 = center_ys + hs / 2

        # 组合成 boxes: (B, K, 4)
        boxes = torch.stack([x1, y1, x2, y2], dim=2)

        # 生成有效掩码 (根据阈值)
        valid_mask = topk_scores > score_threshold

        return boxes, topk_scores, valid_mask


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: (Batch=4, Channel=64, H=256, W=256)
    dummy_input = torch.randn(8, 64, 256, 256)

    # 实例化模型
    model = FeatureToBoxPrompt(in_channels=64, num_classes=1)

    # 前向传播并生成 Box Prompts
    # boxes, scores, valid_mask = model.get_box_prompts(dummy_input, K=5)
    b = model.get_box_prompts(dummy_input, K=5)
    print("Input shape:", dummy_input.shape)
    print("Output Maps shape (Center):", model(dummy_input)[0].shape)
    print("Generated Boxes shape (B, K, 4):", b[0].shape, b[1].shape, b[2].shape)

    # print("Sample Box (x1, y1, x2, y2):", boxes[0, 0])
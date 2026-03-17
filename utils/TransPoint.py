import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureToPointPrompt(nn.Module):
    """
    将CNN特征转换为点提示的模块，用于语义分割的prompt encoder
    输入: [batch, 64, 256, 256] (CNN提取的特征)
    输出:
        points: [batch, num_points, 2]  # 归一化坐标 (x, y) in [0,1]
        labels: [batch, num_points]     # 类别标签 (0-4, 5个类别)

    参数:
        num_classes: 分割类别数 (默认5)
        num_points_per_class: 每个类别选多少个点 (默认10)
        threshold: 热力图阈值 (默认0.5, 值越大点越少)
    """

    def __init__(self, num_classes=5, num_points_per_class=10, threshold=0.5):
        super().__init__()
        # 1x1卷积将64通道映射到类别数，输出热力图
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.num_classes = num_classes
        self.num_points_per_class = num_points_per_class
        self.threshold = threshold

    def forward(self, x):
        # x: [batch, 64, 256, 256]
        batch_size = x.size(0)

        # 生成类别热力图 [batch, num_classes, 256, 256]
        heatmaps = torch.sigmoid(self.conv(x))  # 激活为概率 [0,1]

        all_points = []  # 存储所有点坐标 (x,y)
        all_labels = []  # 存储所有标签 (类别ID)

        for b in range(batch_size):
            batch_points = []
            batch_labels = []

            for cls in range(self.num_classes):
                # 当前类别的热力图 [256, 256]
                heatmap = heatmaps[b, cls]

                # 找出所有 > threshold 的点 (坐标索引)
                mask = (heatmap > self.threshold).float()
                coords = torch.nonzero(mask, as_tuple=True)  # (y, x) 索引

                # 如果点太多，按热力图值排序取 top N
                if coords[0].size(0) > self.num_points_per_class:
                    # 按值从高到低排序
                    _, idx = torch.topk(heatmap.flatten(), self.num_points_per_class)
                    y = idx // 256
                    x = idx % 256
                    coords = (y, x)

                # 如果点太少，用中心点填充（避免空点）
                elif coords[0].size(0) == 0:
                    coords = (torch.tensor([128], device=x.device), torch.tensor([128], device=x.device))

                # 确保点数固定 (填充或截断)
                if coords[0].size(0) < self.num_points_per_class:
                    # 用随机重复填充 (实际项目可优化为高值点)
                    pad_idx = torch.randint(0, coords[0].size(0), (self.num_points_per_class - coords[0].size(0),))
                    y_pad = coords[0][pad_idx]
                    x_pad = coords[1][pad_idx]
                    coords = (torch.cat([coords[0], y_pad]), torch.cat([coords[1], x_pad]))

                # 转成 [num_points, 2] 坐标 (x,y)，并归一化到 [0,1]
                points = torch.stack([coords[1], coords[0]], dim=1).float() / 255.0  # 256x256 -> [0,1]
                labels = torch.full((points.size(0),), cls, dtype=torch.long, device=x.device)

                batch_points.append(points)
                batch_labels.append(labels)

            # 合并当前batch的所有类别点
            batch_points = torch.cat(batch_points, dim=0)  # [num_classes * num_points, 2]
            batch_labels = torch.cat(batch_labels, dim=0)  # [num_classes * num_points]

            all_points.append(batch_points)
            all_labels.append(batch_labels)

        # 组合成 [batch, total_points, 2] 和 [batch, total_points]
        points = torch.stack(all_points, dim=0)  # [batch, total_points, 2]
        labels = torch.stack(all_labels, dim=0)  # [batch, total_points]

        return points, labels


# 使用示例 (假设你有一个特征图)
if __name__ == "__main__":
    # 模拟输入: batch=4, 64 channels, 256x256
    features = torch.randn(4, 64, 256, 256)

    # 初始化模块 (5个类别, 每类10个点)
    point_module = FeatureToPointPrompt(num_classes=5, num_points_per_class=10)

    # 转换为点提示
    points, labels = point_module(features)

    print("Points shape:", points.shape)  # 应该是 [4, 50, 2]
    print("Labels shape:", labels.shape)  # 应该是 [4, 50]
    print("Example points (first batch):", points[0, :3])  # 前3个点
    print("Example labels:", labels[0, :3])  # 对应类别
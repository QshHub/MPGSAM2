import torch
import torch.nn as nn

class PointGuidedRefiner(nn.Module):
    def __init__(self, in_channels=64, sigma=3.0):

        super().__init__()
        self.sigma = sigma

        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _generate_gaussian_heatmap(self, points, img_size):

        B, N, _ = points.shape
        H, W = img_size

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=points.device),
            torch.arange(W, device=points.device),
            indexing='ij'
        )
        grid_x = grid_x.view(1, 1, H, W)  # (1, 1, H, W)
        grid_y = grid_y.view(1, 1, H, W)  # (1, 1, H, W)

        px = points[:, :, 0].view(B, N, 1, 1)
        py = points[:, :, 1].view(B, N, 1, 1)

        dist_sq = (grid_x - px) ** 2 + (grid_y - py) ** 2
        heatmap = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        heatmap, _ = torch.max(heatmap, dim=1, keepdim=True)
        return heatmap

    def forward(self, x, points):

        B, C, H, W = x.shape

        mask = self._generate_gaussian_heatmap(points, (H, W))

        combined = torch.cat([x, mask], dim=1)
        refined_feat = self.refine_conv(combined)

        attn = self.gate(refined_feat)
        out = refined_feat * attn

        return out


class BoxGuidedRefiner(nn.Module):
    def __init__(self, in_channels=64):
        super(BoxGuidedRefiner, self).__init__()

        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def _generate_mask_from_boxes(self, boxes, valid_mask, h, w):

        device = boxes.device
        B, K, _ = boxes.shape

        y_grid = torch.arange(h, device=device).view(1, 1, h, 1).expand(B, 1, h, w)
        x_grid = torch.arange(w, device=device).view(1, 1, 1, w).expand(B, 1, h, w)

        x1 = boxes[:, :, 0].view(B, K, 1, 1)
        y1 = boxes[:, :, 1].view(B, K, 1, 1)
        x2 = boxes[:, :, 2].view(B, K, 1, 1)
        y2 = boxes[:, :, 3].view(B, K, 1, 1)

        # margin = 1.0
        mask = torch.sigmoid(x_grid - x1) * torch.sigmoid(x2 - x_grid) * \
               torch.sigmoid(y_grid - y1) * torch.sigmoid(y2 - y_grid)

        mask = mask * valid_mask.view(B, K, 1, 1).float()

        spatial_mask, _ = torch.max(mask, dim=1, keepdim=True)

        return spatial_mask

    def forward(self, x, boxes, valid_mask):

        B, C, H, W = x.shape

        box_mask = self._generate_mask_from_boxes(boxes, valid_mask, H, W)

        attended_x = x * box_mask

        concat_x = torch.cat([x, attended_x], dim=1)
        refined_features = self.refine_conv(concat_x)

        return refined_features


if __name__ == "__main__":

    feat = torch.randn(2, 64, 64, 64)

    pts = torch.tensor([
        [[10, 20], [50, 50], [100, 150]],
        [[200, 200], [30, 200], [128, 128]]
    ]).float()

    model = PointGuidedRefiner(in_channels=64, sigma=5.0)
    output = model(feat, pts)

    print(feat.shape)
    print(output.shape)  # (2, 64, 256, 256)
    # from TransBox import FeatureToBoxPrompt

    # B, C, H, W = 4, 64, 64, 64
    # features = torch.randn(B, C, H, W)

    # box_module = FeatureToBoxPrompt(in_channels=C)
    # boxes, scores, valid_mask = box_module.get_box_prompts(features, K=5)

    # refinement_module = BoxGuidedRefinement(in_channels=C)
    # refined_features = refinement_module(features, boxes, valid_mask)
    #
    # print(features.shape)
    # print(boxes.shape)
    # print(refined_features.shape)
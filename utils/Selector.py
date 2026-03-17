import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class PointPromptEvaluator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prompt1, prompt2, feature):

        points1, labels1 = prompt1
        points2, labels2 = prompt2

        x1 = (points1[..., 0] * 8).clamp(0, 7.999).long()
        y1 = (points1[..., 1] * 8).clamp(0, 7.999).long()

        x2 = (points2[..., 0] * 8).clamp(0, 7.999).long()
        y2 = (points2[..., 1] * 8).clamp(0, 7.999).long()

        feature1 = feature[:, :, y1, x1]
        feature2 = feature[:, :, y2, x2]

        norms1 = torch.norm(feature1, dim=1)
        norms2 = torch.norm(feature2, dim=1)

        score1 = norms1.mean()
        score2 = norms2.mean()

        return 0.1 * score1, 0.1 * score2



class BoxPromptEvaluator(nn.Module):


    def __init__(self, main_channels=64, roi_resolution=3):
        super(BoxPromptEvaluator, self).__init__()

        self.roi_resolution = roi_resolution

        self.evaluator = nn.Sequential(
            nn.Conv2d(main_channels, main_channels, kernel_size=roi_resolution, padding=0),  # 变成 1x1
            nn.Flatten(),
            nn.Linear(main_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, main_feat, boxes_a, boxes_b):

        B, C, H_main, W_main = main_feat.shape
        _, K, _ = boxes_a.shape

        spatial_scale = H_main / 256.0

        batch_inds = torch.arange(B, device=main_feat.device).repeat_interleave(K).float().unsqueeze(1)

        flat_boxes_a = boxes_a.view(-1, 4)
        flat_boxes_b = boxes_b.view(-1, 4)

        rois_a = torch.cat([batch_inds, flat_boxes_a], dim=1)
        rois_b = torch.cat([batch_inds, flat_boxes_b], dim=1)

        feats_a = ops.roi_align(main_feat, rois_a, output_size=(self.roi_resolution, self.roi_resolution),
                                spatial_scale=spatial_scale)
        feats_b = ops.roi_align(main_feat, rois_b, output_size=(self.roi_resolution, self.roi_resolution),
                                spatial_scale=spatial_scale)

        scores_a = self.evaluator(feats_a).view(B, K)  # (B, K)
        scores_b = self.evaluator(feats_b).view(B, K)  # (B, K)

        # select_b_mask = (scores_b > scores_a)
        # mask_expanded = select_b_mask.unsqueeze(-1).expand_as(boxes_a)
        # best_boxes = torch.where(mask_expanded, boxes_b, boxes_a)

        return scores_a, scores_b


if __name__ == "__main__":

    # batch = 4
    # total_points = 50
    # feature = torch.randn(batch, 64, 8, 8)

    # points1 = torch.rand(batch, total_points, 2)
    # labels1 = torch.randint(0, 5, (batch, total_points))
    #
    # points2 = torch.rand(batch, total_points, 2)
    # labels2 = torch.randint(0, 5, (batch, total_points))

    # evaluator = PointPromptEvaluator()

    # score1, score2 = evaluator(
    #     (points1, labels1),
    #     (points2, labels2),
    #     feature
    # )
    # print(score1)

    # if score1 > score2:
    #     print("score1")
    # else:
    #     print("score2")

    batch_size = 4
    K = 10

    main_feature = torch.randn(batch_size, 64, 8, 8)

    boxes_A = torch.tensor([[[10., 10., 50., 50.]] * K] * batch_size)

    boxes_B = torch.tensor([[[100., 100., 150., 150.]] * K] * batch_size)

    arbiter = BoxPromptEvaluator(main_channels=64, roi_resolution=3)

    sco1, sco2 = arbiter(main_feature, boxes_A, boxes_B)

    print(sco1)
    # print(sco2)


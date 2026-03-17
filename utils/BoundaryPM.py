import torch
import torch.nn as nn
import torch.nn.functional as F

class TransBoundary(nn.Module):

    def __init__(self, input_channels=64, output_dim=128):
        super(TransBoundary, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.threshold = 0.5

        self.inner_channels = [0, 1, 2, 3]
        self.cross_channels = [4, 5, 6, 7]

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        boundary_map = (x > self.threshold).float()

        boundary_points = []
        for batch_idx in range(x.size(0)):
            points = []
            for c in range(x.size(1)):
                coords = torch.nonzero(boundary_map[batch_idx, c, :, :], as_tuple=True)
                coords = torch.stack(coords, dim=1).float()
                coords = coords / 255.0
                points.append(coords)

            points = torch.cat(points, dim=0)

            unique_points = torch.unique(points, dim=0)

            if len(unique_points) < 3:
                boundary_points.append(torch.zeros(128))
            else:

                features = self.compute_geometric_features(unique_points, x[batch_idx])
                boundary_points.append(features)

        boundary_points = torch.stack(boundary_points, dim=0)

        boundary_points = boundary_points.view(boundary_points.size(0), -1).cuda()

        return boundary_points

    def compute_geometric_features(self, points, x_batch):

        sorted_points = points[points[:, 0].argsort()]
        sorted_points = sorted_points[sorted_points[:, 1].argsort()]

        features = []
        for i in range(len(sorted_points)):
            j = (i - 1) % len(sorted_points)
            k = (i + 1) % len(sorted_points)
            dist_ij = torch.norm(sorted_points[i] - sorted_points[j], dim=0)
            dist_jk = torch.norm(sorted_points[j] - sorted_points[k], dim=0)

            vec1 = sorted_points[i] - sorted_points[j]
            vec2 = sorted_points[k] - sorted_points[j]
            cos_theta = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
            theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

            edge_type_ij = self.get_edge_type(sorted_points[j], sorted_points[i], x_batch)
            edge_type_jk = self.get_edge_type(sorted_points[j], sorted_points[k], x_batch)

            features.append(dist_ij)
            features.append(dist_jk)
            features.append(theta)
            features.append(edge_type_ij)
            features.append(edge_type_jk)

        features = torch.tensor(features)
        if features.size(0) < 128:
            features = F.pad(features, (0, 128 - features.size(0)))
        elif features.size(0) > 128:
            features = features[:128]

        return features

    def get_edge_type(self, point1, point2, x_batch):

        x1, y1 = int(point1[0] * 255), int(point1[1] * 255)
        x2, y2 = int(point2[0] * 255), int(point2[1] * 255)

        inner_value = 0
        cross_value = 0

        for c in self.inner_channels:
            if x_batch[c, y1, x1] > self.threshold and x_batch[c, y2, x2] > self.threshold:
                inner_value += 1

        for c in self.cross_channels:
            if x_batch[c, y1, x1] > self.threshold and x_batch[c, y2, x2] > self.threshold:
                cross_value += 1

        return 0 if inner_value > cross_value else 1


class BoundaryOptimizer(nn.Module):

    def __init__(self, feat_channels=64, prompt_dim=128):
        super(BoundaryOptimizer, self).__init__()

        self.prompt_encoder = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(prompt_dim // 2, feat_channels * 2)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feat_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.refine_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, boundary_prompt):
        B, C, H, W = feat.shape

        params = self.prompt_encoder(boundary_prompt)
        params = params.view(B, 2, C, 1, 1)
        gamma = params[:, 0, :, :, :]
        beta = params[:, 1, :, :, :]

        modulated_feat = feat * (1 + gamma) + beta

        out = self.refine_conv(modulated_feat)
        optimized_feat = out

        return optimized_feat

if __name__ == "__main__":

    batch_size = 4
    raw_features = torch.randn(batch_size, 64, 64, 64).cuda()

    boundary_prompt = torch.randn(batch_size, 128).cuda()

    optimizer = BoundaryOptimizer(feat_channels=64, prompt_dim=128).cuda()

    refined_features = optimizer(raw_features, boundary_prompt)

    print(raw_features.shape)
    print(refined_features.shape)

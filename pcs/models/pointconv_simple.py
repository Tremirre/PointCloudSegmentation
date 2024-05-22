import torch
import torch.nn as nn
import torch.nn.functional as F

from pcs.utils.pointconv import (
    index_points,
    square_distance,
    sample_and_group,
    sample_and_group_all,
    compute_density,
)


class WeightNetHidden(nn.Module):
    def __init__(self, hidden_units: list[int]):
        super().__init__()
        self.hidden_units = hidden_units
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for in_ch, out_ch in zip(hidden_units[:-1], hidden_units[1:]):
            self.conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_ch))
        self.activation_fn = nn.ReLU()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_layers):
            xyz = conv_layer(xyz)
            xyz = self.bn_layers[i](xyz)
            xyz = self.activation_fn(xyz)
        return xyz


class WeightNet(nn.Module):
    def __init__(self, hidden_units: list[int], activation_fn: nn.Module | None = None):
        super().__init__()
        self.hidden_units = hidden_units
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self.conv_layers.append(
                nn.Conv2d(hidden_units[i], hidden_units[i + 1], kernel_size=1)
            )
        self.activation_fn = activation_fn or nn.ReLU()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            xyz = self.activation_fn(conv_layer(xyz))
        return xyz


class NonlinearTransform(nn.Module):
    def __init__(self, mlp: list[int]):
        super().__init__()
        self.mlp = mlp
        self.conv_layers = nn.ModuleList()
        for i in range(len(mlp) - 1):
            self.conv_layers.append(nn.Conv2d(mlp[i], mlp[i + 1], kernel_size=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_in: torch.Tensor) -> torch.Tensor:
        net = data_in
        for conv_layer in self.conv_layers[:-1]:
            net = conv_layer(net)
            net = nn.ReLU()(net)
        net = self.conv_layers[-1](net)
        net = self.sigmoid(net)
        return net


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)

        return density_scale


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))

        return weights


class FeatureEncoder(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(FeatureEncoder, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, points):
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = (
                sample_and_group_all(xyz, points, inverse_density.view(B, N, 1))
            )
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = (
                sample_and_group(
                    self.npoint,
                    self.nsample,
                    xyz,
                    points,
                    inverse_density.view(B, N, 1),
                )
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(
            input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)
        ).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class FeatureFlatDecoder(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bandwidth):
        super(FeatureFlatDecoder, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.nsample = nsample
        self.bandwidth = bandwidth

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class Feature2DDecoder(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bandwidth):
        super(Feature2DDecoder, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.nsample = nsample
        self.bandwidth = bandwidth
        self.densitynet = DensityNet()
        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        xyz_density = compute_density(xyz1, self.bandwidth)
        inverse_density = 1.0 / xyz_density
        new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(
            xyz1, interpolated_points, inverse_density.view(B, N, 1)
        )
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(
            input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)
        ).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


XYZ_DIM = 3


class PointConvNet(nn.Module):
    def __init__(self, features: int = 4, classes: int = 8):
        super(PointConvNet, self).__init__()
        self.sa1 = FeatureEncoder(
            npoint=1024,
            nsample=32,
            in_channel=XYZ_DIM + features,
            mlp=[32, 64],
            bandwidth=0.1,
            group_all=False,
        )
        self.sa2 = FeatureEncoder(
            npoint=256,
            nsample=32,
            in_channel=XYZ_DIM + 64,
            mlp=[64, 128],
            bandwidth=0.2,
            group_all=False,
        )
        self.sa3 = FeatureEncoder(
            npoint=64,
            nsample=32,
            in_channel=XYZ_DIM + 128,
            mlp=[128, 256],
            bandwidth=0.4,
            group_all=False,
        )
        # self.sa4 = FeatureEncoder(
        #     npoint=64,
        #     nsample=32,
        #     in_channel=XYZ_DIM + 256,
        #     mlp=[256, 256, 512],
        #     bandwidth=0.8,
        #     group_all=False,
        # )
        # self.fp4 = FeatureFlatDecoder(
        #     nsample=32, in_channel=768, mlp=[256, 256], bandwidth=0.8
        # )
        self.fp3 = FeatureFlatDecoder(
            nsample=32, in_channel=384, mlp=[256, 256], bandwidth=0.4
        )
        self.fp2 = FeatureFlatDecoder(
            nsample=32, in_channel=320, mlp=[256, 256], bandwidth=0.2
        )
        self.fp1 = FeatureFlatDecoder(
            nsample=32, in_channel=256, mlp=[128, 128], bandwidth=0.1
        )
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, classes, 1)

    def forward(self, pointcloud):
        # pointcloud: BxCxN
        xyz = pointcloud[:, :XYZ_DIM, :]
        feat = pointcloud[:, XYZ_DIM:, :]
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        out = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        out = F.log_softmax(self.conv2(out), dim=1)
        return out


if __name__ == "__main__":
    model = PointConvNet()
    xyz = torch.rand(32, 3, 1024)
    feat = torch.rand(32, 4, 1024)
    out = model(xyz, feat)
    print(out.shape)

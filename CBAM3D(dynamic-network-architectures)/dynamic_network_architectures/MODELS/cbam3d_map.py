import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_3d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        # self.latest_channel_scale_vector = None # Optional: if you want to store channel attention

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool3d(x, kernel_size=x.size()[2:])
            elif pool_type == 'max':
                pool = F.max_pool3d(x, kernel_size=x.size()[2:])
            elif pool_type == 'lp':
                pool = F.lp_pool3d(x, norm_type=2, kernel_size=x.size()[2:])
            elif pool_type == 'lse':
                pool = logsumexp_3d(x)
            channel_att_raw = self.mlp(pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        
        scale = torch.sigmoid(channel_att_sum).view(x.size(0), x.size(1), 1, 1, 1)
        # self.latest_channel_scale_vector = scale.detach()
        return x * scale.expand_as(x)


class ChannelPool3D(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        mean_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, mean_pool), dim=1)

class SpatialGate3D(nn.Module):
    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, relu=False)
        self.latest_scale = None

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        self.latest_scale = torch.sigmoid(x_out)
        return x * self.latest_scale 

class CBAM3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()
        self.latest_spatial_attention = None

    def forward(self, x):

        x_channel_attended = self.ChannelGate(x)

        x_final_out = x_channel_attended
        if not self.no_spatial:
            x_final_out = self.SpatialGate(x_channel_attended)
            
            self.latest_spatial_attention = self.SpatialGate.latest_scale
        return x_final_out
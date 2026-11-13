import math
import torch
from torch import nn
import torch.nn.functional as F


class LDConv(nn.Module):
    def __init__(self, c1, c2, num_points, stride=1):
        super().__init__()
        if not isinstance(num_points, int) or num_points < 1 or not isinstance(stride, int) or stride < 1:
            raise ValueError('num_points and stride must be positive integers')
        self.num_points, self.stride = num_points, stride
        self.offset = nn.Conv2d(c1, 2 * num_points, 3, stride, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.aggregate = nn.Sequential(
            nn.Conv2d(c1, c2, (num_points, 1), (num_points, 1), bias=False),
            nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        b, c, h, w = x.shape
        offset = self.offset(x)
        oh, ow = offset.shape[-2:]
        n = self.num_points
        columns = max(1, round(math.sqrt(n)))
        ids = torch.arange(n, device=x.device)
        iy, ix = (ids // columns).to(x.dtype), (ids % columns).to(x.dtype)
        yy, xx = torch.meshgrid(
            torch.arange(oh, device=x.device, dtype=x.dtype) * self.stride,
            torch.arange(ow, device=x.device, dtype=x.dtype) * self.stride, indexing='ij')
        py = (yy[None, :, :, None] + iy + offset[:, :n].permute(0, 2, 3, 1)).clamp(0, h - 1)
        px = (xx[None, :, :, None] + ix + offset[:, n:].permute(0, 2, 3, 1)).clamp(0, w - 1)
        grid = torch.stack((2 * px / max(w - 1, 1) - 1, 2 * py / max(h - 1, 1) - 1), -1)
        grid = grid.reshape(b, oh, ow * n, 2)
        samples = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        samples = samples.reshape(b, c, oh, ow, n).permute(0, 1, 2, 4, 3).reshape(b, c, oh * n, ow)
        return self.aggregate(samples)

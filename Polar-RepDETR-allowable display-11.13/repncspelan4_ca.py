import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        p = d * (k - 1) // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConvN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv3 = Conv(c1, c2, 3, act=False)
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.act = nn.SiLU()

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.act(self.reparam(x))
        return self.act(self.conv3(x) + self.conv1(x))

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError('Call eval() before fusing BatchNorm statistics')
        if hasattr(self, 'reparam'):
            return self
        def fuse(block):
            bn = block.bn
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            return block.conv.weight * scale[:, None, None, None], bn.bias - bn.running_mean * scale
        k3, b3 = fuse(self.conv3)
        k1, b1 = fuse(self.conv1)
        self.reparam = nn.Conv2d(k3.shape[1], k3.shape[0], 3, padding=1, bias=True).to(k3)
        self.reparam.weight.copy_(k3 + F.pad(k1, (1, 1, 1, 1)))
        self.reparam.bias.copy_(b3 + b1)
        self.reparam.eval()
        del self.conv3, self.conv1
        return self


class RepNBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cv1 = RepConvN(channels, channels)
        self.cv2 = Conv(channels, channels, 3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    def __init__(self, c1, c2, repeats=1):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(c1, hidden, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.blocks = nn.Sequential(*(RepNBottleneck(hidden) for _ in range(repeats)))

    def forward(self, x):
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), 1))


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError('Positive channels and reduction are required')
        hidden = max(8, channels // reduction)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden), nn.Hardswish())
        self.height = nn.Conv2d(hidden, channels, 1)
        self.width = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        zh = x.mean(3, keepdim=True)
        zw = x.mean(2, keepdim=True).transpose(2, 3)
        fh, fw = self.reduce(torch.cat((zh, zw), 2)).split((h, w), 2)
        return x * self.height(fh).sigmoid() * self.width(fw.transpose(2, 3)).sigmoid()


class RepNCSPELAN4_CA(nn.Module):
    def __init__(self, c1, c2, c3, c4, repeats=1):
        super().__init__()
        if min(c1, c2, c3, c4, repeats) <= 0 or c3 % 2 or c4 % 2:
            raise ValueError('Positive channels/repeats and even c3/c4 are required')
        self.cv1 = Conv(c1, c3, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, repeats), Conv(c4, c4, 3))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, repeats), Conv(c4, c4, 3))
        self.ca = CoordinateAttention(c3 + 2 * c4)
        self.cv4 = Conv(c3 + 2 * c4, c2, 1)

    def forward(self, x):
        u1, u2 = self.cv1(x).chunk(2, 1)
        u3 = self.cv2(u2)
        u4 = self.cv3(u3)
        return self.cv4(self.ca(torch.cat((u1, u2, u3, u4), 1)))

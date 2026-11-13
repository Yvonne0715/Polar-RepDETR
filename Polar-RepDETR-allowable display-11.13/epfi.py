import torch
from torch import nn
import torch.nn.functional as F


class PolaLinearAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        if not isinstance(channels, int) or not isinstance(num_heads, int) or min(channels, num_heads) < 1 or channels % num_heads or (channels // num_heads) % 2:
            raise ValueError('channels/num_heads must be an even integer')
        self.channels, self.heads = channels, num_heads
        self.head_dim = channels // num_heads
        self.qg = nn.Linear(channels, 2 * channels, bias=False)
        self.kv = nn.Linear(channels, 2 * channels, bias=False)
        self.scale = nn.Parameter(torch.zeros(1, 1, channels))
        self.power = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.position = nn.Parameter(torch.zeros(1, 400, channels))
        self.local = nn.Conv2d(self.head_dim, self.head_dim, 5, padding=2, groups=self.head_dim)
        self.project = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        tokens = x.flatten(2).transpose(1, 2)
        q, gate = self.qg(tokens).chunk(2, -1)
        k, v = self.kv(tokens).chunk(2, -1)
        pos = self.position.transpose(1, 2).reshape(1, c, 20, 20)
        pos = F.interpolate(pos, size=(h, w), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        scale = F.softplus(self.scale).clamp_min(1e-6)
        q, k = (q + pos) / scale, (k + pos) / scale
        q, k, v = [t.reshape(b, n, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        power = 1 + 4 * self.power.sigmoid()
        qp, qn = F.relu(q).pow(power), F.relu(-q).pow(power)
        kp, kn = F.relu(k).pow(power), F.relu(-k).pow(power)
        qs, qo, ks = torch.cat((qp, qn), -1), torch.cat((qn, qp), -1), torch.cat((kp, kn), -1)
        vs, vo = v.chunk(2, -1)
        mean_k = ks.mean(-2, keepdim=True).transpose(-2, -1)
        same = (qs @ (ks.transpose(-2, -1) @ vs / n)) / (qs @ mean_k + 1e-6)
        opposite = (qo @ (ks.transpose(-2, -1) @ vo / n)) / (qo @ mean_k + 1e-6)
        y = torch.cat((same, opposite), -1).transpose(1, 2).reshape(b, n, c)
        local = v.permute(0, 1, 3, 2).reshape(b * self.heads, self.head_dim, h, w)
        local = self.local(local).reshape(b, self.heads, self.head_dim, n)
        local = local.permute(0, 3, 1, 2).reshape(b, n, c)
        y = self.project((y + local) * gate)
        return y.transpose(1, 2).reshape(b, c, h, w)


class EDFFN(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        if channels < 1 or expansion <= 0 or int(channels * expansion) < 1:
            raise ValueError('Positive channels and expansion are required')
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, 2 * hidden, 1, bias=False)
        self.depthwise = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)
        self.frequency = nn.Parameter(torch.ones(channels, 1, 1, 8, 5))

    def forward(self, x):
        dtype = x.dtype
        a, b = self.depthwise(self.project_in(x)).chunk(2, 1)
        x = self.project_out(F.gelu(a) * b)
        batch, channels, h, w = x.shape
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect' if ph < h and pw < w else 'replicate')
        hp, wp = x.shape[-2:]
        patches = x.reshape(batch, channels, hp // 8, 8, wp // 8, 8).permute(0, 1, 2, 4, 3, 5)
        patches = patches.to(torch.float64 if dtype == torch.float64 else torch.float32)
        spectrum = torch.fft.rfft2(patches)
        patches = torch.fft.irfft2(spectrum * self.frequency, s=(8, 8))
        x = patches.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, hp, wp)
        return x[:, :, :h, :w].to(dtype)


class EPFI(nn.Module):
    def __init__(self, channels=256, expansion=2, num_heads=8):
        super().__init__()
        self.attention = PolaLinearAttention(channels, num_heads)
        self.ffn = EDFFN(channels, expansion)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    @staticmethod
    def norm(x, layer):
        return layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        x = self.norm(x + self.attention(x), self.norm1)
        return self.norm(x + self.ffn(x), self.norm2)

import math
import torch
from torch import nn

class WiseInnerMPDIoULoss(nn.Module):

    def __init__(self, rho=0.7, alpha=1.7, delta=2.7, momentum=0.01, eps=1e-12):
        super().__init__()
        if not all(math.isfinite(v) for v in (rho, alpha, delta, momentum, eps)) or rho <= 0 or alpha <= 1 or delta <= 0 or not 0 <= momentum <= 1 or eps <= 0:
            raise ValueError('Require rho>0, alpha>1, delta>0, momentum in [0,1], eps>0')
        self.rho, self.alpha, self.delta = (rho, alpha, delta)
        self.momentum, self.eps = (momentum, eps)
        self.register_buffer('running_mean', torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def corners(boxes, scale=1.0):
        half = boxes[..., 2:] * (scale / 2)
        return (boxes[..., :2] - half, boxes[..., :2] + half)

    def forward(self, prediction, target, domain_width=1.0, domain_height=1.0, reduction='mean', update_mean=None):
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 4:
            raise ValueError('Expected matching [N,4] xywh tensors')
        if domain_width <= 0 or domain_height <= 0:
            raise ValueError('Coordinate-domain dimensions must be positive')
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Unknown reduction')
        if prediction.shape[0] == 0:
            return prediction.sum(-1) if reduction == 'none' else prediction.sum() * 0
        dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
        p, g = (prediction.to(dtype), target.to(dtype))
        p1, p2 = self.corners(p)
        g1, g2 = self.corners(g)
        enclosing_wh = torch.maximum(p2, g2) - torch.minimum(p1, g1)
        distance = (p[:, :2] - g[:, :2]).square().sum(-1)
        rw = torch.exp(distance / enclosing_wh.square().sum(-1).clamp_min(self.eps))
        ap1, ap2 = self.corners(p, self.rho)
        ag1, ag2 = self.corners(g, self.rho)
        intersection = (torch.minimum(ap2, ag2) - torch.maximum(ap1, ag1)).clamp_min(0).prod(-1)
        union = (p[:, 2:].prod(-1) + g[:, 2:].prod(-1)) * self.rho ** 2 - intersection
        inner = 1 - intersection / union.clamp_min(self.eps)
        corner_penalty = ((p1 - g1).square().sum(-1) + (p2 - g2).square().sum(-1)) / (domain_width ** 2 + domain_height ** 2)
        if update_mean is None:
            update_mean = self.training
        with torch.no_grad():
            if update_mean:
                self.running_mean.mul_(1 - self.momentum).add_(inner.detach().mean().to(self.running_mean), alpha=self.momentum)
            reference = self.running_mean.detach().clone().to(inner).clamp_min(self.eps)
        beta = inner.detach() / reference
        focus = beta / self.delta * torch.exp((self.delta - beta) * math.log(self.alpha))
        losses = focus * rw * (inner + corner_penalty)
        if reduction == 'none':
            return losses
        return losses.sum() if reduction == 'sum' else losses.mean()

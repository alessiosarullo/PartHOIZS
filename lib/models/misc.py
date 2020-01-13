import numpy as np
import torch
from torch.nn import functional

from config import cfg


class ResizeLongestEdge:
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def apply(self, *, img_w, img_h, boxes):
        h, w = img_h, img_w
        size = self.max_size

        newh = torch.full_like(h, fill_value=size, device=boxes.device)
        neww = torch.full_like(w, fill_value=size, device=boxes.device)
        newh[h < w] = size * (h / w)[h < w]
        neww[w < h] = size * (w / h)[w < h]
        neww = torch.floor(neww + 0.5)
        newh = torch.floor(newh + 0.5)

        boxes[..., [0, 2]] *= (neww / w).view(-1, 1, 1)
        boxes[..., [1, 3]] *= (newh / h).view(-1, 1, 1)
        return neww.int(), newh.int(), boxes


def bce_loss(logits, labels, pos_weights=None, reduce=True):
    if cfg.fl_gamma != 0:  # Focal loss
        gamma = cfg.fl_gamma
        s = logits
        t = labels
        m = s.clamp(min=0)  # m = max(s, 0)
        x = (-s.abs()).exp()
        z = ((s >= 0) == t.byte()).float()
        loss_mat = (1 + x).pow(-gamma) * (m - s * t + x * (gamma * z).exp() * (1 + x).log())
        if reduce:
            loss = loss_mat.mean()
        else:
            loss = loss_mat
    else:  # standard BCE loss
        loss = functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weights, reduction='elementwise_mean' if reduce else 'none')
    if reduce and not cfg.meanc:
        loss *= logits.shape[1]
    return loss


def LIS(x, w=None, k=None, T=None):  # defaults are as in the paper
    if T is None:
        if w is None and k is None:
            w, k, T = 10, 12, 8.4
        else:
            assert w is not None and k is not None
            # This is basically what it is: a normalisation constant for when x=1.
            T = 1 + np.exp(k - w).item()
    assert w is not None and k is not None and T is not None
    return T * torch.sigmoid(w * x - k)

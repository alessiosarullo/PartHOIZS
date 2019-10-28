from typing import Dict

import torch
from torch import nn as nn


class AbstractModel(nn.Module):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})
        self.values_to_monitor = {}  # type: Dict[str, torch.Tensor]

    def forward(self, x, inference=True, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

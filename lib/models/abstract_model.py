from typing import Dict, Union

import numpy as np
import torch
from torch import nn as nn
from scipy.sparse import csr_matrix


class AbstractModel(nn.Module):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})
        self.extra_infos = {}  # type: Dict[str, torch.Tensor]

    def forward(self, x, inference=True, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class Prediction:
    def __init__(self, prediction_dict=None):
        self.part_state_scores = None  # type: Union[None, np.ndarray]

        self.obj_boxes = None  # type: Union[None, np.ndarray]

        self.ho_pairs = None  # type: Union[None, np.ndarray]
        self.hoi_obj_scores = None  # type: Union[None, np.ndarray]
        self.output_scores = None  # type: Union[None, np.ndarray]

        self.extra_info = None  # type: Union[None, Dict[str, Union[np.ndarray, csr_matrix]]]

        if prediction_dict is not None:
            self.__dict__.update(prediction_dict)

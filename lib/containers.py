from typing import Union

import numpy as np
import torch


class Prediction:
    def __init__(self, prediction_dict=None):
        self.img_fn = None
        self.obj_im_inds = None  # type: Union[None, np.ndarray]
        self.obj_boxes = None  # type: Union[None, np.ndarray]
        self.obj_scores = None  # type: Union[None, np.ndarray]
        self.ho_img_inds = None  # type: Union[None, np.ndarray]
        self.ho_pairs = None  # type: Union[None, np.ndarray]
        self.action_scores = None  # type: Union[None, np.ndarray]
        self.part_action_scores = None  # type: Union[None, np.ndarray]
        self.hoi_scores = None  # type: Union[None, np.ndarray]

        if prediction_dict is not None:
            self.__dict__.update(prediction_dict)


class PrecomputedExample:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.scale = None

        self.precomp_boxes_ext = None
        self.precomp_box_feats = None

        self.precomp_hoi_infos = None
        self.precomp_hoi_union_boxes = None
        self.precomp_hoi_union_feats = None

        self.precomp_box_labels = None
        self.precomp_action_labels = None


class PrecomputedMinibatch:
    def __init__(self):
        # All Torch tensors except when specified

        # Train attributes
        self.epoch = None
        self.iter = None

        # Image attributes
        self.img_scales = None  # NumPy vector of length M

        # Object attributes
        self.boxes_ext = None  # N batch 85, each [img_id, x1, y1, x2, y2, scores]
        self.box_feats = None  # N batch F, where F is the dimensionality of visual features

        # # Human-object pair attributes
        # Note that box indices in `ho_infos` (and its NumPy equivalent) are over all boxes, NOT relative to each specific image
        self.ho_infos_np = None  # R batch 3, each [img_id, human_ind, obj_ind]. NumPy array.
        self._ho_infos = None
        self.ho_union_boxes = None  # R batch 4, each [x1, y1, x2, y2]
        self.ho_union_boxes_feats = None  # R batch F

        # Labels
        self.box_labels = None  # N
        self.action_labels = None  # N batch #actions
        self.hoi_labels = None  # N batch #interactions

    @property
    def ho_infos(self):
        if self._ho_infos is None and self.ho_infos_np is not None:
            self._ho_infos = torch.tensor(self.ho_infos_np, device=self.box_feats.device)
        return self._ho_infos

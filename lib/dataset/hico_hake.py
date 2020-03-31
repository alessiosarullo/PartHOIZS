import json
import os
from typing import Dict, List

import numpy as np
import torch

from config import cfg
from lib.dataset.hico import Hico
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, ImgInstancesFeatProvider
from lib.dataset.utils import Splits, Dims, get_hico_to_coco_mapping
from lib.timer import Timer


class HicoHakeSplit(HoiDatasetSplit):
    def __init__(self, split, full_dataset, object_inds=None, action_inds=None):
        super().__init__(split, full_dataset, object_inds, action_inds)
        self.full_dataset = self.full_dataset  # type: HicoHake
        self.img_pstate_labels = self.full_dataset.split_part_annotations[self.split]

    @classmethod
    def instantiate_full_dataset(cls):
        return HicoHake()

    @property
    def dims(self) -> Dims:
        K_hake = self.full_dataset.num_keypoints
        B, S = self.full_dataset.num_parts, self.full_dataset.num_part_states
        return super().dims._replace(K_hake=K_hake, B=B, S=S)

    def _collate(self, idx_list, device):
        raise NotImplementedError


class HicoHakeKPSplit(HicoHakeSplit):
    def __init__(self, split, full_dataset, object_inds=None, action_inds=None, no_feats=False):
        super().__init__(split, full_dataset, object_inds, action_inds)
        self._feat_provider = ImgInstancesFeatProvider(ds=self, ds_name='hico', no_feats=no_feats,
                                                       obj_mapping=get_hico_to_coco_mapping(self.full_dataset.objects))
        self.non_empty_inds = np.intersect1d(self.non_empty_inds, self._feat_provider.non_empty_imgs)

    @property
    def dims(self) -> Dims:
        F_img = self._feat_provider.pc_img_feats.shape[1]
        F_kp = self._feat_provider.kp_net_dim
        F_obj = self._feat_provider.obj_feats_dim
        return super().dims._replace(F_img=F_img, F_kp=F_kp, F_obj=F_obj)

    def _collate(self, idx_list, device):
        Timer.get('GetBatch').tic()

        mb = self._feat_provider.collate(idx_list, device)
        idxs = np.array(idx_list)
        if self.split != Splits.TEST:
            img_labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            pstate_labels = torch.tensor(self.img_pstate_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            img_labels = pstate_labels = None
        mb = mb._replace(ex_labels=img_labels, pstate_labels=pstate_labels)

        Timer.get('GetBatch').toc(discard=5)
        return mb


class HicoHake(Hico):
    def __init__(self, *args, **kwargs):
        """
        In the following, BP = body parts and PS = (body) part state.
        Attributes:
            - parts: List[str]. The 6 parts in HAKE's annotations.
            - part_inv_ind: Dict[str, int]. Parts inverse index (string -> index in `parts`).
            - bp_ps_pairs: List[List[str]]. This is essentially the content of file `Part_State_76.txt`.
            - states_per_part: List[np.array]. Associates an array of indices in `part_action_pairs` to each part (same order as `parts`).
            - keypoints: List[str]. Names of HAKE's 10 keypoints.
            - part_to_kp: List[np.array]. Associates to each part an array of keypoints (e.g., at the index corresponding to 'foot' there will be
                            indices for ['right foot', 'left foot']).
        """
        super().__init__(*args, **kwargs)

        with open(os.path.join(cfg.data_root, 'HICO', 'HAKE', 'Part_State_76.txt'), 'r') as f:
            bp_ps_dict_str = {}
            bp_ps_pairs = []
            for i, l in enumerate(f.readlines()):
                if not l.strip():
                    break
                bp_ps_pair = [x.strip() for x in l.strip().split(':')]
                body_part, part_state = bp_ps_pair
                if part_state == 'no_interaction':
                    part_state = self.null_action
                bp_ps_dict_str.setdefault(body_part, []).append(len(bp_ps_pairs))
                bp_ps_pairs.append([body_part, part_state])
        self.parts = list(bp_ps_dict_str.keys())  # type: List[str]
        self.part_states = [f'{p} {a}' for p, a in bp_ps_pairs]  # type: List[str]
        self.part_inv_ind = {part: i for i, part in enumerate(self.parts)}  # type: Dict[str, int]
        self.bp_ps_pairs = bp_ps_pairs  # type: List[List[str]]
        self.states_per_part = [np.array(sorted(bp_ps_dict_str[p])) for p in self.parts]  # type: List[np.array]

        with open(os.path.join(cfg.data_root, 'HICO', 'HAKE', 'joints.txt'), 'r') as f:
            self.keypoints = [l.strip().lower().replace('_', ' ') for l in f.readlines()]  # type: List[str]
        self.part_to_kp = [np.array([j for j, kp in enumerate(self.keypoints) if p in kp]) for p in self.parts]  # type: List[np.array]
        assert np.all(np.sort(np.concatenate(self.part_to_kp)) == np.arange(len(self.keypoints)))
        kp_to_part = np.array([[1 if p in kp else 0 for i, p in enumerate(self.parts)] for kp in self.keypoints])
        assert kp_to_part.shape == (len(self.keypoints), len(self.parts)) and np.all(kp_to_part.sum(axis=1) == 1)
        self.kp_to_part = np.where(kp_to_part)[1].tolist()

        # Part annotations
        self.split_part_annotations = {}
        for split, fns in self.split_filenames.items():
            part_anns = json.load(open(os.path.join(cfg.data_root, 'HICO', 'HAKE', f'{split.value}.json'), 'r'))  # type: Dict
            part_ann_list = [part_anns[k] for k in fns]  # There are two extra files in this split
            part_ann_vecs = np.stack([np.concatenate([pa[f'{p}_list'] for p in self.parts]) for pa in part_ann_list], axis=0)
            self.split_part_annotations[split] = part_ann_vecs

    @property
    def num_parts(self):
        return len(self.parts)

    @property
    def num_keypoints(self):
        return len(self.keypoints)

    @property
    def num_part_states(self):
        return len(self.bp_ps_pairs)

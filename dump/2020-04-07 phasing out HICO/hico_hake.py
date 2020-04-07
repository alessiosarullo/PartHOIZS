import json
import os
from typing import Dict, List

import numpy as np

from config import cfg
from lib.dataset.hico import Hico
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, ImgInstancesFeatProvider
from lib.dataset.utils import Dims, get_hico_to_coco_mapping


class HicoHakeSplit(HoiDatasetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: HicoHake
        self.img_pstate_labels = self.full_dataset.split_part_annotations[self.split]

    @classmethod
    def instantiate_full_dataset(cls):
        return HicoHake()

    def _init_feat_provider(self, **kwargs):
        return ImgInstancesFeatProvider(ds=self, ds_name='hico', obj_mapping=get_hico_to_coco_mapping(self.full_dataset.objects), **kwargs)

    @property
    def dims(self) -> Dims:
        dims = super().dims
        K_hake = self.full_dataset.num_keypoints
        B, S = self.full_dataset.num_parts, self.full_dataset.num_part_states
        dims = dims._replace(K_hake=K_hake, B=B, S=S)
        if self._feat_provider is not None:
            F_img = self._feat_provider.pc_img_feats.shape[1]
            F_kp = self._feat_provider.kp_net_dim
            F_obj = self._feat_provider.obj_feats_dim
            dims = dims._replace(F_img=F_img, F_kp=F_kp, F_obj=F_obj)
        return dims


class HicoHake(Hico):
    def __init__(self):
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
        super().__init__()

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
        for split in ['train', 'test']:
            fns = self.split_filenames[split]
            part_anns = json.load(open(os.path.join(cfg.data_root, 'HICO', 'HAKE', f'{split}.json'), 'r'))  # type: Dict
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

import json
import os
from typing import Dict, List

import numpy as np
import torch

from config import cfg
from lib.dataset.hico import Hico, HicoSplit
from lib.dataset.utils import Splits
from lib.utils import Timer


class HicoHake(Hico):
    def __init__(self):
        super().__init__()

        with open(os.path.join(cfg.data_root, 'HICO', 'HAKE', 'Part_State_76.txt'), 'r') as f:
            part_action_dict_str = {}
            part_action_pairs = []
            for i, l in enumerate(f.readlines()):
                if not l.strip():
                    break
                pa_pair = [x.strip() for x in l.strip().split(':')]
                part, action = pa_pair
                if action == 'no_interaction':
                    action = self.null_action
                part_action_dict_str.setdefault(part, []).append(len(part_action_pairs))
                part_action_pairs.append([part, action])
        self.parts = list(part_action_dict_str.keys())  # type: List[str]
        self.part_inv_ind = {part: i for i, part in enumerate(self.parts)}  # type: Dict[str, int]
        self.part_actions_pairs = part_action_pairs  # type: List[List[str]]
        self.part_action_dict = {self.part_inv_ind[k]: np.array(v) for k, v in part_action_dict_str.items()}  # type: Dict[int, np.array]

        # Part annotations
        self.split_part_annotations = {}
        for split, fns in self.split_filenames.items():
            part_anns = json.load(open(os.path.join(cfg.data_root, 'HICO', 'HAKE', f'{split.value}.json'), 'r'))  # type: Dict
            part_ann_list = [part_anns[k] for k in fns]  # There are two extra files in this split
            part_ann_vecs = np.stack([np.concatenate([pa[f'{p}_list'] for p in self.parts]) for pa in part_ann_list], axis=0)
            self.split_part_annotations[split] = part_ann_vecs

    @property
    def human_class(self) -> int:
        return self.object_index['person']

    def get_img_dir(self, split):
        return self.split_img_dir[split]


class HicoHakeSplit(HicoSplit):
    def __init__(self, *args, **kwargs):
        super(HicoHakeSplit, self).__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: HicoHake
        self.part_labels = self.full_dataset.split_part_annotations[self._data_split]

    @classmethod
    def get_full_dataset(cls) -> HicoHake:
        return HicoHake()

    @property
    def num_parts(self):
        return len(self.full_dataset.parts)

    @property
    def num_part_actions(self):
        return len(self.full_dataset.part_actions_pairs)

    def _collate(self, idx_list, device):
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
        if self.split != Splits.TEST:
            labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            part_labels = torch.tensor(self.part_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            labels = part_labels = None
        Timer.get('GetBatch').toc()
        return feats, labels, part_labels, []


if __name__ == '__main__':
    HicoHake()

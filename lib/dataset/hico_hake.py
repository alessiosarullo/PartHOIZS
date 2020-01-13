import json
import os
import pickle
from collections import namedtuple
from typing import Dict, List

import h5py
import numpy as np
import torch

from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hico import Hico, HicoSplit
from lib.dataset.utils import Splits, get_hico_to_coco_mapping, COCO_CLASSES
from lib.utils import Timer
from PIL import Image


class PrecomputedFilesHandler:
    files = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def get_file(cls, file_name):
        return cls.files.setdefault(file_name, {}).setdefault('_handler', h5py.File(file_name, 'r'))

    @classmethod
    def get(cls, file_name, attribute_name, load_in_memory=True):
        file = cls.get_file(file_name)
        key = attribute_name
        if load_in_memory:
            key += '__loaded'
        if key not in cls.files[file_name].keys():
            attribute = file[attribute_name]
            if load_in_memory:
                attribute = attribute[:]
            cls.files[file_name][key] = attribute
        return cls.files[file_name][key]


class HicoHake(Hico):
    def __init__(self):
        """
        Attributes:
            - parts: List[str]. The 6 parts in HAKE's annotations.
            - part_inv_ind: Dict[str, int]. Parts inverse index (string -> index in `parts`).
            - part_actions_pairs: List[List[str]]. This is essentially the content of file `Part_State_76.txt`.
            - actions_per_part: List[np.array]. Associates an array of indices in `part_action_pairs` to each part (same order as `parts`).
            - keypoints: List[str]. Names of HAKE's 10 keypoints.
            - part_to_kp: List[np.array]. Associates to each part an array of keypoints (e.g., at the index corresponding to 'foot' there will be
                            indices for ['right foot', 'left foot']).
        """
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
        self.actions_per_part = [np.array(sorted(part_action_dict_str[p])) for p in self.parts]  # type: List[np.array]

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
    def human_class(self) -> int:
        return self.object_index['person']

    @property
    def num_parts(self):
        return len(self.parts)

    @property
    def num_part_actions(self):
        return len(self.part_actions_pairs)

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


class HicoHakeKPSplit(HicoHakeSplit):
    def __init__(self, *args, **kwargs):
        super(HicoHakeKPSplit, self).__init__(*args, **kwargs)
        hico_to_coco_mapping = get_hico_to_coco_mapping(self.full_dataset.objects)
        try:
            self.objects_fn = cfg.precomputed_feats_format % ('hicoobjs', 'mask_rcnn_X_101_32x8d_FPN_3x', self._data_split.value)
            fname_ids = PrecomputedFilesHandler.get(self.objects_fn, 'fname_ids')
            self.pc_obj_boxes = PrecomputedFilesHandler.get(self.objects_fn, 'boxes')
            self.pc_obj_feats = PrecomputedFilesHandler.get(self.objects_fn, 'box_feats')
            self.pc_obj_scores = PrecomputedFilesHandler.get(self.objects_fn, 'box_scores')

            # Filter out BG objects and persons (already have the BBs provided by the keypoint model).
            obj_classes = np.argmax(self.pc_obj_scores, axis=1)
            obj_to_keep = (obj_classes != COCO_CLASSES.index('__background__')) & (obj_classes != COCO_CLASSES.index('person'))
            fname_id_to_obj_inds = {}
            for i, fid in enumerate(fname_ids):
                if obj_to_keep[i]:
                    fname_id_to_obj_inds.setdefault(fid, []).append(i)
            fname_id_to_obj_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_obj_inds.items()}
            self.pc_obj_scores = self.pc_obj_scores[:, hico_to_coco_mapping]

            self.keypoints_fn = cfg.precomputed_feats_format % ('hicokps', 'keypoint_rcnn_R_101_FPN_3x', self._data_split.value)
            fname_ids = PrecomputedFilesHandler.get(self.keypoints_fn, 'fname_ids')
            self.pc_person_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'boxes')
            self.pc_coco_kps = PrecomputedFilesHandler.get(self.keypoints_fn, 'keypoints')
            assert np.all(np.min(self.pc_coco_kps[:, :, :2], axis=1) >= self.pc_person_boxes[:, :2])
            assert np.all(np.max(self.pc_coco_kps[:, :, :2], axis=1) <= self.pc_person_boxes[:, 2:4])
            self.pc_person_scores = PrecomputedFilesHandler.get(self.keypoints_fn, 'scores')
            self.pc_hake_kp_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_boxes')
            self.pc_hake_kp_feats = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_feats')
            fname_id_to_kp_inds = {}
            for i, fid in enumerate(fname_ids):
                fname_id_to_kp_inds.setdefault(fid, []).append(i)
            fname_id_to_kp_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_kp_inds.items()}

            # Useful quantities
            self.num_hake_kps = self.pc_hake_kp_boxes.shape[1]
            self.pc_kp_feats_dim = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_feats').shape[-1]

            # Cache
            self.pc_img_data = []  # type: List[Dict]

            imgs_with_kps = []
            for i, fname in enumerate(self.full_dataset.split_filenames[self._data_split]):
                fname_id = int(os.path.splitext(fname)[0].split('_')[-1])
                im_data = {}
                im_person_inds = fname_id_to_kp_inds.get(fname_id, None)
                if im_person_inds is not None:
                    im_person_inds = im_person_inds[np.any(self.pc_hake_kp_boxes[im_person_inds, :], axis=(1, 2))]
                    if im_person_inds.size > 0 and cfg.sbmar > 0:
                        pboxes = self.pc_person_boxes[im_person_inds, :]
                        pboxes_areas = np.prod(pboxes[:, 2:4] - pboxes[:, :2], axis=1)
                        pboxes_area_ratios = pboxes_areas / np.amax(pboxes_areas)
                        im_person_inds = im_person_inds[pboxes_area_ratios >= cfg.sbmar]
                    if im_person_inds.size > 0:
                        im_person_inds = np.atleast_1d(im_person_inds)
                        if 0 < cfg.max_ppl < len(im_person_inds):
                            im_person_scores = self.pc_person_scores[im_person_inds]
                            inds = np.argsort(im_person_scores)[::-1]
                            im_person_scores = im_person_scores[inds]
                            keep = im_person_scores > cfg.min_ppl_score
                            keep[0] = True  # always keep at least one person
                            if cfg.max_ppl > 0:
                                keep[cfg.max_ppl:] = False
                            im_person_inds = np.sort(im_person_inds[inds][keep])
                            assert len(im_person_inds) == cfg.max_ppl or np.all(im_person_scores[keep][1:] > cfg.min_ppl_score)
                        im_data['person_inds'] = im_person_inds
                        imgs_with_kps.append(i)

                        obj_inds = fname_id_to_obj_inds.get(fname_id, None)
                        if obj_inds is not None:
                            im_data['obj_inds'] = np.atleast_1d(obj_inds)

                self.pc_img_data.append(im_data)
            assert len(self.pc_img_data) == len(self.full_dataset.split_filenames[self._data_split])

            if cfg.spcfmdim > 0:
                fmap_dim = cfg.spcfmdim
                cache_fn = os.path.join(cfg.cache_root, f'kp_boxes_obj_prox_fmap{fmap_dim}_{self._data_split.value}.pkl')
                try:
                    with open(cache_fn, 'rb') as f:
                        self.kp_boxes_obj_proximity_fmaps, self.most_relevant_obj_per_kp_box = pickle.load(f)
                except FileNotFoundError:
                    self.kp_boxes_obj_proximity_fmaps = np.zeros((self.pc_hake_kp_boxes.shape[0],
                                                                  self.pc_hake_kp_boxes.shape[1],
                                                                  fmap_dim,
                                                                  fmap_dim
                                                                  ))
                    self.most_relevant_obj_per_kp_box = np.full((self.pc_hake_kp_boxes.shape[0], self.pc_hake_kp_boxes.shape[1]), fill_value=-1)
                    for i, im_data in enumerate(self.pc_img_data):
                        try:
                            im_person_inds = im_data['person_inds']
                            im_obj_inds = im_data['obj_inds']
                        except KeyError:
                            continue
                        img_wh = self.img_dims[i]
                        im_obj_boxes = np.round(self.pc_obj_boxes[im_obj_inds])

                        norm_dists = []
                        for obox in im_obj_boxes:
                            norm_dist_on_axes = []
                            for c in range(2):  # x, y
                                img_size = img_wh[c]
                                proj_on_axis = np.arange(obox[0 + c], obox[2 + c] + 1)
                                norm_dist_on_axis = np.min(np.abs(np.arange(img_size)[:, None] - proj_on_axis[None, :]), axis=1) / img_size
                                assert np.all((0 <= norm_dist_on_axis) & (norm_dist_on_axis < 1))
                                if c == 0:
                                    tiled_dist = np.tile(norm_dist_on_axis, [img_wh[1], 1])
                                else:
                                    tiled_dist = np.tile(norm_dist_on_axis, [img_wh[0], 1]).T
                                norm_dist_on_axes.append(tiled_dist)
                            norm_dist = np.max(np.stack(norm_dist_on_axes, axis=2), axis=2)
                            norm_dists.append(norm_dist)
                        normalised_proximity_from_objs = 1 - np.stack(norm_dists, axis=2)
                        assert np.all((0 < normalised_proximity_from_objs) & (normalised_proximity_from_objs <= 1))

                        for prs_idx in im_person_inds:
                            pboxes = self.pc_hake_kp_boxes[prs_idx, :, :4]
                            ious = compute_ious(pboxes, im_obj_boxes)
                            box_idx_per_kp = np.argmax(ious, axis=1)
                            for kpidx, pbox in enumerate(pboxes.astype(np.int)):
                                best_obj_idx = box_idx_per_kp[kpidx]
                                kp_box_proximity_to_obj = normalised_proximity_from_objs[pbox[1]:pbox[3] + 1, pbox[0]:pbox[2] + 1, best_obj_idx]

                                # Note: this does NOT preserve aspect ratio
                                kp_box_proximity_to_obj_fmap = np.asarray(Image.fromarray(kp_box_proximity_to_obj).
                                                                          resize((fmap_dim, fmap_dim), Image.BILINEAR))

                                self.kp_boxes_obj_proximity_fmaps[prs_idx, kpidx] = kp_box_proximity_to_obj_fmap
                                self.most_relevant_obj_per_kp_box[prs_idx, kpidx] = im_obj_inds[best_obj_idx]
                    with open(cache_fn, 'wb') as f:
                        pickle.dump((self.kp_boxes_obj_proximity_fmaps, self.most_relevant_obj_per_kp_box), f)

            self.non_empty_split_imgs = np.intersect1d(self.non_empty_split_imgs, np.array(imgs_with_kps))
        except OSError:
            raise
            # self.pc_img_data = None
            # self.pc_kp_feats_dim = None

    def _collate(self, idx_list, device):
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
        orig_img_wh = torch.tensor(self.img_dims[idxs, :], dtype=torch.float32, device=device)
        person_inds, obj_inds = [], []
        for i, idx in enumerate(idxs):
            im_data = self.pc_img_data[idx]
            if im_data:
                i_person_inds = im_data['person_inds']
                person_inds.append(i_person_inds)

                try:
                    i_obj_inds = im_data['obj_inds']
                    obj_inds.append(i_obj_inds)
                except KeyError:
                    obj_inds.append(np.arange(0))

        person_im_inds = obj_im_inds = []
        person_boxes = person_kps = kp_boxes = kp_feats = None
        obj_boxes = obj_feats = obj_scores = kp_box_prox_to_obj_fmaps = obj_scores_per_kp_box = None
        if person_inds:
            c_num_per_img = np.cumsum([pi.shape[0] for pi in person_inds])
            person_im_inds = [np.arange((c_num_per_img[i - 1] if i > 0 else 0), c) for i, c in enumerate(c_num_per_img)]

            c_num_per_img = np.cumsum([pi.shape[0] for pi in obj_inds])
            obj_im_inds = [np.arange((c_num_per_img[i - 1] if i > 0 else 0), c) for i, c in enumerate(c_num_per_img)]

            person_inds = np.concatenate(person_inds)
            obj_inds = np.concatenate(obj_inds)

            person_boxes = torch.tensor(self.pc_person_boxes[person_inds], dtype=torch.float32, device=device)
            person_kps = torch.tensor(self.pc_coco_kps[person_inds], dtype=torch.float32, device=device)
            kp_boxes = torch.tensor(self.pc_hake_kp_boxes[person_inds], dtype=torch.float32, device=device)
            kp_feats = torch.tensor(self.pc_hake_kp_feats[person_inds], dtype=torch.float32, device=device)
            obj_boxes = torch.tensor(self.pc_obj_boxes[obj_inds], dtype=torch.float32, device=device)
            obj_scores = torch.tensor(self.pc_obj_scores[obj_inds], dtype=torch.float32, device=device)
            obj_feats = torch.tensor(self.pc_obj_feats[obj_inds], dtype=torch.float32, device=device)
            if cfg.spcfmdim > 0:
                kp_box_prox_to_obj_fmaps = torch.tensor(self.kp_boxes_obj_proximity_fmaps[person_inds], dtype=torch.float32, device=device)
                inds = self.most_relevant_obj_per_kp_box[person_inds]
                obj_scores_per_kp_box = np.where((inds >= 0)[:, :, None], self.pc_obj_scores[inds, :], 0)
                obj_scores_per_kp_box = torch.tensor(obj_scores_per_kp_box, dtype=torch.float32, device=device)

        if self.split != Splits.TEST:
            labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            part_labels = torch.tensor(self.part_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            labels = part_labels = None

        Timer.get('GetBatch').toc()
        mb = namedtuple('Minibatch', ['img_attrs', 'person_attrs', 'obj_attrs', 'img_labels', 'part_labels', 'other'])
        return mb(img_attrs=[feats, orig_img_wh],
                  person_attrs=[person_im_inds, person_boxes, person_kps, kp_boxes, kp_feats],
                  obj_attrs=[obj_im_inds, obj_boxes, obj_scores, obj_feats, kp_box_prox_to_obj_fmaps, obj_scores_per_kp_box],
                  img_labels=labels, part_labels=part_labels,
                  other=[])


if __name__ == '__main__':
    HicoHake()

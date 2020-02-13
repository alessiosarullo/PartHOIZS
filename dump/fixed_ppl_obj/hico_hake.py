import json
import os
from collections import namedtuple
from typing import Dict, List

import h5py
import numpy as np
import torch

from config import cfg
from lib.dataset.hico import Hico, HicoSplit
from lib.dataset.utils import Splits, get_hico_to_coco_mapping, COCO_CLASSES, filter_on_score
from lib.utils import Timer


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
    def human_class(self) -> int:
        return self.object_index['person']

    @property
    def num_parts(self):
        return len(self.parts)

    @property
    def num_keypoints(self):
        return len(self.keypoints)

    @property
    def num_part_states(self):
        return len(self.bp_ps_pairs)

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
        return len(self.full_dataset.bp_ps_pairs)

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
    def __init__(self, no_feats=False, *args, **kwargs):
        super(HicoHakeKPSplit, self).__init__(*args, **kwargs)
        hico_to_coco_mapping = get_hico_to_coco_mapping(self.full_dataset.objects)
        try:
            self.objects_fn = cfg.precomputed_feats_format % ('hicoobjs', 'mask_rcnn_X_101_32x8d_FPN_3x',
                                                              f'{self._data_split.value}{"__nofeats" if no_feats else ""}')
            self.obj_boxes = PrecomputedFilesHandler.get(self.objects_fn, 'boxes')
            self.obj_scores = PrecomputedFilesHandler.get(self.objects_fn, 'box_scores')[:, hico_to_coco_mapping]
            if not no_feats:
                self.obj_feats = PrecomputedFilesHandler.get(self.objects_fn, 'box_feats')
                self.obj_feats_dim = self.obj_feats.shape[-1]
            assert self.obj_boxes.dtype == self.obj_scores.dtype == self.obj_feats.dtype == np.float32

            self.keypoints_fn = cfg.precomputed_feats_format % \
                                ('hicokps', 'keypoint_rcnn_R_101_FPN_3x', f'{self._data_split.value}{"__nofeats" if no_feats else ""}')
            self.person_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'boxes')
            self.coco_kps = PrecomputedFilesHandler.get(self.keypoints_fn, 'keypoints')
            self.person_scores = PrecomputedFilesHandler.get(self.keypoints_fn, 'scores')
            self.hake_kp_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_boxes')
            assert self.hake_kp_boxes.shape[1] == self.full_dataset.num_keypoints
            if not no_feats:
                self.hake_kp_feats = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_feats')
                self.kp_feats_dim = self.hake_kp_feats.shape[-1]
            assert np.all(np.min(self.coco_kps[:, :, :2], axis=1) >= self.person_boxes[:, :2])
            assert np.all(np.max(self.coco_kps[:, :, :2], axis=1) <= self.person_boxes[:, 2:4])
            assert self.person_boxes.dtype == self.coco_kps.dtype == self.person_scores.dtype == self.hake_kp_boxes.dtype == np.float32

            # Cache
            self.prs_and_obj_inds_per_img, imgs_with_kps = self._compute_img_data()  # type: List[Dict]
            assert len(self.prs_and_obj_inds_per_img) == len(self.full_dataset.split_filenames[self._data_split])

            if cfg.keep_unlabelled:
                self.non_empty_split_imgs = np.array(imgs_with_kps)
            else:
                self.non_empty_split_imgs = np.intersect1d(self.non_empty_split_imgs, np.array(imgs_with_kps))
        except OSError:
            raise

    def _compute_img_data(self):
        # Filter out BG objects and persons (already have the BBs provided by the keypoint model).
        obj_classes = np.argmax(PrecomputedFilesHandler.get(self.objects_fn, 'box_scores'), axis=1)
        obj_to_keep = (obj_classes != COCO_CLASSES.index('__background__')) & (obj_classes != COCO_CLASSES.index('person'))
        fname_id_to_obj_inds = {}
        for i, fid in enumerate(PrecomputedFilesHandler.get(self.objects_fn, 'fname_ids')):
            if obj_to_keep[i]:
                fname_id_to_obj_inds.setdefault(fid, []).append(i)
        fname_id_to_obj_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_obj_inds.items()}

        fname_id_to_kp_inds = {}
        for i, fid in enumerate(PrecomputedFilesHandler.get(self.keypoints_fn, 'fname_ids')):
            fname_id_to_kp_inds.setdefault(fid, []).append(i)
        fname_id_to_kp_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_kp_inds.items()}

        img_inds = []
        imgs_with_kps = []
        for i, fname in enumerate(self.full_dataset.split_filenames[self._data_split]):
            fname_id = int(os.path.splitext(fname)[0].split('_')[-1])
            im_data = {}

            # People
            im_person_inds = fname_id_to_kp_inds.get(fname_id, None)
            if im_person_inds is not None:
                im_person_inds = im_person_inds[np.any(self.hake_kp_boxes[im_person_inds, :], axis=(1, 2))]
                if im_person_inds.size > 0 and cfg.sbmar > 0:  # filter too small persons (with respect to the largest one)
                    pboxes = self.person_boxes[im_person_inds, :]
                    pboxes_areas = np.prod(pboxes[:, 2:4] - pboxes[:, :2], axis=1)
                    pboxes_area_ratios = pboxes_areas / np.amax(pboxes_areas)
                    im_person_inds = im_person_inds[pboxes_area_ratios >= cfg.sbmar]
                if im_person_inds.size > 0:
                    im_person_inds = np.atleast_1d(im_person_inds)
                    inds = filter_on_score(scores=self.person_scores[im_person_inds], min_score=cfg.min_ppl_score, maxn=cfg.max_ppl, keep_one=True)
                    assert inds.size > 0
                    im_data['person_inds'] = np.sort(im_person_inds[inds])
                    imgs_with_kps.append(i)

            # Objects
            obj_inds = fname_id_to_obj_inds.get(fname_id, None)
            if obj_inds is not None:
                im_obj_inds = np.atleast_1d(obj_inds)
                scores = np.max(self.obj_scores[im_obj_inds, :], axis=1)
                inds = filter_on_score(scores=scores, min_score=None, maxn=cfg.max_obj, keep_one=False)
                assert inds.size > 0
                im_data['obj_inds'] = np.sort(im_obj_inds[inds])

            img_inds.append(im_data)
        return img_inds, imgs_with_kps

    def _old_collate(self, idx_list, device):
        # TODO delete
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
        orig_img_wh = torch.tensor(self.img_dims[idxs, :], dtype=torch.float32, device=device)
        person_inds, obj_inds = [], []
        for i, idx in enumerate(idxs):
            im_data = self.prs_and_obj_inds_per_img[idx]
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

            person_boxes = torch.tensor(self.person_boxes[person_inds], dtype=torch.float32, device=device)
            person_kps = torch.tensor(self.coco_kps[person_inds], dtype=torch.float32, device=device)
            kp_boxes = torch.tensor(self.hake_kp_boxes[person_inds], dtype=torch.float32, device=device)
            kp_feats = torch.tensor(self.hake_kp_feats[person_inds], dtype=torch.float32, device=device)
            obj_boxes = torch.tensor(self.obj_boxes[obj_inds], dtype=torch.float32, device=device)
            obj_scores = torch.tensor(self.obj_scores[obj_inds], dtype=torch.float32, device=device)
            obj_feats = torch.tensor(self.obj_feats[obj_inds], dtype=torch.float32, device=device)

        if self.split != Splits.TEST:
            labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            part_labels = torch.tensor(self.part_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            labels = part_labels = None

        im_ids = [f'{self.split.value}_{idx}' for idx in idxs]

        Timer.get('GetBatch').toc()
        mb = namedtuple('Minibatch', ['img_attrs', 'person_attrs', 'obj_attrs', 'img_labels', 'part_labels', 'other'])
        return mb(img_attrs=[im_ids, feats, orig_img_wh],
                  person_attrs=[person_im_inds, person_boxes, person_kps, kp_boxes, kp_feats],
                  obj_attrs=[obj_im_inds, obj_boxes, obj_scores, obj_feats, kp_box_prox_to_obj_fmaps, obj_scores_per_kp_box],
                  img_labels=labels, part_labels=part_labels,
                  other=[])

    def _collate(self, idx_list, device):
        if cfg.max_ppl == 0 or cfg.max_obj == 0:
            return self._old_collate(idx_list, device)

        # This assumes the filtering has been done on initialisation (i.e., there are no more people/objects than the max number)
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)

        # Image data
        im_ids = [f'{self.split.value}_{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
        orig_img_wh = torch.tensor(self.img_dims[idxs, :], dtype=torch.float32, device=device)

        N, P, M = idxs.size, cfg.max_ppl, cfg.max_obj  # #imgs, #people, #objects
        K_coco, K_hake = self.coco_kps.shape[1], self.full_dataset.num_keypoints  # #keypoints
        B, M = self.full_dataset.num_parts, self.full_dataset.num_objects  # #body parts, #object classes
        F_kp, F_obj = self.kp_feats_dim, self.obj_feats_dim  # feature vector dimensionality
        ppl_boxes = np.zeros((N, P, 4), dtype=np.float32)
        coco_kps = np.zeros((N, P, K_coco, 3), dtype=np.float32)
        kp_boxes = np.zeros((N, P, K_hake, 5), dtype=np.float32)
        kp_feats = np.zeros((N, P, K_hake, F_kp), dtype=np.float32)
        obj_boxes = np.zeros((N, M, 4), dtype=np.float32)
        obj_scores = np.zeros((N, M, M), dtype=np.float32)
        obj_feats = np.zeros((N, M, F_obj), dtype=np.float32)

        person_inds = np.full((N, P), fill_value=np.nan)
        obj_inds = np.full((N, M), fill_value=np.nan)
        for i, idx in enumerate(idxs):
            im_data = self.prs_and_obj_inds_per_img[idx]
            try:
                i_person_inds = im_data['person_inds']
                P_i = i_person_inds.size
                assert P_i <= P
                person_inds[i, :P_i] = i_person_inds
            except KeyError:
                pass

            try:
                i_obj_inds = im_data['obj_inds']
                M_i = i_obj_inds.size
                assert M_i <= M
                obj_inds[i, :M_i] = i_obj_inds
            except KeyError:
                pass

        valid_person_inds_mask = ~np.isnan(person_inds)
        valid_person_inds = person_inds[valid_person_inds_mask].astype(np.int)
        ppl_boxes[valid_person_inds_mask] = self.person_boxes[valid_person_inds]
        coco_kps[valid_person_inds_mask] = self.coco_kps[valid_person_inds]
        kp_boxes[valid_person_inds_mask] = self.hake_kp_boxes[valid_person_inds]
        kp_feats[valid_person_inds_mask] = self.hake_kp_feats[valid_person_inds]

        valid_obj_inds_mask = ~np.isnan(obj_inds)
        valid_obj_inds = obj_inds[valid_obj_inds_mask].astype(np.int)
        obj_boxes[valid_obj_inds_mask] = self.obj_boxes[valid_obj_inds]
        obj_scores[valid_obj_inds_mask] = self.obj_scores[valid_obj_inds]
        obj_feats[valid_obj_inds_mask] = self.obj_feats[valid_obj_inds]

        ppl_boxes = torch.from_numpy(ppl_boxes).to(device=device)
        coco_kps = torch.from_numpy(coco_kps).to(device=device)
        kp_boxes = torch.from_numpy(kp_boxes).to(device=device)
        kp_feats = torch.from_numpy(kp_feats).to(device=device)
        obj_boxes = torch.from_numpy(obj_boxes).to(device=device)
        obj_scores = torch.from_numpy(obj_scores).to(device=device)
        obj_feats = torch.from_numpy(obj_feats).to(device=device)

        if self.split != Splits.TEST:
            labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            part_labels = torch.tensor(self.part_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            labels = part_labels = None

        Timer.get('GetBatch').toc(discard=5)
        mb = namedtuple('Minibatch', ['img_attrs', 'person_attrs', 'obj_attrs', 'img_labels', 'part_labels', 'other'])
        return mb(img_attrs=[im_ids, feats, orig_img_wh],
                  person_attrs=[ppl_boxes, coco_kps, kp_boxes, kp_feats],
                  obj_attrs=[obj_boxes, obj_scores, obj_feats],
                  img_labels=labels, part_labels=part_labels,
                  other=[])


if __name__ == '__main__':
    HicoHake()

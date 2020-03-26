from typing import List, Dict, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset
from lib.dataset.tin_utils import get_next_sp_with_pose
from lib.dataset.utils import Splits, Dims, Minibatch, PrecomputedFilesHandler, COCO_CLASSES, filter_on_score, get_hico_to_coco_mapping, HoiData


class AbstractHoiDatasetSplit(Dataset):
    def __init__(self, split):
        assert split != Splits.VAL
        self.split = split

    @property
    def num_objects(self):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError

    @property
    def num_interactions(self):
        raise NotImplementedError

    @property
    def num_images(self):
        raise NotImplementedError

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class HoiDatasetSplit(AbstractHoiDatasetSplit):
    def __init__(self, split, full_dataset: HoiDataset, object_inds=None, action_inds=None):
        super().__init__(split)
        self.full_dataset = full_dataset  # type: HoiDataset
        self.img_dims = self.full_dataset.split_img_dims[self.split]
        self.fnames = self.full_dataset.split_filenames[self.split]
        self.keep_inds = self.holdout_inds = None  # These will be set later, if needed.

        object_inds = sorted(object_inds) if object_inds is not None else range(self.full_dataset.num_objects)
        self.objects = [full_dataset.objects[i] for i in object_inds]
        self.seen_objects = np.array(object_inds, dtype=np.int)

        action_inds = sorted(action_inds) if action_inds is not None else range(self.full_dataset.num_actions)
        self.actions = [full_dataset.actions[i] for i in action_inds]
        self.seen_actions = np.array(action_inds, dtype=np.int)

        seen_op_mat = self.full_dataset.oa_pair_to_interaction[self.seen_objects, :][:, self.seen_actions]
        seen_interactions = set(np.unique(seen_op_mat).tolist()) - {-1}
        self.seen_interactions = np.array(sorted(seen_interactions), dtype=np.int)
        self.interactions = self.full_dataset.interactions[self.seen_interactions, :]  # original action and object inds

        self.labels = self._get_labels()
        if self.seen_interactions.size < self.full_dataset.num_interactions:
            all_labels = self.labels
            self.labels = np.zeros_like(all_labels)
            self.labels[:, self.seen_interactions] = all_labels[:, self.seen_interactions]
        self.non_empty_inds = np.flatnonzero(np.any(self.labels, axis=1))

        self._feat_provider = None  # type: Union[None, FeatProvider]

    def _get_labels(self):
        return self.full_dataset.split_labels[self.split]

    def hold_out(self, ratio):
        if cfg.no_filter_bg_only:
            num_imgs = self.num_images
            image_ids = np.arange(num_imgs)
        else:
            num_imgs = len(self.non_empty_inds)
            image_ids = self.non_empty_inds
        num_keep_imgs = num_imgs - int(num_imgs * ratio)
        keep_inds = np.random.choice(image_ids, size=num_keep_imgs, replace=False)
        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(image_ids, keep_inds)

    def get_img(self, img_id):
        img_fn = self.full_dataset.get_img_path(self.split, self.full_dataset.split_filenames[self.split][img_id])
        img = Image.open(img_fn).convert('RGB')
        return img

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def num_images(self):
        return len(self.full_dataset.split_filenames[self.split])

    @property
    def dims(self) -> Dims:
        P, M = cfg.max_ppl, cfg.max_obj
        K_coco, K_hake = 17, None  # FIXME magic constant
        B, S = None, None
        O, A, C = self.full_dataset.num_objects, self.full_dataset.num_actions, self.full_dataset.num_interactions
        F_img, F_kp, F_obj = None, None, None
        if cfg.tin:
            D = cfg.ipsize
        else:
            D = None
        return Dims(N=None, P=P, M=M, K_coco=K_coco, K_hake=K_hake, B=B, S=S, O=O, A=A, C=C, F_img=F_img, F_kp=F_kp, F_obj=F_obj, D=D)

    def _collate(self, idx_list, device):
        raise NotImplementedError

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=None, holdout_set=False, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        if drop_last is None:
            drop_last = False if self.split == Splits.TEST else True
        batch_size = batch_size * num_gpus

        if self.split == Splits.TEST:
            ds = self
        else:
            if self.keep_inds is None:
                assert self.holdout_inds is None and holdout_set is False
                ds = self
            else:
                ds = Subset(self, self.holdout_inds if holdout_set else self.keep_inds)
        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: self._collate(x, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, idx):
        # This should only be used by the data loader (see above).
        return idx

    def __len__(self):
        return self.num_images

    @classmethod
    def instantiate_full_dataset(cls) -> HoiDataset:
        raise NotImplementedError

    @classmethod
    def get_splits(cls, act_inds=None, obj_inds=None):
        splits = {}
        full_dataset = cls.instantiate_full_dataset()

        train_split = cls(split=Splits.TRAIN, full_dataset=full_dataset, object_inds=obj_inds, action_inds=act_inds)
        if cfg.val_ratio > 0:
            train_split.hold_out(ratio=cfg.val_ratio)
        splits[Splits.TRAIN] = train_split
        splits[Splits.TEST] = cls(split=Splits.TEST, full_dataset=full_dataset)

        train_str = Splits.TRAIN.value.capitalize()
        if obj_inds is not None:
            print(f'{train_str} objects ({train_split.seen_objects.size}):', train_split.seen_objects.tolist())
        if act_inds is not None:
            print(f'{train_str} actions ({train_split.seen_actions.size}):', train_split.seen_actions.tolist())
        if obj_inds is not None or act_inds is not None:
            print(f'{train_str} interactions ({train_split.seen_interactions.size}):', train_split.seen_interactions.tolist())

        return splits


class FeatProvider:
    def __init__(self, ds: HoiDatasetSplit, no_feats=False):
        super().__init__()
        self.wrapped_ds = ds
        self.full_dataset = self.wrapped_ds.full_dataset
        self.split = self.wrapped_ds.split

        hico_to_coco_mapping = get_hico_to_coco_mapping(self.full_dataset.objects)

        #############################################################################################################################
        # Load precomputed data
        #############################################################################################################################
        #####################################################
        # Image
        #####################################################
        img_fn = cfg.precomputed_feats_format % ('hico', 'resnet152', self.split.value)
        self.pc_img_feats = PrecomputedFilesHandler.get(img_fn, 'img_feats', load_in_memory=True)

        #####################################################
        # Objects
        #####################################################
        self.objects_fn = cfg.precomputed_feats_format % ('hicoobjs', 'mask_rcnn_X_101_32x8d_FPN_3x',
                                                          f'{self.split.value}{"__nofeats" if no_feats else ""}')
        self.obj_boxes = PrecomputedFilesHandler.get(self.objects_fn, 'boxes', load_in_memory=True)
        self.obj_scores = PrecomputedFilesHandler.get(self.objects_fn, 'box_scores', load_in_memory=True)[:, hico_to_coco_mapping]
        if not no_feats:
            self.obj_feats = PrecomputedFilesHandler.get(self.objects_fn, 'box_feats')
            self.obj_feats_dim = self.obj_feats.shape[-1]
            assert self.obj_feats.dtype == np.float32
        assert self.obj_boxes.dtype == self.obj_scores.dtype == np.float32

        #####################################################
        # People
        #####################################################
        self.keypoints_fn = cfg.precomputed_feats_format % \
                            ('hicokps', 'keypoint_rcnn_R_101_FPN_3x', f'{self.split.value}{"__nofeats" if no_feats else ""}')
        self.person_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'boxes', load_in_memory=True)
        self.coco_kps = PrecomputedFilesHandler.get(self.keypoints_fn, 'keypoints', load_in_memory=True)
        self.person_scores = PrecomputedFilesHandler.get(self.keypoints_fn, 'scores', load_in_memory=True)
        self.hake_kp_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_boxes', load_in_memory=True)
        try:
            assert self.hake_kp_boxes.shape[1] == self.full_dataset.num_keypoints
        except AttributeError:
            pass
        if not no_feats:
            self.hake_kp_feats = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_feats')
            self.person_feats = PrecomputedFilesHandler.get(self.keypoints_fn, 'person_feats')
            self.kp_net_dim = self.hake_kp_feats.shape[-1]
            assert self.kp_net_dim == self.person_feats.shape[-1]
            assert self.hake_kp_feats.dtype == np.float32
        assert np.all(np.min(self.coco_kps[:, :, :2], axis=1) >= self.person_boxes[:, :2])
        assert np.all(np.max(self.coco_kps[:, :, :2], axis=1) <= self.person_boxes[:, 2:4])
        assert self.person_boxes.dtype == self.coco_kps.dtype == self.person_scores.dtype == self.hake_kp_boxes.dtype == np.float32

        #############################################################################################################################
        # Cache variables to speed up loading
        #############################################################################################################################
        self.img_data_cache, imgs_with_kps = self._compute_img_data()  # type: List[Dict], List
        assert len(self.img_data_cache) == len(self.wrapped_ds.fnames)
        self.non_empty_imgs = np.array(imgs_with_kps)

    def _compute_img_data(self):
        """
        Fields: 'fname_id': int, 'person_inds': array, 'obj_inds': array
        """

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

        all_img_data = []
        imgs_with_kps = []
        for i, fname in enumerate(self.wrapped_ds.fnames):
            fname_id = self.wrapped_ds.full_dataset.get_fname_id(fname)
            im_data = {'fname_id': fname_id}

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

            all_img_data.append(im_data)
        return all_img_data, imgs_with_kps

    def collate(self, idx_list, device) -> Minibatch:
        raise NotImplementedError


class ImgInstancesFeatProvider(FeatProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate(self, idx_list, device) -> Minibatch:
        # This assumes the filtering has been done on initialisation (i.e., there are no more people/objects than the max number)
        idxs = np.array(idx_list)

        # Example data
        ex_ids = [f'ex{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)

        # Image data
        im_ids = [f'{self.split.value}_{idx}' for idx in idxs]
        im_feats = feats
        orig_img_wh = torch.tensor(self.wrapped_ds.img_dims[idxs, :], dtype=torch.float32, device=device)

        dims = self.wrapped_ds.dims
        P, M, K_coco, K_hake, O, F_kp, F_obj, D = dims.P, dims.M, dims.K_coco, dims.K_hake, dims.O, dims.F_kp, dims.F_obj, dims.D
        N = idxs.size  # #imgs

        ppl_boxes = np.zeros((N, P, 4), dtype=np.float32)
        ppl_feats = np.zeros((N, P, F_kp), dtype=np.float32)
        coco_kps = np.zeros((N, P, K_coco, 3), dtype=np.float32)
        kp_boxes = np.zeros((N, P, K_hake, 5), dtype=np.float32)
        kp_feats = np.zeros((N, P, K_hake, F_kp), dtype=np.float32)
        obj_boxes = np.zeros((N, M, 4), dtype=np.float32)
        obj_scores = np.zeros((N, M, O), dtype=np.float32)
        obj_feats = np.zeros((N, M, F_obj), dtype=np.float32)

        person_inds = np.full((N, P), fill_value=np.nan)
        obj_inds = np.full((N, M), fill_value=np.nan)
        for i, idx in enumerate(idxs):
            im_data = self.img_data_cache[idx]
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
        valid_person_mask = np.zeros(self.person_boxes.shape[0], dtype=bool)  # using mask instead of inds in case H5 file are not loaded in memory.
        valid_person_mask[person_inds[valid_person_inds_mask].astype(np.int)] = True
        ppl_boxes[valid_person_inds_mask] = self.person_boxes[valid_person_mask]
        ppl_feats[valid_person_inds_mask] = self.person_feats[valid_person_mask, ...]
        coco_kps[valid_person_inds_mask] = self.coco_kps[valid_person_mask]
        kp_boxes[valid_person_inds_mask] = self.hake_kp_boxes[valid_person_mask]
        kp_feats[valid_person_inds_mask] = self.hake_kp_feats[valid_person_mask, ...]

        valid_obj_inds_mask = ~np.isnan(obj_inds)
        valid_obj_mask = np.zeros(self.obj_boxes.shape[0], dtype=bool)
        valid_obj_mask[obj_inds[valid_obj_inds_mask].astype(np.int)] = True
        obj_boxes[valid_obj_inds_mask] = self.obj_boxes[valid_obj_mask]
        obj_scores[valid_obj_inds_mask] = self.obj_scores[valid_obj_mask]
        obj_feats[valid_obj_inds_mask] = self.obj_feats[valid_obj_mask, ...]

        t_ppl_boxes = torch.from_numpy(ppl_boxes).to(device=device)
        t_ppl_feats = torch.from_numpy(ppl_feats).to(device=device)
        t_coco_kps = torch.from_numpy(coco_kps).to(device=device)
        t_kp_boxes = torch.from_numpy(kp_boxes).to(device=device)
        t_kp_feats = torch.from_numpy(kp_feats).to(device=device)
        t_obj_boxes = torch.from_numpy(obj_boxes).to(device=device)
        t_obj_scores = torch.from_numpy(obj_scores).to(device=device)
        t_obj_feats = torch.from_numpy(obj_feats).to(device=device)

        mb = Minibatch(ex_data=[ex_ids, feats],
                       im_data=[im_ids, im_feats, orig_img_wh],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       ex_labels=None,
                       pstate_labels=None,
                       other=[])
        if cfg.tin:
            interactiveness_patterns = np.zeros((N, P, M, D, D, 3 + K_hake), dtype=np.float32)
            for i in range(N):
                valid_person_inds_mask_i = valid_person_inds_mask[i]
                valid_obj_inds_mask_i = valid_obj_inds_mask[i]
                pattern = get_next_sp_with_pose(human_boxes=ppl_boxes[i, valid_person_inds_mask_i],
                                                human_poses=coco_kps[i, valid_person_inds_mask_i],
                                                object_boxes=obj_boxes[i, valid_obj_inds_mask_i],
                                                size=D,
                                                part_boxes=kp_boxes[i, valid_person_inds_mask_i, :, :4]
                                                )
                r, c = np.where(valid_person_inds_mask_i[:, None] & valid_obj_inds_mask_i[None, :])
                interactiveness_patterns[i, r, c] = pattern.reshape((-1, *pattern.shape[2:]))
            t_interactiveness_patterns = torch.from_numpy(interactiveness_patterns).to(device=device)
            mb.person_data.append(t_interactiveness_patterns)
        return mb


class HoiInstancesFeatProvider(FeatProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hoi_data_cache = self._compute_hoi_data()  # type: List[HoiData]
        self.hoi_data_cache_np = np.stack([np.concatenate([x[:-1], x[-1]]) for x in self.hoi_data_cache])  # type: np.ndarray[np.int]

    def _compute_hoi_data(self):
        """
        Fields: 'fname_id', 'person_idx', 'obj_idx', 'hoi_class'
        """
        det_data = self.full_dataset._split_det_data[self.split]
        im_idx_to_ho_pairs_inds = {}
        for i, (image_idx, hum_idx, obj_idx) in enumerate(det_data.ho_pairs):
            im_idx_to_ho_pairs_inds.setdefault(image_idx, []).append(i)
        im_idx_to_ho_pairs_inds = {k: np.array(v) for k, v in im_idx_to_ho_pairs_inds.items()}

        all_hoi_data = []
        for im_idx, imdata in enumerate(self.img_data_cache):
            if 'person_inds' in imdata and 'obj_inds' in imdata:
                ho_pairs_inds = im_idx_to_ho_pairs_inds[im_idx]
                gt_ho_pairs = det_data.ho_pairs[ho_pairs_inds, 1:]
                gt_hoi_labels = det_data.labels[ho_pairs_inds, :]
                # TODO
                for p in imdata.get('person_inds', []):
                    for o in imdata.get('obj_inds', []):
                        hoi_data = HoiData(im_idx=im_idx,
                                           fname_id=imdata['fname_id'],
                                           p_idx=p,
                                           o_idx=o,
                                           hoi_classes=None,  # FIXME
                                           )
                        all_hoi_data.append(hoi_data)
        return all_hoi_data

    def collate(self, idx_list, device) -> Minibatch:
        idxs = np.array(idx_list)
        im_idxs = self.hoi_data_cache_np[idxs, 0]

        # Example data
        ex_ids = [f'ex{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[im_idxs, :], dtype=torch.float32, device=device)

        # Image data
        im_ids = [f'{self.split.value}_{idx}' for idx in im_idxs]
        im_feats = feats
        orig_img_wh = torch.tensor(self.img_dims[im_idxs, :], dtype=torch.float32, device=device)

        dims = self.wrapped_ds.dims
        P, M, K_coco, K_hake, B, O, F_kp, F_obj, D = dims.P, dims.M, dims.K_coco, dims.K_hake, dims.B, dims.O, dims.F_kp, dims.F_obj, dims.D
        N = idxs.size

        person_inds = self.hoi_data_cache_np[idxs, 2].astype(np.int, copy=False)
        obj_inds = self.hoi_data_cache_np[idxs, 3].astype(np.int, copy=False)

        ppl_boxes = self.person_boxes[person_inds]
        ppl_feats = self.person_feats[person_inds]
        coco_kps = self.coco_kps[person_inds]
        kp_boxes = self.hake_kp_boxes[person_inds]
        kp_feats = self.hake_kp_feats[person_inds]

        obj_boxes = self.obj_boxes[obj_inds]
        obj_scores = self.obj_scores[obj_inds]
        obj_feats = self.obj_feats[obj_inds]

        t_ppl_boxes = torch.from_numpy(ppl_boxes).to(device=device)
        t_ppl_feats = torch.from_numpy(ppl_feats).to(device=device)
        t_coco_kps = torch.from_numpy(coco_kps).to(device=device)
        t_kp_boxes = torch.from_numpy(kp_boxes).to(device=device)
        t_kp_feats = torch.from_numpy(kp_feats).to(device=device)
        t_obj_boxes = torch.from_numpy(obj_boxes).to(device=device)
        t_obj_scores = torch.from_numpy(obj_scores).to(device=device)
        t_obj_feats = torch.from_numpy(obj_feats).to(device=device)

        mb = Minibatch(ex_data=[ex_ids, feats],
                       im_data=[im_ids, im_feats, orig_img_wh],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       ex_labels=None,
                       pstate_labels=None,
                       other=[])
        if cfg.tin:
            assert P == M == 1
            interactiveness_patterns = np.zeros((N, P, M, D, D, 3 + K_hake), dtype=np.float32)
            for i in range(N):
                pattern = get_next_sp_with_pose(human_boxes=ppl_boxes[i, 0],
                                                human_poses=coco_kps[i, 0],
                                                object_boxes=obj_boxes[i, 0],
                                                size=D,
                                                part_boxes=kp_boxes[i, 0, :, :4]
                                                )
                interactiveness_patterns[i] = pattern
            t_interactiveness_patterns = torch.from_numpy(interactiveness_patterns).to(device=device)
            mb.person_data.append(t_interactiveness_patterns)

        return mb

import os
from typing import List, Dict, Union, NamedTuple

import numpy as np
import torch
from PIL import Image
import torch.utils.data

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset, GTImgData
from lib.dataset.tin_utils import get_next_sp_with_pose
from lib.dataset.utils import Dims, PrecomputedFilesHandler, COCO_CLASSES, filter_on_score
from lib.timer import Timer


class Labels(NamedTuple):
    obj: Union[None, torch.Tensor] = None
    act: Union[None, torch.Tensor] = None
    hoi: Union[None, torch.Tensor] = None
    pstate: Union[None, torch.Tensor] = None


class Minibatch(NamedTuple):
    ex_data: List
    im_data: List
    person_data: List
    obj_data: List
    labels: Labels = Labels()
    other: List = []


class AbstractHoiDatasetSplit:
    def __init__(self, split):
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

    def get_loader(self, *args, **kwargs) -> torch.utils.data.DataLoader:
        raise NotImplementedError


class HoiDatasetSplit(AbstractHoiDatasetSplit):
    def __init__(self, split, full_dataset: HoiDataset, object_inds=None, action_inds=None, use_precomputed_data=False, **kwargs):
        super().__init__(split)
        self.full_dataset = full_dataset  # type: HoiDataset
        self.all_gt_img_data = self.all_gt_img_data

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

        if self.split == 'train':
            if object_inds is not None:
                print(f'Train objects ({self.seen_objects.size}):', self.seen_objects.tolist())
            if action_inds is not None:
                print(f'Train actions ({self.seen_actions.size}):', self.seen_actions.tolist())
            if object_inds is not None or action_inds is not None:
                print(f'Train interactions ({self.seen_interactions.size}):', self.seen_interactions.tolist())

        if use_precomputed_data:
            self._feat_provider = self._init_feat_provider(**kwargs)  # type: FeatProvider

    def _init_feat_provider(self, **kwargs):
        raise NotImplementedError

    def get_img(self, img_idx):
        im_data = self.all_gt_img_data[img_idx]  # type: GTImgData
        img_fn = self.full_dataset.get_img_path(self.split, im_data.filename)
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
        return len(self.all_gt_img_data)

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

    def get_loader(self, *args, **kwargs):
        return self._feat_provider.get_loader(*args, **kwargs)

    def hold_out(self, ratio):
        self._feat_provider.hold_out(ratio)

    @classmethod
    def instantiate_full_dataset(cls) -> HoiDataset:
        raise NotImplementedError

    @classmethod
    def get_splits(cls, action_inds=None, object_inds=None, **kwargs):
        splits = {}
        full_dataset = cls.instantiate_full_dataset()

        train_split = cls(split='train', full_dataset=full_dataset, object_inds=object_inds, action_inds=action_inds, **kwargs)
        if cfg.val_ratio > 0:
            train_split.hold_out(ratio=cfg.val_ratio)
        splits['train'] = train_split
        splits['test'] = cls(split='test', full_dataset=full_dataset, **kwargs)

        return splits


class FeatProvider(torch.utils.data.Dataset):
    def __init__(self, ds: HoiDatasetSplit, ds_name,
                 labels_are_actions=False,
                 no_feats=False,
                 obj_mapping=None, filter_objs=True,
                 max_ppl=None, max_obj=None):
        super().__init__()
        self.wrapped_ds = ds
        self.full_dataset = self.wrapped_ds.full_dataset
        self.split = self.wrapped_ds.split
        self.keep_inds = self.holdout_inds = None  # These will be set later, if needed.

        #############################################################################################################################
        # Load precomputed data
        #############################################################################################################################
        #####################################################
        # Image
        #####################################################
        img_fn = cfg.precomputed_feats_format % (f'{ds_name}', 'resnet152', self.split)
        self.pc_img_feats = PrecomputedFilesHandler.get(img_fn, 'img_feats', load_in_memory=True)

        #####################################################
        # Objects
        #####################################################
        self.objects_fn = cfg.precomputed_feats_format % (f'{ds_name}objs', 'mask_rcnn_X_101_32x8d_FPN_3x', f'{self.split}')
        if not os.path.isfile(self.objects_fn) and no_feats:
            self.objects_fn = cfg.precomputed_feats_format % (f'{ds_name}objs', 'mask_rcnn_X_101_32x8d_FPN_3x',
                                                              f'{self.split}{"__nofeats" if no_feats else ""}')
        self.obj_boxes = PrecomputedFilesHandler.get(self.objects_fn, 'boxes', load_in_memory=True)
        self.obj_scores = PrecomputedFilesHandler.get(self.objects_fn, 'box_scores', load_in_memory=True)
        if obj_mapping is not None:
            self.obj_scores = self.obj_scores[:, obj_mapping]
        if not no_feats:
            self.obj_feats = PrecomputedFilesHandler.get(self.objects_fn, 'box_feats')
            self.obj_feats_dim = self.obj_feats.shape[-1]
            assert self.obj_feats.dtype == np.float32
        assert self.obj_boxes.dtype == self.obj_scores.dtype == np.float32

        #####################################################
        # People
        #####################################################
        self.keypoints_fn = cfg.precomputed_feats_format % (f'{ds_name}kps', 'keypoint_rcnn_R_101_FPN_3x', f'{self.split}')
        if not os.path.isfile(self.keypoints_fn) and no_feats:
            self.keypoints_fn = cfg.precomputed_feats_format % \
                                (f'{ds_name}kps', 'keypoint_rcnn_R_101_FPN_3x', f'{self.split}{"__nofeats" if no_feats else ""}')
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
        self.img_data_cache = self._compute_img_data(filter_objs, max_ppl=max_ppl, max_obj=max_obj)  # type: List[Dict]
        assert len(self.img_data_cache) == self.wrapped_ds.num_images

        #############################################################################################################################
        # Labels
        #############################################################################################################################
        self.labels_are_actions = labels_are_actions
        self.labels = self.non_empty_inds = None  # these will be initialised later

    def _filter_zs_labels(self, labels):
        if self.labels_are_actions:
            seen, num_all = self.wrapped_ds.seen_actions, self.full_dataset.num_actions
        else:  # labels are interactions
            seen, num_all = self.wrapped_ds.seen_interactions, self.full_dataset.num_interactions
        assert labels.shape[1] == num_all
        if seen.size < num_all:
            all_labels = labels
            labels = np.zeros_like(all_labels)
            labels[:, seen] = all_labels[:, seen]
        return labels

    def _compute_img_data(self, filter_bg_and_human_objs, max_ppl, max_obj):
        """
        Fields: 'fname_id': int, 'person_inds': array, 'obj_inds': array
        """
        if max_ppl is None:
            max_ppl = cfg.max_ppl
        if max_obj is None:
            max_obj = cfg.max_obj

        # Filter out BG objects and persons (already have the BBs provided by the keypoint model).
        obj_classes = np.argmax(PrecomputedFilesHandler.get(self.objects_fn, 'box_scores'), axis=1)
        obj_to_keep = (obj_classes != COCO_CLASSES.index('__background__')) & (obj_classes != COCO_CLASSES.index('person'))
        fname_id_to_obj_inds = {}
        for i, fid in enumerate(PrecomputedFilesHandler.get(self.objects_fn, 'fname_ids')):
            if filter_bg_and_human_objs and obj_to_keep[i]:
                fname_id_to_obj_inds.setdefault(fid, []).append(i)
        fname_id_to_obj_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_obj_inds.items()}

        fname_id_to_kp_inds = {}
        for i, fid in enumerate(PrecomputedFilesHandler.get(self.keypoints_fn, 'fname_ids')):
            fname_id_to_kp_inds.setdefault(fid, []).append(i)
        fname_id_to_kp_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_kp_inds.items()}

        all_img_data = []
        for i, gt_img_data in enumerate(self.wrapped_ds.all_gt_img_data):
            fname = gt_img_data.filename
            fname_id = self.full_dataset.get_fname_id(fname)
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
                    inds = filter_on_score(scores=self.person_scores[im_person_inds], min_score=cfg.min_ppl_score, maxn=max_ppl, keep_one=True)
                    assert inds.size > 0
                    im_data['person_inds'] = np.sort(im_person_inds[inds])

            # Objects
            obj_inds = fname_id_to_obj_inds.get(fname_id, None)
            if obj_inds is not None:
                im_obj_inds = np.atleast_1d(obj_inds)
                scores = np.max(self.obj_scores[im_obj_inds, :], axis=1)
                inds = filter_on_score(scores=scores, min_score=None, maxn=max_obj, keep_one=False)
                assert inds.size > 0
                im_data['obj_inds'] = np.sort(im_obj_inds[inds])

            all_img_data.append(im_data)
        return all_img_data

    def hold_out(self, ratio):
        if cfg.no_filter_bg_only or self.non_empty_inds is None:
            num_examples = len(self)
            ex_inds = np.arange(num_examples)
        else:
            num_examples = len(self.non_empty_inds)
            ex_inds = self.non_empty_inds
        num_to_keep = num_examples - int(num_examples * ratio)
        keep_inds = np.random.choice(ex_inds, size=num_to_keep, replace=False)
        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(ex_inds, keep_inds)

    def get_loader(self, batch_size, shuffle=None, drop_last=None, holdout_set=False, **kwargs):
        if self.split == 'test':
            return self._get_test_loader(batch_size=batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         **kwargs)
        else:
            if self.keep_inds is None:
                assert self.holdout_inds is None and holdout_set is False
                ds = self
            else:
                ds = torch.utils.data.Subset(self, self.holdout_inds if holdout_set else self.keep_inds)
            if shuffle is None:
                shuffle = not holdout_set
            if drop_last is None:
                drop_last = True
            data_loader = torch.utils.data.DataLoader(
                dataset=ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=lambda x: self.collate(x),
                **kwargs,
            )
        return data_loader

    def _get_test_loader(self, batch_size, **kwargs):
        raise NotImplementedError

    def __getitem__(self, idx):
        # This should only be used by the data loader (see above).
        return idx

    def __len__(self):
        raise NotImplementedError

    def collate(self, idx_list) -> Minibatch:
        Timer.get('GetBatch').tic()
        mb = self._collate(idx_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        Timer.get('GetBatch').toc(discard=5)
        return mb

    def _collate(self, idx_list, device) -> Minibatch:
        raise NotImplementedError


class ImgInstancesFeatProvider(FeatProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.split != 'test':
            try:
                labels = self.wrapped_ds.full_dataset.split_labels[self.split]  # type: np.ndarray
            except AttributeError:
                raise ValueError(f'Dataset of type {type(self.full_dataset)} not supported.')

            self.labels = self._filter_zs_labels(labels)

            non_empty_inds = np.flatnonzero(np.any(labels, axis=1))
            imgs_with_kps = [i for i, imd in enumerate(self.img_data_cache) if 'person_inds' in imd]
            self.non_empty_inds = np.intersect1d(non_empty_inds, np.array(imgs_with_kps))

    def _get_test_loader(self, batch_size, **kwargs):
        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=lambda x: self.collate(x),
            **kwargs,
        )
        return data_loader

    def __len__(self):
        return len(self.img_data_cache)

    def _collate(self, idx_list, device) -> Minibatch:
        # This assumes the filtering has been done on initialisation (i.e., there are no more people/objects than the max number)
        idxs = np.array(idx_list)

        # Example data
        ex_ids = [f'ex{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)

        # Image data
        im_ids = [self.wrapped_ds.all_gt_img_data[idx].filename for idx in idxs]
        im_feats = feats
        orig_img_wh = torch.tensor(np.stack([self.wrapped_ds.all_gt_img_data[idx].img_size for idx in idxs], axis=0),
                                   dtype=torch.float32, device=device)

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

        if self.split != 'test':
            img_labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            pstate_labels = torch.tensor(self.img_pstate_labels[idxs, :], dtype=torch.float32, device=device)
            labels = Labels(hoi=img_labels, pstate=pstate_labels)
        else:
            labels = Labels()

        mb = Minibatch(ex_data=[ex_ids, feats],
                       im_data=[im_ids, im_feats, orig_img_wh],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       labels=labels)

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
    def __init__(self, ds_name, *args, **kwargs):
        super().__init__(ds_name=ds_name, max_ppl=0, max_obj=0, *args, **kwargs)
        self.hoi_fn = os.path.join(cfg.cache_root, f'precomputed_{ds_name}_hoi_assignment_file__{self.split}.h5')
        ho_infos = PrecomputedFilesHandler.get(self.hoi_fn, 'ho_infos', load_in_memory=True)

        fname_ids_to_img_idx = {imdata['fname_id']: im_idx for im_idx, imdata in enumerate(self.img_data_cache)}
        ho_im_idxs = np.array([fname_ids_to_img_idx[fname_id] for fname_id in ho_infos[:, 0]])
        ho_infos = np.concatenate([ho_im_idxs[:, None], ho_infos], axis=1)  # [im_idx, fname_id, hum_idx, obj_idx, is_obj_human]

        # Sanity check and filtering. Indices might not be in the image data because filtering is performed that might remove some person/objects.
        all_person_inds = {idx for im_data in self.img_data_cache for idx in im_data.get('person_inds', [])}
        all_obj_inds = {idx for im_data in self.img_data_cache for idx in im_data.get('obj_inds', [])}
        valid_hois = []
        for i, (im_idx, fname_id, hum_idx, obj_idx, is_obj_human) in enumerate(ho_infos):
            im_data = self.img_data_cache[im_idx]
            assert fname_id == im_data['fname_id']

            if hum_idx in all_person_inds:
                assert hum_idx in im_data['person_inds']
            else:
                continue

            if is_obj_human:
                if obj_idx in all_person_inds:
                    assert obj_idx in im_data['person_inds']
                else:
                    continue
            else:
                if obj_idx in all_obj_inds:
                    assert obj_idx in im_data['obj_inds']
                else:
                    continue
            valid_hois.append(i)

        valid_hois = np.array(valid_hois)
        assert not np.any(np.isnan(valid_hois)) and not np.any(valid_hois < 0)
        self.ho_infos = ho_infos[valid_hois]

        if self.split != 'test':
            labels = PrecomputedFilesHandler.get(self.hoi_fn, 'action_labels', load_in_memory=True)[valid_hois]  # type: np.ndarray
            print(f'Negatives per positive: {np.sum(labels[:, 0] > 0) / np.sum(labels[:, 0] == 0)}')

            self.labels = self._filter_zs_labels(labels)
            # self.non_empty_inds = np.flatnonzero(np.any(labels, axis=1))
            self.non_empty_inds = None  # FIXME enable
            # non_empty_imgs = np.unique(self.ho_infos[:, 0])

    def __len__(self):
        return self.ho_infos.shape[0]

    def _get_test_loader(self, batch_size, **kwargs):
        num_imgs = self.wrapped_ds.num_images
        hoi_idxs_per_img_idx = {}
        for hoi_idx, im_idx in enumerate(self.ho_infos[:, 0]):
            hoi_idxs_per_img_idx.setdefault(im_idx, []).append(hoi_idx)
        assert not (set(hoi_idxs_per_img_idx.keys()) - set(range(num_imgs)))  # some images might be empty
        hoi_idxs_per_img_idx = [hoi_idxs_per_img_idx.get(i, []) for i in range(num_imgs)]

        class ImgSampler(torch.utils.data.Sampler):
            def __len__(self):
                return num_imgs

            def __iter__(self):
                return iter(hoi_idxs_per_img_idx)

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            sampler=ImgSampler(self),
            collate_fn=lambda x: None if not x[0] else self.collate(x[0]),
            **kwargs,
        )
        return data_loader

    def _collate(self, idx_list, device) -> Minibatch:
        idxs = np.array(idx_list)
        im_idxs = self.ho_infos[idxs, 0]

        # Example data
        ex_ids = [f'ex{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[im_idxs, :], dtype=torch.float32, device=device)  # FIXME

        # Image data
        im_ids = [self.wrapped_ds.all_gt_img_data[idx].filename for idx in im_idxs]
        im_feats = feats
        orig_img_wh = torch.tensor(np.stack([self.wrapped_ds.all_gt_img_data[idx].img_size for idx in im_idxs], axis=0),
                                   dtype=torch.float32, device=device)

        dims = self.wrapped_ds.dims
        assert dims.F_kp == dims.F_obj
        P, M, K_hake, D = dims.P, dims.M, dims.K_hake, dims.D
        assert P == M == 1
        N = idxs.size

        person_inds = self.ho_infos[idxs, 2].astype(np.int, copy=False)
        obj_inds = self.ho_infos[idxs, 3].astype(np.int, copy=False)
        is_obj_human = self.ho_infos[idxs, 4].astype(bool, copy=False)

        ppl_boxes = self.person_boxes[person_inds]
        ppl_feats = self.person_feats[person_inds]
        coco_kps = self.coco_kps[person_inds]
        kp_boxes = self.hake_kp_boxes[person_inds]
        kp_feats = self.hake_kp_feats[person_inds]

        obj_boxes = np.full((obj_inds.shape[0], self.obj_boxes.shape[1]), fill_value=np.nan)
        obj_scores = np.full((obj_inds.shape[0], self.obj_scores.shape[1]), fill_value=np.nan)
        obj_feats = np.full((obj_inds.shape[0], self.obj_feats.shape[1]), fill_value=np.nan)
        # Object is human
        obj_boxes[is_obj_human] = self.person_boxes[obj_inds[is_obj_human]]
        _hum_obj_scores = self.person_scores[obj_inds[is_obj_human]]
        obj_scores[is_obj_human, :] = (1 - _hum_obj_scores[:, None]) / (obj_scores.shape[1] - 1)  # equally distributed among all the other classes
        obj_scores[is_obj_human, self.wrapped_ds.full_dataset.human_class] = _hum_obj_scores
        obj_feats[is_obj_human] = self.person_feats[obj_inds[is_obj_human]]
        # Object is not human
        obj_boxes[~is_obj_human] = self.obj_boxes[obj_inds[~is_obj_human]]
        obj_scores[~is_obj_human] = self.obj_scores[obj_inds[~is_obj_human]]
        obj_feats[~is_obj_human] = self.obj_feats[obj_inds[~is_obj_human]]
        assert not np.any(np.isnan(obj_boxes))
        assert not np.any(np.isnan(obj_scores))
        assert not np.any(np.isnan(obj_feats))

        t_ppl_boxes = torch.from_numpy(ppl_boxes).unsqueeze(dim=1).to(device=device)
        t_ppl_feats = torch.from_numpy(ppl_feats).unsqueeze(dim=1).to(device=device)
        t_coco_kps = torch.from_numpy(coco_kps).unsqueeze(dim=1).to(device=device)
        t_kp_boxes = torch.from_numpy(kp_boxes).unsqueeze(dim=1).to(device=device)
        t_kp_feats = torch.from_numpy(kp_feats).unsqueeze(dim=1).to(device=device)
        t_obj_boxes = torch.from_numpy(obj_boxes).unsqueeze(dim=1).to(device=device)
        t_obj_scores = torch.from_numpy(obj_scores).unsqueeze(dim=1).to(device=device)
        t_obj_feats = torch.from_numpy(obj_feats).unsqueeze(dim=1).to(device=device)

        all_box_inds, u_idx = np.unique(np.concatenate([person_inds, obj_inds]), return_index=True)
        all_boxes = np.concatenate([ppl_boxes, obj_boxes], axis=0)[u_idx, :]
        box_ind_mapping = {idx: i for i, idx in enumerate(all_box_inds)}
        local_ho_pairs = np.stack([np.array([box_ind_mapping[idx] for idx in person_inds]),
                                   np.array([box_ind_mapping[idx] for idx in obj_inds]),
                                   ], axis=1)

        if self.split != 'test':
            act_labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
        else:
            act_labels = None

        mb = Minibatch(ex_data=[ex_ids, feats, all_boxes, local_ho_pairs],
                       im_data=[im_ids, im_feats, orig_img_wh],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       labels=Labels(act=act_labels))
        if cfg.tin:
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

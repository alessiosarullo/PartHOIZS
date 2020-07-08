import os
from typing import List, Dict, Union, NamedTuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset, GTImgData
from lib.dataset.utils import Dims, PrecomputedFilesHandler, COCO_CLASSES, filter_on_score
from lib.timer import Timer


class Labels(NamedTuple):
    obj: Union[None, torch.Tensor] = None
    act: Union[None, torch.Tensor] = None
    hoi: Union[None, torch.Tensor] = None
    pstate: Union[None, torch.Tensor] = None


class Minibatch(NamedTuple):
    # TODO Lists -> NamedTuples
    ex_data: List
    im_data: List
    person_data: List
    obj_data: List
    labels: Labels = Labels()
    epoch: Union[None, int] = None
    iter: Union[None, int] = None


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
    def __init__(self, split, full_dataset: HoiDataset, object_inds=None, action_inds=None, interaction_inds=None,
                 load_precomputed_data=False, **kwargs):
        super().__init__(split)
        assert interaction_inds is None or (object_inds is None and action_inds is None)
        self.full_dataset = full_dataset  # type: HoiDataset
        self.all_gt_img_data = self.full_dataset.get_img_data(self.split)

        object_inds = sorted(object_inds) if object_inds is not None else range(self.full_dataset.num_objects)
        self.objects = [full_dataset.objects[i] for i in object_inds]
        self.seen_objects = np.array(object_inds, dtype=np.int)

        action_inds = sorted(action_inds) if action_inds is not None else range(self.full_dataset.num_actions)
        self.actions = [full_dataset.actions[i] for i in action_inds]
        self.seen_actions = np.array(action_inds, dtype=np.int)

        if interaction_inds is None:
            seen_op_mat = self.full_dataset.oa_to_interaction[self.seen_objects, :][:, self.seen_actions]
            seen_interactions = set(np.unique(seen_op_mat).tolist()) - {-1}
        else:
            assert isinstance(interaction_inds, list)
            seen_interactions = set(interaction_inds)
        self.seen_interactions = np.array(sorted(seen_interactions), dtype=np.int)
        self.interactions = self.full_dataset.interactions[self.seen_interactions, :]  # original action and object inds

        if self.split == 'train':
            if object_inds is not None:
                print(f'Train objects ({self.seen_objects.size}):', self.seen_objects.tolist())
            if action_inds is not None:
                print(f'Train actions ({self.seen_actions.size}):', self.seen_actions.tolist())
            if object_inds is not None or action_inds is not None:
                print(f'Train interactions ({self.seen_interactions.size}):', self.seen_interactions.tolist())

        if load_precomputed_data:
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
        K = 17  # FIXME magic constant
        O, A, C = self.full_dataset.num_objects, self.full_dataset.num_actions, self.full_dataset.num_interactions
        if cfg.tin:
            D = cfg.ipsize
        else:
            D = None
        return Dims(N=None, P=P, M=M, K=K, O=O, A=A, C=C, D=D)

    def get_loader(self, *args, **kwargs):
        return self._feat_provider.get_loader(*args, **kwargs)

    def hold_out(self, ratio):
        self._feat_provider.hold_out(ratio)

    @classmethod
    def instantiate_full_dataset(cls, **kwargs) -> HoiDataset:
        raise NotImplementedError

    @classmethod
    def get_splits(cls, object_inds=None, action_inds=None, interaction_inds=None, val_ratio=0, **kwargs):
        splits = {}
        full_dataset = cls.instantiate_full_dataset()

        train_split = cls(split='train', full_dataset=full_dataset,
                          object_inds=object_inds, action_inds=action_inds, interaction_inds=interaction_inds,
                          **kwargs)
        if val_ratio > 0:
            train_split.hold_out(ratio=val_ratio)
        splits['train'] = train_split
        splits['test'] = cls(split='test', full_dataset=full_dataset, **kwargs)

        return splits


class FeatProvider(torch.utils.data.Dataset):
    def __init__(self, ds: HoiDatasetSplit, ds_name,
                 obj_mapping=None, filter_objs=True,
                 label_mapping=None,
                 max_ppl=None, max_obj=None):
        super().__init__()
        self.wrapped_ds = ds
        self.full_dataset = self.wrapped_ds.full_dataset
        self.split = self.wrapped_ds.split
        self.keep_inds = self.holdout_inds = None  # These will be set later, if needed.
        self.holdout_ratio = None  # This will be set later, if needed.

        #############################################################################################################################
        # Load precomputed data
        #############################################################################################################################
        no_feats = False  # FIXME

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
            assert self.hake_kp_boxes.shape[1] == self.full_dataset.num_parts
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

        #####################################################
        # Whole examples
        #####################################################
        self.ex_feat_dim = self.kp_net_dim + self.obj_feats_dim

        #############################################################################################################################
        # Cache variables to speed up loading
        #############################################################################################################################
        self.img_data_cache = self._compute_img_data(filter_objs, max_ppl=max_ppl, max_obj=max_obj)  # type: List[Dict]
        self.obj_scores_per_img = np.stack([np.max(self.obj_scores[imd['obj_inds'], :], axis=0)
                                            if 'obj_inds' in imd else np.zeros(self.full_dataset.num_objects)
                                            for imd in self.img_data_cache],
                                           axis=0)
        assert len(self.img_data_cache) == self.wrapped_ds.num_images
        assert self.obj_scores_per_img.shape == (self.wrapped_ds.num_images, self.full_dataset.num_objects)

        #############################################################################################################################
        # Labels
        #############################################################################################################################
        self.labels = self.non_empty_inds = None  # these will be initialised later
        self.label_mapping = label_mapping

    def _filter_zs_labels(self, labels):
        if self.full_dataset.labels_are_actions:
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
        assert self.holdout_ratio is None
        self.holdout_ratio = ratio

        if cfg.include_bg_only:
            num_examples = len(self)
            ex_inds = np.arange(num_examples)
        else:
            num_examples = len(self.non_empty_inds)
            ex_inds = self.non_empty_inds
        num_to_keep = num_examples - int(num_examples * self.holdout_ratio)
        keep_inds = np.random.choice(ex_inds, size=num_to_keep, replace=False)
        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(ex_inds, keep_inds)

    def get_loader(self, batch_size, evaluation=None, shuffle=None, drop_last=None, holdout_set=False, **kwargs):
        if evaluation is None:
            evaluation = (self.split == 'test')

        if evaluation:
            assert not holdout_set
            data_loader = self._get_eval_loader(ds=self,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                **kwargs)
        else:
            if shuffle is None:
                shuffle = not holdout_set
            if drop_last is None:
                drop_last = True
            data_loader = self._get_train_loader(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, holdout_set=holdout_set, **kwargs)
        return data_loader

    def _get_train_loader(self, batch_size, shuffle, drop_last, holdout_set, **kwargs):
        if self.keep_inds is None:
            assert self.holdout_inds is None and holdout_set is False
            ds = self
        else:
            ds = torch.utils.data.Subset(self, indices=self.holdout_inds if holdout_set else self.keep_inds)

        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: self.collate(x),
            **kwargs,
        )
        return data_loader

    def _get_eval_loader(self, ds, batch_size, **kwargs):
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


class HoiInstancesFeatProvider(FeatProvider):
    def __init__(self, ds_name, *args, **kwargs):
        if cfg.include_bg_only:
            raise NotImplementedError('Background-only images would only increase the number of possible negatives to sample from.')
        super().__init__(ds_name=ds_name, max_ppl=0, max_obj=0, *args, **kwargs)
        self.hoi_fn = os.path.join(cfg.cache_root, f'precomputed_{ds_name}__hoi_pairs_{self.split}.h5')
        ho_infos = PrecomputedFilesHandler.get(self.hoi_fn, 'ho_infos', load_in_memory=True)

        fname_ids_to_img_idx = {imdata['fname_id']: im_idx for im_idx, imdata in enumerate(self.img_data_cache)}
        ho_im_idxs = np.array([fname_ids_to_img_idx[fname_id] for fname_id in ho_infos[:, 0]])
        ho_infos = np.concatenate([ho_im_idxs[:, None], ho_infos], axis=1)  # [im_idx, fname_id, hum_idx, obj_idx, is_obj_human]
        assert ho_infos.shape[1] == 5

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
        self.ho_infos = ho_infos[valid_hois]  # see above for format

        if self.split != 'test':
            _labels = PrecomputedFilesHandler.get(self.hoi_fn, 'labels', load_in_memory=True)[valid_hois]  # type: np.ndarray
            if self.label_mapping is not None:
                assert self.label_mapping.size == _labels.shape[1]
                num_all = self.full_dataset.num_actions if self.full_dataset.labels_are_actions else self.full_dataset.num_interactions
                _mapped_labels = np.zeros((_labels.shape[0], num_all))
                _mapped_labels[:, self.label_mapping] = _labels
                _labels = _mapped_labels
            assert self.ho_infos.shape[0] == _labels.shape[0]
            if self.full_dataset.labels_are_actions:
                self.hoi_obj_labels = PrecomputedFilesHandler.get(self.hoi_fn, 'obj_labels', load_in_memory=True)[valid_hois]  # type: np.ndarray
            positives, negatives = self._get_pos_neg_mask(_labels)
            assert np.all(positives ^ negatives)
            print(f'Negatives per positive (before ZS): {np.sum(negatives) / np.sum(positives)}.')
            self.labels = _labels

            try:
                self.pstate_labels = PrecomputedFilesHandler.get(self.hoi_fn, 'pstate_labels', load_in_memory=True)[valid_hois]  # type: np.ndarray
            except KeyError:
                self.pstate_labels = None

            self.labels = self._filter_zs_labels(self.labels)
            self.non_empty_inds = np.flatnonzero(np.any(self.labels, axis=1))
            # non_empty_imgs = np.unique(self.ho_infos[:, 0])

            # Compute masks for positives and negatives. This has to be done again to take into account possibly removed ZS labels.
            # Note: negatives = background, as opposed to not being labelled at all.
            positives, negatives = self._get_pos_neg_mask(self.labels)
            assert np.all((positives ^ negatives) | (~self.labels.any(axis=1)))  # Sanity check: either positive, negative or unlabelled
            assert np.all((positives ^ negatives)[self.non_empty_inds])  # Sanity check: if labelled, either positive or negative
            print(f'Negatives per positive (after ZS): {np.sum(negatives) / np.sum(positives)}.')
            self.positives_mask = positives
            self.negatives_mask = negatives

    def _get_pos_neg_mask(self, labels):
        if self.full_dataset.labels_are_actions:
            positives = np.any(labels[:, 1:] > 0, axis=1)
            negatives = (labels[:, 0] > 0)
        else:
            null_interactions = (self.full_dataset.interactions[:, 0] == 0)
            positives = np.any(labels[:, ~null_interactions] > 0, axis=1)
            negatives = np.any(labels[:, null_interactions] > 0, axis=1)
        return positives, negatives

    def __len__(self):
        return self.ho_infos.shape[0]

    def hold_out(self, ratio):
        assert self.holdout_ratio is None
        self.holdout_ratio = ratio

        num_positives = np.sum(self.positives_mask)
        pos_inds = np.flatnonzero(self.positives_mask)
        pos_to_keep = num_positives - int(num_positives * self.holdout_ratio)
        pos_keep_inds = np.random.choice(pos_inds, size=pos_to_keep, replace=False)

        num_negatives = np.sum(self.negatives_mask)
        neg_inds = np.flatnonzero(self.negatives_mask)
        neg_to_keep = num_negatives - int(num_negatives * self.holdout_ratio)
        neg_keep_inds = np.random.choice(neg_inds, size=neg_to_keep, replace=False)

        keep_inds = np.union1d(pos_keep_inds, neg_keep_inds)
        assert np.intersect1d(pos_keep_inds, neg_keep_inds).size == 0  # either positive or negative
        assert np.intersect1d(keep_inds, self.non_empty_inds).size == keep_inds.size  # no unlabelled examples

        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(self.non_empty_inds, keep_inds)

    def _get_train_loader(self, batch_size, shuffle, drop_last, holdout_set, **kwargs):
        pos_samples = np.flatnonzero(self.positives_mask)
        neg_samples = np.flatnonzero(self.negatives_mask)
        if self.keep_inds is None:
            assert self.holdout_inds is None and holdout_set is False
        else:
            split_inds = self.holdout_inds if holdout_set else self.keep_inds
            pos_samples = np.intersect1d(pos_samples, split_inds)
            neg_samples = np.intersect1d(neg_samples, split_inds)

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=BalancedTripletSampler(self,
                                                 batch_size=batch_size,
                                                 drop_last=drop_last,
                                                 shuffle=shuffle,
                                                 pos_samples=pos_samples,
                                                 neg_samples=neg_samples,
                                                 ),
            collate_fn=lambda x: self.collate(x),
            **kwargs,
        )
        return data_loader

    def _get_eval_loader(self, ds, batch_size, **kwargs):
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

        def _collate(x):
            x = [y.tolist() if isinstance(y, np.ndarray) else y for y in x]
            return None if not x[0] else self.collate(x[0])

        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=ImgSampler(ds),
            collate_fn=_collate,
            **kwargs,
        )
        return data_loader

    def _collate(self, idx_list, device) -> Minibatch:
        idxs = np.array(idx_list)
        im_idxs = self.ho_infos[idxs, 0]

        # Image data
        im_ids = [self.wrapped_ds.all_gt_img_data[idx].filename for idx in im_idxs]
        im_feats = torch.tensor(self.pc_img_feats[im_idxs, :], dtype=torch.float32, device=device)
        orig_img_wh = torch.tensor(np.stack([self.wrapped_ds.all_gt_img_data[idx].img_size for idx in im_idxs], axis=0),
                                   dtype=torch.float32, device=device)
        im_obj_scores = torch.tensor(self.obj_scores_per_img[im_idxs, :], dtype=torch.float32, device=device)

        dims = self.wrapped_ds.dims
        assert dims.F_kp == dims.F_obj
        P, M, B, D, O = dims.P, dims.M, dims.B, dims.D, dims.O
        assert P == M == 1
        N = idxs.size

        person_inds = self.ho_infos[idxs, 2].astype(np.int, copy=False)
        obj_inds = self.ho_infos[idxs, 3].astype(np.int, copy=False)
        is_obj_human = self.ho_infos[idxs, 4].astype(bool, copy=False)

        # Masks instead of inds in case H5 file are not loaded in memory.
        person_mask = np.zeros(self.person_boxes.shape[0], dtype=bool)
        person_mask[person_inds] = True
        _, p_inv_index = np.unique(person_inds, return_inverse=True)

        hum_obj_inds = obj_inds[is_obj_human]
        hum_object_mask = np.zeros(self.person_boxes.shape[0], dtype=bool)
        hum_object_mask[hum_obj_inds] = True
        _, hobj_inv_index = np.unique(hum_obj_inds, return_inverse=True)

        obj_obj_inds = obj_inds[~is_obj_human]
        obj_object_mask = np.zeros(self.obj_boxes.shape[0], dtype=bool)
        obj_object_mask[obj_obj_inds] = True
        _, oobj_inv_index = np.unique(obj_obj_inds, return_inverse=True)

        ppl_boxes = self.person_boxes[person_inds]
        ppl_feats = self.person_feats[person_mask, ...][p_inv_index, ...]
        coco_kps = self.coco_kps[person_inds]
        kp_boxes = self.hake_kp_boxes[person_inds]
        kp_feats = self.hake_kp_feats[person_mask, ...][p_inv_index, ...]

        obj_boxes = np.full((obj_inds.shape[0], self.obj_boxes.shape[1]), fill_value=np.nan)
        obj_scores = np.full((obj_inds.shape[0], self.obj_scores.shape[1]), fill_value=np.nan)
        obj_feats = np.full((obj_inds.shape[0], self.obj_feats.shape[1]), fill_value=np.nan)
        # Object is human
        obj_boxes[is_obj_human] = self.person_boxes[hum_obj_inds]
        _hum_obj_scores = self.person_scores[hum_obj_inds]
        obj_scores[is_obj_human, :] = (1 - _hum_obj_scores[:, None]) / (O - 1)  # equally distributed among all the other classes
        obj_scores[is_obj_human, self.full_dataset.human_class] = _hum_obj_scores
        obj_feats[is_obj_human] = self.person_feats[hum_object_mask, ...][hobj_inv_index, ...]
        # Object is not human
        obj_boxes[~is_obj_human] = self.obj_boxes[obj_obj_inds]
        obj_scores[~is_obj_human] = self.obj_scores[obj_obj_inds]
        obj_feats[~is_obj_human] = self.obj_feats[obj_object_mask, ...][oobj_inv_index, ...]
        assert not np.any(np.isnan(obj_boxes))
        assert not np.any(np.isnan(obj_scores))
        assert not np.any(np.isnan(obj_feats))

        t_ppl_boxes = torch.from_numpy(ppl_boxes).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_ppl_feats = torch.from_numpy(ppl_feats).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_coco_kps = torch.from_numpy(coco_kps).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_kp_boxes = torch.from_numpy(kp_boxes).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_kp_feats = torch.from_numpy(kp_feats).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_obj_boxes = torch.from_numpy(obj_boxes).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_obj_scores = torch.from_numpy(obj_scores).unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        t_obj_feats = torch.from_numpy(obj_feats).unsqueeze(dim=1).to(device=device, dtype=torch.float32)

        all_box_inds, u_idx = np.unique(np.concatenate([person_inds, obj_inds]), return_index=True)
        all_boxes = np.concatenate([ppl_boxes, obj_boxes], axis=0)[u_idx, :]
        ppl_scores = np.full((ppl_boxes.shape[0], O), fill_value=1 / (O - 1))
        ppl_scores[:, self.full_dataset.human_class] = self.person_scores[person_inds]
        all_box_scores = np.concatenate([ppl_scores, obj_scores], axis=0)[u_idx, :]  # TODO check
        box_ind_mapping = {idx: i for i, idx in enumerate(all_box_inds)}
        local_ho_pairs = np.stack([np.array([box_ind_mapping[idx] for idx in person_inds]),
                                   np.array([box_ind_mapping[idx] for idx in obj_inds]),
                                   ], axis=1)

        # Example data
        ex_ids = [f'{self.split}_ex{idx}' for idx in idxs]
        ex_feats = torch.cat([t_ppl_feats.squeeze(dim=1), t_obj_feats.squeeze(dim=1)], dim=1)

        labels = obj_labels_onehot = pstate_labels = None
        if self.split != 'test':
            labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            if self.full_dataset.labels_are_actions:
                obj_labels = self.hoi_obj_labels[idxs]
                assert np.all(obj_labels >= 0)
                obj_labels_onehot = np.zeros((N, self.full_dataset.num_objects))
                obj_labels_onehot[np.arange(N), obj_labels.astype(np.int)] = 1
                obj_labels_onehot = torch.tensor(obj_labels_onehot, dtype=torch.float32, device=device)
            if self.pstate_labels is not None:
                pstate_labels = torch.tensor(self.pstate_labels[idxs, :], dtype=torch.float32, device=device)
        if self.full_dataset.labels_are_actions:
            all_labels = Labels(obj=obj_labels_onehot, act=labels, pstate=pstate_labels)
        else:
            all_labels = Labels(hoi=labels, pstate=pstate_labels)

        mb = Minibatch(ex_data=[ex_ids, ex_feats, local_ho_pairs, all_boxes, all_box_scores],
                       im_data=[im_ids, im_feats, orig_img_wh, im_obj_scores],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       labels=all_labels)
        if cfg.tin:
            from lib.dataset.tin_utils import get_next_sp_with_pose
            interactiveness_patterns = np.zeros((N, P, M, D, D, 3 + B), dtype=np.float32)
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


class BalancedTripletSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: HoiInstancesFeatProvider, batch_size, drop_last, shuffle, pos_samples, neg_samples):
        super().__init__(dataset)
        if not drop_last:
            raise NotImplementedError()
        assert dataset.split == 'train'

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset = dataset

        self.pos_samples = pos_samples  # type: np.ndarray
        self.neg_samples = neg_samples  # type: np.ndarray

        self.neg_pos_ratio = cfg.hoi_bg_ratio
        pos_per_batch = batch_size / (self.neg_pos_ratio + 1)
        self.pos_per_batch = int(pos_per_batch)
        self.neg_per_batch = batch_size - self.pos_per_batch
        assert pos_per_batch == self.pos_per_batch
        assert self.neg_pos_ratio == int(self.neg_pos_ratio)
        assert self.neg_samples.shape[0] >= self.neg_pos_ratio * self.pos_samples.shape[0]

        self.batches = self.get_all_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.batches = self.get_all_batches()

    def __len__(self):
        return len(self.batches)

    def get_all_batches(self):
        batches = []

        ####################
        # Positive samples #
        ####################
        pos_samples = np.random.permutation(self.pos_samples) if self.shuffle else self.pos_samples
        batch = []
        for sample in pos_samples:
            batch.append(sample)
            if len(batch) >= self.pos_per_batch:
                assert len(batch) == self.pos_per_batch
                batches.append(batch)
                batch = []

        ####################
        # Negative samples #
        ####################

        # Repeat if there are not enough negative samples. Note: this case is actually not supported anymore.
        neg_samples = []
        for n in range(int(np.ceil(self.neg_pos_ratio * self.pos_samples.shape[0] / self.neg_samples.shape[0]))):
            ns = np.random.permutation(self.neg_samples) if self.shuffle else self.neg_samples
            neg_samples.append(ns)
        neg_samples = np.concatenate(neg_samples, axis=0)

        batch_idx = 0
        for sample in neg_samples:
            if batch_idx == len(batches):
                break
            batch = batches[batch_idx]
            batch.append(sample)
            if len(batch) >= self.batch_size:
                assert len(batch) == self.batch_size
                batch_idx += 1
        assert batch_idx == len(batches)

        # Check
        for i, batch in enumerate(batches):
            assert len(batch) == self.batch_size, (i, len(batch), len(batches))

        return batches

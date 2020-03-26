from typing import List

import numpy as np
import torch

from config import cfg
from lib.dataset.hoi_dataset_split import HoiInstancesFeatProvider
from lib.dataset.hico_hake import HicoHake, HicoHakeSplit
from lib.dataset.tin_utils import get_next_sp_with_pose
from lib.dataset.utils import Splits, HoiData, Minibatch, Dims
from lib.timer import Timer
import torch.utils.data
from lib.bbox_utils import compute_ious


def hoi_gt_assignments(ex, boxes_ext, box_labels, neg_ratio=3, gt_iou_thr=0.5):
    raise NotImplementedError  # TODO

    gt_box_classes = ex.gt_obj_classes
    gt_resc_boxes = ex.gt_boxes * ex.scale
    gt_interactions = ex.gt_hois
    predict_boxes = boxes_ext[:, 1:5]
    predict_box_labels = box_labels

    num_predict_boxes = predict_boxes.shape[0]

    # Find rel distribution
    iou_predict_to_gt_i = compute_ious(predict_boxes, gt_resc_boxes)
    predict_gt_match_i = (predict_box_labels[:, None] == gt_box_classes[None, :]) & (iou_predict_to_gt_i >= gt_iou_thr)

    action_labels_i = np.zeros((num_predict_boxes, num_predict_boxes, self.dataset.num_actions))
    for head_gt_ind, rel_id, tail_gt_ind in gt_interactions:
        for head_predict_ind in np.flatnonzero(predict_gt_match_i[:, head_gt_ind]):
            for tail_predict_ind in np.flatnonzero(predict_gt_match_i[:, tail_gt_ind]):
                if head_predict_ind != tail_predict_ind:
                    action_labels_i[head_predict_ind, tail_predict_ind, rel_id] = 1.0

    if cfg.null_as_bg:  # treat null action as negative/background
        ho_fg_mask = action_labels_i[:, :, 1:].any(axis=2)
        assert not np.any(action_labels_i[:, :, 0].astype(bool) & ho_fg_mask)  # it's either foreground or background
        ho_bg_mask = ~ho_fg_mask
        action_labels_i[:, :, 0] = ho_bg_mask.astype(np.float)
    else:  # null action is separate from background
        ho_fg_mask = action_labels_i.any(axis=2)
        ho_bg_mask = ~ho_fg_mask

    # Filter irrelevant BG relationships (i.e., those where the subject is not human or self-relations).
    non_human_box_inds_i = (predict_box_labels != self.dataset.human_class)
    ho_bg_mask[non_human_box_inds_i, :] = 0
    ho_bg_mask[np.arange(num_predict_boxes), np.arange(num_predict_boxes)] = 0

    ho_fg_pairs_i = np.stack(np.where(ho_fg_mask), axis=1)
    ho_bg_pairs_i = np.stack(np.where(ho_bg_mask), axis=1)
    num_bg_to_sample = max(ho_fg_pairs_i.shape[0], 1) * neg_ratio
    bg_inds = np.random.permutation(ho_bg_pairs_i.shape[0])[:num_bg_to_sample]
    ho_bg_pairs_i = ho_bg_pairs_i[bg_inds, :]

    ho_pairs_i = np.concatenate([ho_fg_pairs_i, ho_bg_pairs_i], axis=0)
    ho_infos_i = np.stack([np.full(ho_pairs_i.shape[0], fill_value=0),
                           ho_pairs_i[:, 0],
                           ho_pairs_i[:, 1]], axis=1)
    if ho_infos_i.shape[0] == 0:  # since GT boxes are added to predicted ones during training this cannot be empty
        print(gt_resc_boxes)
        print(predict_boxes)
        print(gt_box_classes)
        print(predict_box_labels)
        raise RuntimeError

    ho_infos_and_action_labels = np.concatenate([ho_infos_i, action_labels_i[ho_pairs_i[:, 0], ho_pairs_i[:, 1]]], axis=1)
    ho_infos = ho_infos_and_action_labels[:, :3].astype(np.int, copy=False)  # [im_id, sub_ind, obj_ind]
    action_labels = ho_infos_and_action_labels[:, 3:].astype(np.float32, copy=False)  # [pred]
    return ho_infos, action_labels


class HicoDetHakeSplit(HicoHakeSplit):
    def __init__(self,split, full_dataset, object_inds=None, action_inds=None, no_feats=False):
        super().__init__(split, full_dataset, object_inds, action_inds)
        self._feat_provider = HoiInstancesFeatProvider(ds=self, no_feats=no_feats)
        # self.non_empty_inds = np.intersect1d(self.non_empty_inds, self._feat_provider.non_empty_imgs)  # TODO

    def _get_labels(self):
        return self.full_dataset._split_det_data[self.split].labels

    @classmethod
    def instantiate_full_dataset(cls) -> HicoHake:
        return HicoHake(hicodet=True)

    @property
    def dims(self) -> Dims:
        F_kp, F_obj = self._feat_provider.kp_net_dim, self._feat_provider.obj_feats_dim
        return super().dims._replace(P=1, M=1, F_kp=F_kp, F_obj=F_obj)  # each example is an interaction, so 1 person and 1 object

    def hold_out(self, ratio):
        if not cfg.no_filter_bg_only:
            print('!!!!!!!!!! Not filtering background-only images.')
        num_examples = len(self._feat_provider.hoi_data_cache)
        example_ids = np.arange(num_examples)
        num_examples_to_keep = num_examples - int(num_examples * ratio)
        keep_inds = np.random.choice(example_ids, size=num_examples_to_keep, replace=False)
        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(example_ids, keep_inds)

    # TODO
    # def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=None, **kwargs):
    #     if shuffle is None:
    #         shuffle = True if self.split == Splits.TRAIN else False
    #     if drop_last is None:
    #         drop_last = False if self.split == Splits.TEST else True
    #     batch_size = batch_size * num_gpus
    #
    #     if self.split == Splits.TEST:
    #         ds = self
    #     else:
    #         if cfg.no_filter_bg_only:
    #             if self.split_inds is None:
    #                 assert self.split != Splits.VAL
    #                 ds = self
    #             else:
    #                 ds = Subset(self, self.split_inds)
    #         else:
    #             ds = Subset(self, self.non_empty_split_examples)
    #
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset=ds,
    #         batch_sampler=BalancedTripletMLSampler(self, batch_size, drop_last, shuffle),
    #         num_workers=num_workers,
    #         collate_fn=lambda x: self._collate(x, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
    #         # pin_memory=True,  # disable this in case of freezes
    #         **kwargs,
    #     )
    #     return data_loader

    def __len__(self):
        return len(self._feat_provider.hoi_data_cache)

    def _collate(self, idx_list, device):  # FIXME
        Timer.get('GetBatch').tic()

        idxs = np.array(idx_list)
        im_idxs = self._feat_provider.hoi_data_cache_np[idxs, 0]
        mb = self._feat_provider.collate(idx_list, device)
        if self.split != Splits.TEST:
            img_labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            pstate_labels = torch.tensor(self.img_pstate_labels[im_idxs, :], dtype=torch.float32, device=device)
        else:
            img_labels = pstate_labels = None
        mb = mb._replace(ex_labels=img_labels, pstate_labels=pstate_labels)

        Timer.get('GetBatch').toc(discard=5)
        return mb


class BalancedTripletMLSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: HicoDetHakeSplit, hoi_batch_size, drop_last, shuffle):
        super().__init__(dataset)
        if not drop_last:
            raise NotImplementedError()
        raise NotImplementedError()  # FIXME

        self.batch_size = hoi_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset = dataset

        act_labels = dataset.pc_action_labels

        if cfg.null_as_bg:
            pos_hois_mask = np.any(act_labels[:, 1:], axis=1)
            neg_hois_mask = (act_labels[:, 0] > 0)
        else:
            pos_hois_mask = np.any(act_labels, axis=1)
            neg_hois_mask = np.all(act_labels == 0, axis=1)
        assert np.all(pos_hois_mask ^ neg_hois_mask)

        pc_ho_im_ids = dataset.pc_image_ids[dataset.pc_ho_im_idxs]
        split_ids_mask = np.zeros(max(np.max(pc_ho_im_ids), np.max(dataset.image_ids)) + 1, dtype=bool)
        split_ids_mask[dataset.image_ids] = True
        split_mask = split_ids_mask[pc_ho_im_ids]

        pos_hois_mask = pos_hois_mask & split_mask
        self.pos_samples = np.flatnonzero(pos_hois_mask)

        neg_hois_mask = neg_hois_mask & split_mask
        self.neg_samples = np.flatnonzero(neg_hois_mask)

        self.neg_pos_ratio = cfg.hoi_bg_ratio
        pos_per_batch = hoi_batch_size / (self.neg_pos_ratio + 1)
        self.pos_per_batch = int(pos_per_batch)
        self.neg_per_batch = hoi_batch_size - self.pos_per_batch
        assert pos_per_batch == self.pos_per_batch
        assert self.neg_pos_ratio == int(self.neg_pos_ratio)

        self.batches = self.get_all_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.batches = self.get_all_batches()

    def __len__(self):
        return len(self.batches)

    def get_all_batches(self):
        batches = []

        # Positive samples
        pos_samples = np.random.permutation(self.pos_samples) if self.shuffle else self.pos_samples
        batch = []
        for sample in pos_samples:
            batch.append(sample)
            if len(batch) >= self.pos_per_batch:
                assert len(batch) == self.pos_per_batch
                batches.append(batch)
                batch = []

        # Negative samples
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

from typing import List

import numpy as np
import torch

from config import cfg
from lib.dataset.hico_hake import HicoHakeKPSplit, Minibatch, Dims
from lib.dataset.tin_utils import get_next_sp_with_pose
from lib.dataset.utils import Splits, HoiData
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


class HicoDetHakeKPSplit(HicoHakeKPSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cache_tin:
            raise NotImplementedError
        self.hoi_data_cache = self._compute_hoi_data()  # type: List[HoiData]
        self.hoi_data_cache_np = np.stack([np.concatenate([x[:-1], x[-1]]) for x in self.hoi_data_cache])  # type: np.ndarray[np.int]

        self.hoi_labels = self.hoi_data_cache_np[:, 4:]
        if self.seen_interactions.size < self.full_dataset.num_interactions:
            all_labels = self.hoi_labels
            self.hoi_labels = np.zeros_like(all_labels)
            self.hoi_labels[:, self.seen_interactions] = all_labels[:, self.seen_interactions]
        self.non_empty_split_inds = np.flatnonzero(np.any(self.hoi_labels, axis=1))

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
                #TODO
                for p in imdata.get('person_inds', []):
                    for o in imdata.get('obj_inds', []):
                        hoi_data = HoiData(im_idx=im_idx,
                                           fname_id=imdata['fname_id'],
                                           p_idx=p,
                                           o_idx=o,
                                           hoi_classes=self.img_labels[im_idx, :],  # FIXME
                                           )
                        all_hoi_data.append(hoi_data)
        return all_hoi_data

    @property
    def dims(self) -> Dims:
        return super().dims._replace(P=1, M=1)  # each example is an interaction, so 1 person and 1 object

    def hold_out(self, ratio):
        if not cfg.no_filter_bg_only:
            print('!!!!!!!!!! Not filtering background-only images.')
        num_examples = len(self.hoi_data_cache)
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
        return len(self.hoi_data_cache)

    def _collate(self, idx_list, device):
        # This assumes the filtering has been done on initialisation (i.e., there are no more people/objects than the max number)
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)
        im_idxs = self.hoi_data_cache_np[idxs, 0]

        # Example data
        ex_ids = [f'ex{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)  # FIXME

        # Image data
        im_ids = [f'{self.split.value}_{idx}' for idx in idxs]
        im_feats = feats
        orig_img_wh = torch.tensor(self.img_dims[idxs, :], dtype=torch.float32, device=device)

        dims = self.dims
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

        if self.split != Splits.TEST:
            labels = torch.tensor(self.hoi_labels[idxs, :], dtype=torch.float32, device=device)
            part_labels = torch.tensor(self.img_pstate_labels[im_idxs, :], dtype=torch.float32, device=device)
        else:
            labels = part_labels = None

        mb = Minibatch(ex_data=[ex_ids, feats],
                       im_data=[im_ids, im_feats, orig_img_wh],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       ex_labels=labels,
                       pstate_labels=part_labels,
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
        Timer.get('GetBatch').toc(discard=5)
        return mb


class BalancedTripletMLSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: HicoDetHakeKPSplit, hoi_batch_size, drop_last, shuffle):
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

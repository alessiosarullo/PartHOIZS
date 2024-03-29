import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hoi_dataset import GTImgData, HoiDataset


class HumObjPairsModule:
    def __init__(self, dataset: HoiDataset, gt_iou_thr=0.5, hoi_bg_ratio=3, null_as_bg=False):
        super().__init__()
        self.full_dataset = dataset  # type: HoiDataset
        self.gt_iou_thr = gt_iou_thr
        self.hoi_bg_ratio = hoi_bg_ratio
        self.null_as_bg = null_as_bg

    def compute_pairs(self, gt_img_data: GTImgData, boxes, is_human):
        # Return: `ho_infos` is an R x 2 NumPy array of [subject index, object index].
        _ho_pairs, labels, obj_labels, pstate_labels = self.process_boxes(gt_img_data, boxes, is_human)
        if _ho_pairs.shape[0] > 0:
            ho_pairs = _ho_pairs.astype(np.int, copy=False)  # [sub_ind, obj_ind]
        else:
            ho_pairs = labels = obj_labels = pstate_labels = None
        return ho_pairs, labels, obj_labels, pstate_labels

    def process_boxes(self, gt_img_data: GTImgData, boxes, is_human):
        pred_gt_box_ious = compute_ious(boxes, gt_img_data.boxes.astype(np.float, copy=False))

        box_labels = self.box_gt_assignment(gt_img_data, pred_gt_box_ious, is_human)
        assert box_labels.shape[0] == boxes.shape[0]

        # Note: labels are multi-label
        ho_pairs, labels, obj_labels, pstate_labels = self.hoi_gt_assignments(gt_img_data, boxes, box_labels, pred_gt_box_ious)
        assert ho_pairs.shape[0] == labels.shape[0] == obj_labels.shape[0]
        assert pstate_labels is None or ho_pairs.shape[0] == pstate_labels.shape[0]
        return ho_pairs, labels, obj_labels, pstate_labels

    def box_gt_assignment(self, gt_img_data: GTImgData, pred_gt_box_ious, pred_is_human):
        gt_labels = gt_img_data.box_classes
        gt_is_human = (gt_labels == self.full_dataset.human_class)
        box_labels = np.full(pred_gt_box_ious.shape[0], fill_value=np.nan)

        # If a predicted human overlaps with a GT human assign human class, otherwise assign -1
        pred_gt_box_ious_hum = pred_gt_box_ious[pred_is_human, :][:, gt_is_human]  # IoUs of predicted humans with GT humans
        overlaps_with_gt_hum = np.any(pred_gt_box_ious_hum >= self.gt_iou_thr, axis=1)
        label_assignment_hum = np.full(pred_gt_box_ious_hum.shape[0], fill_value=-1, dtype=np.int)
        label_assignment_hum[overlaps_with_gt_hum] = self.full_dataset.human_class  # only assign if the IoU is high enough
        box_labels[pred_is_human] = label_assignment_hum

        pred_gt_box_ious_obj = pred_gt_box_ious[~pred_is_human, :].copy()
        pred_gt_box_ious_obj[:, gt_is_human] = -np.inf  # prevents assignment to humans
        pred_gt_best_match_obj = np.argmax(pred_gt_box_ious_obj, axis=1)  # type: np.ndarray
        overlaps_with_gt_obj = np.any(pred_gt_box_ious_obj >= self.gt_iou_thr, axis=1)
        label_assignment_obj = np.full(pred_gt_box_ious_obj.shape[0], fill_value=-1, dtype=np.int)
        label_assignment_obj[overlaps_with_gt_obj] = gt_labels[pred_gt_best_match_obj[overlaps_with_gt_obj]]  # assigns the best match if IoU >= thr
        box_labels[~pred_is_human] = label_assignment_obj

        assert not np.any(np.isnan(box_labels))
        return box_labels

    def hoi_gt_assignments(self, gt_img_data: GTImgData, predicted_boxes, gt_label_of_predicted_boxes, pred_gt_box_ious):
        bg_ratio = self.hoi_bg_ratio
        num_predict_boxes = predicted_boxes.shape[0]

        gt_box_classes = gt_img_data.box_classes
        gt_labels = gt_img_data.labels
        if not self.full_dataset.labels_are_actions:  # convert to action labels
            gt_labels = self.full_dataset.interactions[gt_labels, 0]
        gt_ho_pairs = gt_img_data.ho_pairs

        keep = (gt_ho_pairs[:, 1] >= 0)  # filter actions with no objects such as 'walk' or 'smile'
        gt_ho_pairs = gt_ho_pairs[keep, :].astype(np.int)
        gt_labels = gt_labels[keep].astype(np.int)

        # Find rel distribution
        predict_gt_match = (gt_label_of_predicted_boxes[:, None] == gt_box_classes[None, :]) & (pred_gt_box_ious >= self.gt_iou_thr)

        # This will be multi-label, NOT one-hot encoded
        action_labels_mat = np.zeros((num_predict_boxes, num_predict_boxes, self.full_dataset.num_actions))
        for (head_gt_ind, tail_gt_ind), rel_id in zip(gt_ho_pairs, gt_labels):
            for head_predict_ind in np.flatnonzero(predict_gt_match[:, head_gt_ind]):
                for tail_predict_ind in np.flatnonzero(predict_gt_match[:, tail_gt_ind]):
                    if head_predict_ind != tail_predict_ind:
                        action_labels_mat[head_predict_ind, tail_predict_ind, rel_id] = 1.0

        if self.null_as_bg:  # treat null action as negative/background
            ho_fg_mask = action_labels_mat[:, :, 1:].any(axis=2)
            assert not np.any(action_labels_mat[:, :, 0].astype(bool) & ho_fg_mask)  # it's either foreground or background
            ho_bg_mask = ~ho_fg_mask
            action_labels_mat[:, :, 0] = ho_bg_mask.astype(np.float)
        else:  # null action is separate from background
            ho_fg_mask = action_labels_mat.any(axis=2)
            ho_bg_mask = ~ho_fg_mask

        # Filter irrelevant BG relationships (i.e., those where the subject is not human or self-relations).
        non_human_box_inds = (gt_label_of_predicted_boxes != self.full_dataset.human_class)
        ho_bg_mask[non_human_box_inds, :] = 0
        ho_bg_mask[np.arange(num_predict_boxes), np.arange(num_predict_boxes)] = 0

        ho_fg_pairs = np.stack(np.where(ho_fg_mask), axis=1)
        possible_ho_bg_pairs = np.stack(np.where(ho_bg_mask), axis=1)
        num_bg_to_sample = max(ho_fg_pairs.shape[0], 1) * bg_ratio
        bg_inds = np.random.permutation(possible_ho_bg_pairs.shape[0])[:num_bg_to_sample]
        ho_bg_pairs = possible_ho_bg_pairs[bg_inds, :]

        # Assign human-object pairs, action labels and object labels
        ho_pairs = np.concatenate([ho_fg_pairs, ho_bg_pairs], axis=0).astype(np.int, copy=False)  # [sub_ind, obj_ind]
        action_labels = action_labels_mat[ho_pairs[:, 0], ho_pairs[:, 1]].astype(np.int8, copy=False)  # multi-label
        obj_labels = gt_label_of_predicted_boxes[ho_pairs[:, 1]].astype(np.float32, copy=False)
        neg_box_labels = (obj_labels < 0)
        assert np.all(action_labels[neg_box_labels, 0]) and not np.any(action_labels[neg_box_labels, 1:])  # obj=-1 => null action
        obj_labels[neg_box_labels] = self.full_dataset.human_class  # FIXME this is non correct... but it also only affects null actions

        if not self.full_dataset.labels_are_actions:  # convert back to interaction labels
            hoi_inds, acts = np.where(action_labels)
            hoi_objs = obj_labels[hoi_inds].astype(np.int)
            hoi_labels = self.full_dataset.oa_to_interaction[hoi_objs, acts]
            assert np.all(hoi_labels >= 0)
            hoi_labels_onehot = np.zeros((action_labels.shape[0], self.full_dataset.num_interactions), dtype=np.int8)
            hoi_labels_onehot[hoi_inds, hoi_labels] = 1
            labels = hoi_labels_onehot
        else:
            labels = action_labels

        if gt_img_data.ps_labels is not None:
            gt_pstate_labels = gt_img_data.ps_labels[keep]
            pstate_labels_mat = np.zeros((num_predict_boxes, num_predict_boxes, self.full_dataset.num_states))
            for (head_gt_ind, tail_gt_ind), ps_labels in zip(gt_ho_pairs, gt_pstate_labels):
                for head_predict_ind in np.flatnonzero(predict_gt_match[:, head_gt_ind]):
                    for tail_predict_ind in np.flatnonzero(predict_gt_match[:, tail_gt_ind]):
                        if head_predict_ind != tail_predict_ind:
                            pstate_labels_mat[head_predict_ind, tail_predict_ind, :] += ps_labels
            pstate_labels_mat = np.minimum(pstate_labels_mat, 1)
            pstate_labels = pstate_labels_mat[ho_pairs[:, 0], ho_pairs[:, 1]].astype(np.int8, copy=False)  # multi-label
        else:
            pstate_labels = None
        assert all([x is None or x.dtype == np.int8 for x in [labels, pstate_labels]])
        return ho_pairs, labels, obj_labels, pstate_labels

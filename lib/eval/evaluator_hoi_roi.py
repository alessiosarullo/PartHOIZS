from collections import Counter
from typing import List, Dict

import numpy as np

from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hicodet_hake import HoiDatasetSplit
from lib.dataset.hoi_dataset import GTImgData
from lib.eval.eval_utils import Evaluator, MetricFormatter, sort_and_filter
from lib.models.abstract_model import Prediction
from lib.timer import Timer


class EvaluatorHoiRoi(Evaluator):
    """
    Efficient porting of the evaluation code for HICO-DET.
    """

    def __init__(self, dataset_split: HoiDatasetSplit, iou_thresh=0.5, hoi_score_thr=None, num_hoi_thr=None):
        super().__init__()

        self.dataset_split = dataset_split  # type: HoiDatasetSplit
        self.full_dataset = self.dataset_split.full_dataset
        self.gt_data = self.dataset_split.all_gt_img_data
        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr
        self.num_hoi_thr = num_hoi_thr

        self._init()

    def _init(self):
        self.gt_labels = []
        self.pred_gt_assignment_per_hoi = []
        self.gt_count = 0

        self.metrics = {}

    def output_metrics(self, to_keep=None, compute_pos=True, sort=False, **kwargs):
        mf = MetricFormatter()

        metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=to_keep)

        if compute_pos:
            # Same, but with null interaction filtered
            no_null_interaction_mask = (self.full_dataset.interactions[:, 0] > 0)
            if to_keep is None:
                interactions_to_keep = sorted(np.flatnonzero(no_null_interaction_mask).tolist())
            else:
                interaction_mask = np.zeros(self.full_dataset.num_interactions, dtype=bool)
                interaction_mask[np.array(to_keep).astype(np.int)] = True
                interactions_to_keep = sorted(np.flatnonzero(no_null_interaction_mask & interaction_mask).tolist())
            pos_metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=interactions_to_keep, prefix='p')

            for k, v in pos_metrics.items():
                assert k not in metrics.keys()
                metrics[k] = v
        return metrics

    def _output_metrics(self, mformatter, sort, interactions_to_keep, prefix=''):
        gt_labels = np.concatenate(self.gt_labels, axis=0)
        gt_hoi_class_hist, hoi_metrics, hoi_class_inds = sort_and_filter(metrics=self.metrics,
                                                                         gt_labels=gt_labels,
                                                                         all_classes=list(range(self.full_dataset.num_interactions)),
                                                                         sort=sort,
                                                                         keep_inds=interactions_to_keep,
                                                                         metric_prefix=prefix)
        mformatter.format_metric_and_gt_lines(gt_hoi_class_hist, hoi_metrics, hoi_class_inds, gt_str='GT HOIs', verbose=cfg.verbose)
        return hoi_metrics

    def evaluate_predictions(self, predictions: List[Dict], **kwargs):
        self._init()
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()

        Timer.get('Eval epoch', 'Conversion').tic()
        predictions = [Prediction(p) for p in predictions]
        Timer.get('Eval epoch', 'Conversion').toc()

        Timer.get('Eval epoch', 'Predictions').tic()
        for i, p in enumerate(predictions):
            self.match_prediction_to_gt(self.gt_data[i], p)
        Timer.get('Eval epoch', 'Predictions').toc()

        Timer.get('Eval epoch', 'Metrics').tic()
        self.compute_metrics(predictions=predictions)
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

    def compute_metrics(self, predictions: List[Prediction]):
        gt_hoi_labels = np.concatenate(self.gt_labels, axis=0)
        assert self.gt_count == gt_hoi_labels.shape[0]
        pred_gt_ho_assignment = np.concatenate(self.pred_gt_assignment_per_hoi, axis=0)

        gt_hoi_classes_count = Counter(gt_hoi_labels.tolist())

        ap = np.zeros(self.full_dataset.num_interactions)
        recall = np.zeros(self.full_dataset.num_interactions)
        for j in range(self.full_dataset.num_interactions):
            num_gt_hois = gt_hoi_classes_count[j]
            if num_gt_hois == 0:
                continue

            p_hoi_scores = np.concatenate([p.output_scores[:, j] for p in predictions if p.output_scores is not None], axis=0)
            p_gt_ho_assignment = pred_gt_ho_assignment[:, j]
            assert p_hoi_scores.shape == p_gt_ho_assignment.shape
            if self.hoi_score_thr is not None:
                inds = (p_hoi_scores >= self.hoi_score_thr)
                p_hoi_scores = p_hoi_scores[inds]
                p_gt_ho_assignment = p_gt_ho_assignment[inds]
            if self.num_hoi_thr is not None:
                p_hoi_scores = p_hoi_scores[:self.num_hoi_thr]
                p_gt_ho_assignment = p_gt_ho_assignment[:self.num_hoi_thr]
            rec_j, prec_j, ap_j = self.eval_single_interaction_class(p_hoi_scores, p_gt_ho_assignment, num_gt_hois)
            ap[j] = ap_j
            if rec_j.size > 0:
                recall[j] = rec_j[-1]
        self.metrics['M-mAP'] = ap

    def match_prediction_to_gt(self, gt_entry: GTImgData, prediction: Prediction):
        gt_boxes, gt_ho_pairs, gt_labels = self._process_gt(gt_entry=gt_entry)
        if self.full_dataset.labels_are_actions:
            gt_action_labels = gt_labels
            gt_labels = self.full_dataset.oa_to_interaction[gt_entry.box_classes[gt_ho_pairs[:, 1]], gt_action_labels]
            assert np.all(gt_labels >= 0)
        num_gt_hois = gt_ho_pairs.shape[0]

        gt_ho_ids = self.gt_count + np.arange(num_gt_hois)
        self.gt_count += num_gt_hois

        predict_boxes = np.zeros((0, 4))
        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        if prediction.obj_boxes is not None:
            predict_boxes = prediction.obj_boxes
            if prediction.ho_pairs is not None:
                predict_ho_pairs = prediction.ho_pairs
        else:
            assert prediction.ho_pairs is None

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_assignment_per_hoi = np.full((predict_ho_pairs.shape[0], self.full_dataset.num_interactions), fill_value=-1, dtype=np.int)
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go) in enumerate(gt_ho_pairs):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious >= self.iou_thresh):
                gt_pair_ious_per_hoi = np.zeros((num_gt_hois, self.full_dataset.num_interactions))
                gt_pair_ious_per_hoi[np.arange(num_gt_hois), gt_labels] = gt_pair_ious
                gt_assignments = gt_pair_ious_per_hoi.argmax(axis=0)[np.any(gt_pair_ious_per_hoi >= self.iou_thresh, axis=0)]
                gt_hoi_assignments = gt_labels[gt_assignments]
                assert np.unique(gt_assignments).size == gt_assignments.size
                assert np.unique(gt_hoi_assignments).size == gt_hoi_assignments.size
                pred_gt_assignment_per_hoi[predict_idx, gt_hoi_assignments] = gt_ho_ids[gt_assignments]

        self.gt_labels.append(gt_labels)
        self.pred_gt_assignment_per_hoi.append(pred_gt_assignment_per_hoi)

    @staticmethod
    def eval_single_interaction_class(predicted_conf_scores, pred_gtid_assignment, num_hoi_gt_positives):
        num_predictions = predicted_conf_scores.shape[0]
        tp = np.zeros(num_predictions)

        if num_predictions > 0:
            inds = np.argsort(predicted_conf_scores)[::-1]
            pred_gtid_assignment = pred_gtid_assignment[inds]

            matched_gt_inds, highest_scoring_pred_idx_per_gt_ind = np.unique(pred_gtid_assignment, return_index=True)
            if matched_gt_inds[0] == -1:
                # matched_gt_inds = matched_gt_inds[1:]
                highest_scoring_pred_idx_per_gt_ind = highest_scoring_pred_idx_per_gt_ind[1:]
            tp[highest_scoring_pred_idx_per_gt_ind] = 1

        fp = 1 - tp
        assert np.all(fp[pred_gtid_assignment < 0] == 1)

        # compute precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / num_hoi_gt_positives
        prec = tp / (fp + tp)

        # compute average precision
        num_bins = 10  # uniformly distributed in [0, 1) (e.g., use 10 for 0.1 spacing)
        thr_values, thr_inds = np.unique(np.floor(rec * num_bins) / num_bins, return_index=True)
        rec_thresholds = np.full(num_bins + 1, fill_value=-1, dtype=np.int)
        rec_thresholds[np.floor(thr_values * num_bins).astype(np.int)] = thr_inds
        for i in range(num_bins, 0, -1):  # fix gaps of -1s
            if rec_thresholds[i - 1] < 0 <= rec_thresholds[i]:
                rec_thresholds[i - 1] = rec_thresholds[i]
        assert rec_thresholds[0] == 0

        max_p = np.maximum.accumulate(prec[::-1])[::-1]
        ap = np.sum(max_p[rec_thresholds[rec_thresholds >= 0]] / rec_thresholds.size)
        return rec, prec, ap

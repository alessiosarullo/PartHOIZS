import pickle
from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from lib.eval.eval_utils import Evaluator, MetricFormatter, sort_and_filter
from lib.dataset.hico_hake import HicoHakeSplit
from lib.models.abstract_model import Prediction
from lib.timer import Timer
from config import cfg


class EvaluatorHico(Evaluator):
    def __init__(self, dataset_split: HicoHakeSplit):
        self.dataset_split = dataset_split
        self.full_dataset = self.dataset_split.full_dataset
        self.metrics = {}  # type: Dict[str, np.ndarray]
        self.gt_scores = self.full_dataset.split_labels[self.dataset_split.split]

    def evaluate_predictions(self, predictions: List[Dict], **kwargs):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()
        Timer.get('Eval epoch', 'Predictions').tic()
        predict_hoi_scores = np.full_like(self.gt_scores, fill_value=np.nan)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_hoi_scores[i, :] = prediction.output_scores
        Timer.get('Eval epoch', 'Predictions').toc()

        Timer.get('Eval epoch', 'Metrics').tic()
        gt_scores = self.gt_scores
        gt_scores[gt_scores < 0] = 0
        self.metrics['M-mAP'] = average_precision_score(gt_scores, predict_hoi_scores, average=None)
        # self.metrics['M-rec'] = recall_score(gt_scores, predict_hoi_scores, average=None)
        # self.metrics['M-pr'] = precision_score(gt_scores, predict_hoi_scores, average=None)
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

    def output_metrics(self, to_keep=None, compute_pos=True, **kwargs):
        mf = MetricFormatter()

        metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=to_keep)

        if compute_pos:
            # Same, but with null interaction filtered
            no_null_interaction_mask = (self.full_dataset.interactions[:, 0] > 0)
            if to_keep is None:
                to_keep = sorted(np.flatnonzero(no_null_interaction_mask).tolist())
            else:
                interaction_mask = np.zeros(self.full_dataset.num_interactions, dtype=bool)
                interaction_mask[np.array(to_keep).astype(np.int)] = True
                to_keep = sorted(np.flatnonzero(no_null_interaction_mask & interaction_mask).tolist())
            pos_metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=to_keep, prefix='p')

            for k, v in pos_metrics.items():
                assert k not in metrics.keys()
                metrics[k] = v
        return metrics

    def _output_metrics(self, mformatter, sort, interactions_to_keep, prefix=''):
        gt_hoi_class_hist, hoi_metrics, hoi_class_inds = sort_and_filter(metrics=self.metrics,
                                                                         gt_labels=np.where(self.gt_scores)[1],
                                                                         all_classes=list(range(self.full_dataset.num_interactions)),
                                                                         sort=sort,
                                                                         keep_inds=interactions_to_keep,
                                                                         metric_prefix=prefix)
        mformatter.format_metric_and_gt_lines(gt_hoi_class_hist, hoi_metrics, hoi_class_inds, gt_str='GT HOIs', verbose=cfg.verbose)
        return hoi_metrics

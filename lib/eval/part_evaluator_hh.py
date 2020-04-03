import pickle
from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from config import cfg
from lib.models.abstract_model import Prediction
from lib.dataset.hico_hake import HicoHakeSplit
from lib.eval.eval_utils import MetricFormatter, sort_and_filter
from lib.timer import Timer


class PartEvaluatorHH:
    def __init__(self, dataset_split: HicoHakeSplit):
        super().__init__()
        self.dataset_split = dataset_split  # type: HicoHakeSplit
        self.full_dataset = dataset_split.full_dataset
        self.metrics = {}  # type: Dict[str, np.ndarray]

        self.gt_scores = self.full_dataset.split_part_annotations[self.dataset_split.split]

    def load(self, fn):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
            self.__dict__.update(d)

    @property
    def gt_part_action_labels(self):
        return np.where(self.gt_scores)[1]

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump({'metrics': self.metrics}, f)

    def evaluate_predictions(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()
        Timer.get('Eval epoch', 'Predictions').tic()
        predict_part_action_scores = np.full_like(self.gt_scores, fill_value=np.nan, dtype=np.float32)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_part_action_scores[i, :] = prediction.part_state_scores
        Timer.get('Eval epoch', 'Predictions').toc()

        Timer.get('Eval epoch', 'Metrics').tic()
        gt_scores = self.gt_scores
        gt_scores[gt_scores < 0] = 0
        macro_map = average_precision_score(gt_scores, predict_part_action_scores, average=None)
        macro_map[~np.any(gt_scores, axis=0)] = 0
        self.metrics['M-mAP'] = macro_map
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

    def output_metrics(self, sort=False, interactions_to_keep=None, compute_pos=True):
        mf = MetricFormatter()
        metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=interactions_to_keep)

        # Same, but with null interaction filtered
        if compute_pos:
            no_null_actions = [i for i, p in enumerate(self.full_dataset.bp_ps_pairs) if p[1] != self.full_dataset.null_action]
            pos_metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=no_null_actions, prefix='p')

            for k, v in pos_metrics.items():
                assert k not in metrics.keys()
                metrics[k] = v
        return metrics

    def _output_metrics(self, mformatter, sort, interactions_to_keep, prefix=''):
        gt_hoi_class_hist, hoi_metrics, hoi_class_inds = sort_and_filter(metrics=self.metrics,
                                                                         gt_labels=self.gt_part_action_labels,
                                                                         all_classes=list(range(len(self.full_dataset.bp_ps_pairs))),
                                                                         sort=sort,
                                                                         keep_inds=interactions_to_keep,
                                                                         metric_prefix=prefix)
        mformatter.format_metric_and_gt_lines(gt_hoi_class_hist, hoi_metrics, hoi_class_inds, gt_str='GT HOIs', verbose=cfg.verbose)
        return hoi_metrics

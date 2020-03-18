import pickle
from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from lib.dataset.hico import HicoSplit
from lib.eval.eval_utils import BaseEvaluator
from lib.models.abstract_model import Prediction
from lib.timer import Timer


class SeenEvaluatorImg(BaseEvaluator):
    def __init__(self, dataset_split: HicoSplit):
        super().__init__(dataset_split)
        gt_scores = self.full_dataset.split_img_labels[self.dataset_split.split]
        unseen_interactions = np.setdiff1d(np.arange(self.full_dataset.num_interactions), self.dataset_split.seen_interactions)
        self.gt_scores = ~gt_scores[:, unseen_interactions].any(axis=1, keepdims=True)

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump({'metrics': self.metrics}, f)

    def evaluate_predictions(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()
        Timer.get('Eval epoch', 'Predictions').tic()
        predict_seen_hoi_scores = np.full_like(self.gt_scores, fill_value=np.nan)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_seen_hoi_scores[i, :] = prediction.seen_scores
        Timer.get('Eval epoch', 'Predictions').toc()

        Timer.get('Eval epoch', 'Metrics').tic()
        gt_scores = self.gt_scores
        gt_scores[gt_scores < 0] = 0
        self.metrics['M-mAP'] = average_precision_score(gt_scores, predict_seen_hoi_scores, average=None)
        # self.metrics['M-rec'] = recall_score(gt_scores, predict_hoi_scores, average=None)
        # self.metrics['M-pr'] = precision_score(gt_scores, predict_hoi_scores, average=None)
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

    def output_metrics(self, sort=False, interactions_to_keep=None, compute_pos=True):
        return self.metrics

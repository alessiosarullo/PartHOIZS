import pickle
from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from lib.dataset.hico import HicoSplit
from lib.eval.eval_utils import BaseEvaluator
from lib.containers import Prediction
from lib.utils import Timer


class EvaluatorImg(BaseEvaluator):
    def __init__(self, dataset_split: HicoSplit):
        super().__init__(dataset_split)

        self.gt_scores = self.full_dataset.split_annotations[self.dataset_split._data_split]

    @property
    def gt_hoi_labels(self):
        return np.where(self.gt_scores)[1]

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump({'metrics': self.metrics}, f)

    def evaluate_predictions(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()
        Timer.get('Eval epoch', 'Predictions').tic()
        predict_hoi_scores = np.full_like(self.gt_scores, fill_value=np.nan)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_hoi_scores[i, :] = prediction.hoi_scores
        Timer.get('Eval epoch', 'Predictions').toc()

        Timer.get('Eval epoch', 'Metrics').tic()
        gt_scores = self.gt_scores
        gt_scores[gt_scores < 0] = 0
        self.metrics['M-mAP'] = average_precision_score(gt_scores, predict_hoi_scores, average=None)
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

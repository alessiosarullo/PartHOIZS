import argparse
import os
import pickle
import sys
from typing import List, Dict

import matplotlib

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['zs', '--save_dir', 'output/hicoall/zs185_gc_nobg/standard/2019-08-23_19-05-46_SINGLE']
    sys.argv[1:] = ['pa', '--save_dir', 'output/base/po_trnull/standard/2019-12-02_16-40-23_SINGLE/']

except ImportError:
    pass

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from matplotlib import pyplot as plt
from PIL import Image

from analysis.utils import plot_mat
from config import cfg
from lib.containers import Prediction
from lib.dataset.hico import HicoSplit
from lib.dataset.utils import Splits, interactions_to_mat
from lib.dataset.hico_hake import HicoHake, HicoHakeKPSplit
from analysis.visualise_utils import Visualizer


class Analyser:
    def __init__(self, dataset: HicoSplit, hoi_score_thr=0.05):  # FIXME thr default
        super().__init__()

        self.hoi_score_thr = hoi_score_thr
        print(f'Thr score: {hoi_score_thr}')

        self.dataset_split = dataset  # type: HicoSplit
        self.full_dataset = dataset.full_dataset

        # self.ph_num_bins = 20
        # self.ph_bins = np.arange(self.ph_num_bins + 1) / self.ph_num_bins
        # self.prediction_act_hist = np.zeros((self.ph_bins.size, self.dataset_split.num_actions), dtype=np.int)

        gt_hoi_labels = self.full_dataset.split_annotations[self.dataset_split._data_split]
        gt_hoi_labels[gt_hoi_labels < 0] = 0
        self.gt_hoi_labels = gt_hoi_labels
        self.gt_act_labels = self._hoi_labels_to_act_labels(gt_hoi_labels)

    @property
    def num_gt_acts(self):
        return np.sum(self.gt_act_labels, axis=0)

    def compute_act_stats(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        predict_hoi_scores = np.full_like(self.gt_hoi_labels, fill_value=np.nan)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_hoi_scores[i, :] = prediction.hoi_scores
        assert not np.any(np.isnan(predict_hoi_scores)) and np.all(predict_hoi_scores >= 0)

        hoi_predictions = (predict_hoi_scores >= self.hoi_score_thr)
        act_predictions = self._hoi_labels_to_act_labels(hoi_predictions)

        conf_mat = multilabel_confusion_matrix(self.gt_act_labels, act_predictions)
        conf_mat = conf_mat.reshape(-1, 4)  # TN, FP, FN, TP
        conf_mat = conf_mat[:, [3, 1, 2, 0]]  # TP, FP, FN, TN
        conf_mat = conf_mat / conf_mat.sum(axis=1, keepdims=True)  # normalise

        with np.errstate(divide='ignore', invalid='ignore'):
            hits = (act_predictions > 0) & (self.gt_act_labels > 0)
            recall = np.sum(hits, axis=0) / self.num_gt_acts
            precision = np.sum(hits, axis=0) / np.maximum(1, np.sum(act_predictions, axis=0))
            assert not np.any(np.isinf(recall)) and not np.any(np.isnan(recall)) and np.all(recall <= 1) and np.all(recall >= 0)
            assert not np.any(np.isinf(precision)) and not np.any(np.isnan(precision)) and np.all(precision <= 1) and np.all(precision >= 0)
        return conf_mat, recall, precision

    def _hoi_labels_to_act_labels(self, hoi_labels):
        act_labels = np.max(interactions_to_mat(hoi_labels, self.full_dataset, np2np=True), axis=1)
        assert np.all((act_labels == 0) | (act_labels == 1))
        return act_labels


def _setup_and_load():
    cfg.parse_args(fail_if_missing=False, reset=True)

    with open(cfg.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results


def zs_stats():
    results = _setup_and_load()
    res_save_path = cfg.output_analysis_path
    os.makedirs(res_save_path, exist_ok=True)

    inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
    seen_act_inds = sorted(inds_dict[Splits.TRAIN.value]['pred'].tolist())
    seen_obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())

    splits = HicoSplit.get_splits(obj_inds=seen_obj_inds, act_inds=seen_act_inds)
    train_split, val_split, test_split = splits[Splits.TRAIN], splits[Splits.VAL], splits[Splits.TEST]
    hicodet = train_split.full_dataset
    seen_interactions = np.zeros((hicodet.num_objects, hicodet.num_actions), dtype=bool)
    seen_interactions[train_split.interactions[:, 1], train_split.interactions[:, 0]] = 1

    unseen_interactions = np.zeros((hicodet.num_objects, hicodet.num_actions), dtype=bool)
    unseen_interactions[test_split.interactions[:, 1], test_split.interactions[:, 0]] = 1
    unseen_interactions[seen_interactions] = 0

    analyser = Analyser(dataset=test_split)
    act_conf_mat, recall, precision = analyser.compute_act_stats(results)
    rec_pr = np.stack([recall, precision], axis=1)
    conf_mat_labels = ['TP', 'FP', 'FN', 'TN']

    act_inds = (np.argsort(analyser.num_gt_acts[1:])[::-1] + 1).tolist()
    # act_inds += [0]  # add no_interaction at the end
    unseen_act_inds_set = set(test_split.active_actions.tolist()) - set(train_split.active_actions.tolist())

    # actions = [f'{test_split.actions[i]}{"*" if i in unseen_act_inds_set else ""}' for i in act_inds]
    # plot_mat(np.concatenate([act_conf_mat, rec_pr], axis=1)[act_inds, :].T, actions, conf_mat_labels + ['Rec', 'Prec'],
    #          y_inds=np.arange(6), x_inds=act_inds, plot=False, alternate_labels=False, cbar=False)
    # plt.savefig(os.path.join(res_save_path, 'stats.png'), dpi=300)
    # print(f'Avg recall: {100 * np.mean(recall):.3f}%')
    # print(f'Avg precision: {100 * np.mean(precision):.3f}%')

    unseen_act_inds = np.array([i for i in act_inds if i in unseen_act_inds_set])
    zs_actions = [test_split.actions[i] for i in unseen_act_inds]
    plot_mat(np.concatenate([act_conf_mat, rec_pr], axis=1)[unseen_act_inds, :].T, zs_actions, conf_mat_labels + ['Rec', 'Prec'],
             y_inds=np.arange(6), x_inds=unseen_act_inds, plot=False, alternate_labels=False, cbar=False, zero_color=[0.8, 0, 0.8, 1])
    plt.savefig(os.path.join(res_save_path, 'zs_stats.png'), dpi=300)
    print(f'Avg recall: {100 * np.mean(recall[unseen_act_inds]):.3f}%')
    print(f'Avg precision: {100 * np.mean(precision[unseen_act_inds]):.3f}%')

    # plt.show()


def examine_part_action_predictions():
    results = _setup_and_load()
    hh = HicoHake()
    split = Splits.TEST
    # splits = HicoHakeKPSplit.get_splits()

    gt_part_action_labels = hh.split_part_annotations[split]
    gt_part_action_labels[gt_part_action_labels < 0] = 0

    predict_part_act_scores = np.full_like(gt_part_action_labels, fill_value=np.nan)
    for i, res in enumerate(results):
        fname = hh.split_filenames[split][i]

        path = os.path.join(hh.get_img_dir(split), fname)

        try:
            img = Image.open(path)
        except:
            print(f'Error on image {i}: {fname}.')
            continue

        prediction = Prediction(res)
        pred_i = prediction.part_action_scores.squeeze()
        predict_part_act_scores[i, :] = pred_i

        gt_i = gt_part_action_labels[i, :]

        print(f'Image {i:6d} ({fname}).')
        for j, acts in enumerate(hh.actions_per_part):
            print(f'{hh.parts[j].upper():10s} {"GT":10s}', ' '.join([f'{str(a) + ("*" if gt_i[a] else ""):>5s}' for a in acts]))
            print(f'{"":10s} {"Pred":10s}', ' '.join([f'{s:5.3f}'.replace("0.000", "    0").replace("0.", " .") for s in pred_i[acts]]))
        print()

        visualizer = Visualizer(img)
        plt.imshow(visualizer.output.get_image())
        plt.show()

    assert not np.any(np.isnan(predict_part_act_scores)) and np.all(predict_part_act_scores >= 0)


def main():
    funcs = {
        'zs': zs_stats,
        'pa': examine_part_action_predictions
    }
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=funcs.keys())
    namespace = parser.parse_known_args()
    func = vars(namespace[0])['func']
    sys.argv = sys.argv[:1] + namespace[1]
    print(sys.argv)
    funcs[func]()


if __name__ == '__main__':
    main()

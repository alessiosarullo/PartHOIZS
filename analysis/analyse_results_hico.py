import argparse
import os
import pickle
import sys
from typing import List, Dict

import matplotlib

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['zs', '--save_dir', 'output/base/zs4/nopart/2020-01-20_16-26-42_SINGLE']
    # sys.argv[1:] = ['zs', '--save_dir', 'output/part2hoi/zs4/standard/2020-01-20_16-27-47_SINGLE']
    # sys.argv[1:] = ['zs', '--save_dir', 'output/part2hoi/zs4/pbf/2020-01-20_19-06-50_SINGLE']
    # sys.argv[1:] = ['zs', '--save_dir', 'output/attp/zs4/pbf/2020-01-20_19-07-18_SINGLE']

    # sys.argv[1:] = ['ppred', '--save_dir', 'output/base/zs4/nopart/2020-01-20_16-26-42_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/base/zs4_awsu1/nopart/2020-01-21_11-23-21_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/part2hoi/zs4_awsu1/pbf/2020-01-21_11-24-00_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/base/zs4_awsu1_ptres/pbf/2020-01-22_12-32-42_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/base/zs4_awsu1_pint/pbf/2020-01-22_15-52-57_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/tri/zs4/pbf/2020-01-23_11-41-04_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/tri/zs4_awsufp1/pbf/2020-01-23_11-50-31_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/tri/zs4_awsu1_awsufp1/pbf/2020-01-23_11-52-04_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/tri/zs4_awsu1_awsufp1/pbf/2020-01-23_11-52-04_SINGLE']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/hoi/zs4/nopart/2020-01-23_19-23-29_SINGLE', '--hoi_thr', '0.02', '--vis']

    sys.argv[1:] = ['ppred', '--save_dir', 'output/tin/zs4_adam/pbf_lr1e4/2020-02-11_15-47-36_SINGLE', '--vis', '--null']
except ImportError:
    pass

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from matplotlib import pyplot as plt
from PIL import Image

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico import HicoSplit
from lib.dataset.utils import Splits, interactions_to_mat
from lib.dataset.hico_hake import HicoHake, Hico
from analysis.visualise_utils import Visualizer
from analysis.utils import analysis_hub, plot_mat


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
    return results, Splits.TEST


def zs_stats():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoi_thr', type=float, default=0.05)
    namespace = parser.parse_known_args()
    args = namespace[0]
    sys.argv = sys.argv[:1] + namespace[1]

    results, res_split = _setup_and_load()
    res_save_path = cfg.output_analysis_path
    os.makedirs(res_save_path, exist_ok=True)

    inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
    seen_act_inds = sorted(inds_dict[Splits.TRAIN.value]['act'].tolist())
    seen_obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())

    splits = HicoSplit.get_splits(obj_inds=seen_obj_inds, act_inds=seen_act_inds)
    train_split = splits[Splits.TRAIN]  # type: HicoSplit
    val_split = splits[Splits.VAL]  # type: HicoSplit
    test_split = splits[Splits.TEST]  # type: HicoSplit
    hicodet = train_split.full_dataset
    seen_interactions = np.zeros((hicodet.num_objects, hicodet.num_actions), dtype=bool)
    seen_interactions[train_split.interactions[:, 1], train_split.interactions[:, 0]] = 1

    unseen_interactions = np.zeros((hicodet.num_objects, hicodet.num_actions), dtype=bool)
    unseen_interactions[test_split.interactions[:, 1], test_split.interactions[:, 0]] = 1
    unseen_interactions[seen_interactions] = 0

    analyser = Analyser(dataset=test_split, hoi_score_thr=args.hoi_thr)
    act_conf_mat, recall, precision = analyser.compute_act_stats(results)
    rec_pr = np.stack([recall, precision], axis=1)
    conf_mat_labels = ['TP', 'FP', 'FN', 'TN']

    act_inds = (np.argsort(analyser.num_gt_acts[1:])[::-1] + 1).tolist()
    # act_inds += [0]  # add no_interaction at the end
    unseen_act_inds_set = set(test_split.seen_actions.tolist()) - set(train_split.seen_actions.tolist())

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
    plt.savefig(os.path.join(res_save_path, 'zs_stats.png'), dpi=300, bbox_inches='tight')
    print(f'Avg recall: {100 * np.mean(recall[unseen_act_inds]):.3f}%')
    print(f'Avg precision: {100 * np.mean(precision[unseen_act_inds]):.3f}%')

    # plt.show()


def examine_part_action_predictions():
    results, res_split = _setup_and_load()
    hh = HicoHake()
    # splits = HicoHakeKPSplit.get_splits()

    gt_part_action_labels = hh.split_part_annotations[res_split]
    gt_part_action_labels[gt_part_action_labels < 0] = 0

    predict_part_act_scores = np.full_like(gt_part_action_labels, fill_value=np.nan)
    for i, res in enumerate(results):
        fname = hh.split_filenames[res_split][i]

        path = os.path.join(hh.get_img_dir(res_split), fname)

        try:
            img = Image.open(path)
        except:
            print(f'Error on image {i}: {fname}.')
            continue

        prediction = Prediction(res)
        pred_i = prediction.part_state_scores.squeeze()
        predict_part_act_scores[i, :] = pred_i

        gt_i = gt_part_action_labels[i, :]

        print(f'Image {i:6d} ({fname}).')
        for j, acts in enumerate(hh.states_per_part):
            print(f'{hh.parts[j].upper():10s} {"GT":10s}', ' '.join([f'{str(a) + ("*" if gt_i[a] else ""):>5s}' for a in acts]))
            print(f'{"":10s} {"Pred":10s}', ' '.join([f'{s:5.3f}'.replace("0.000", "    0").replace("0.", " .") for s in pred_i[acts]]))
        print()

        visualizer = Visualizer(img)
        plt.imshow(visualizer.output.get_image())
        plt.show()

    assert not np.any(np.isnan(predict_part_act_scores)) and np.all(predict_part_act_scores >= 0)


def print_predictions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--hoi_thr', type=float, default=0.05)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--null', action='store_true', default=False)
    namespace = parser.parse_known_args()
    args = namespace[0]
    sys.argv = sys.argv[:1] + namespace[1]

    results, res_split = _setup_and_load()
    hd = Hico()
    hds_fns = hd.split_filenames[res_split]

    assert cfg.seenf >= 0
    inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
    obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())
    act_inds = sorted(inds_dict[Splits.TRAIN.value]['act'].tolist())

    print(f'{Splits.TRAIN.value.capitalize()} objects ({len(obj_inds)}):', [hd.objects[i] for i in obj_inds])
    print(f'{Splits.TRAIN.value.capitalize()} actions ({len(act_inds)}):', [hd.actions[i] for i in act_inds])
    seen_objs = set(obj_inds)
    seen_acts = set(act_inds)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.output_path.split('/')[1:]), res_split.value)
    os.makedirs(output_dir, exist_ok=True)
    for idx, im_fn in enumerate(hds_fns):
        img_id = int(im_fn.split('_')[-1].split('.')[0])
        prediction_dict = results[idx]
        prediction = Prediction(prediction_dict)

        img_anns = hd.split_annotations[res_split][idx, :]
        img_hits = np.zeros_like(img_anns, dtype=bool)
        hoi_scores = np.squeeze(prediction.hoi_scores, axis=0)
        inds = hoi_scores.argsort()[::-1]

        hoi_str = []
        for i, s in zip(inds, hoi_scores[inds]):
            act_ind = hd.interactions[i, 0]
            obj_ind = hd.interactions[i, 1]
            if s >= args.hoi_thr and (args.null or act_ind > 0):
                act_str = ('*' if act_ind not in seen_acts else '') + hd.actions[act_ind]
                obj_str = ('*' if obj_ind not in seen_objs else '') + hd.objects[obj_ind]
                if img_anns[i] > 0:
                    hit = '#'
                    img_hits[i] = True
                else:
                    hit = ''
                hoi_str.append(f'{act_str} {obj_str} ({s * 100:.2f}{hit})')

        misses_str = []
        for i, ann in enumerate(img_anns):
            act_ind = hd.interactions[i, 0]
            obj_ind = hd.interactions[i, 1]
            if ann > 0 and not img_hits[i] and (args.null or act_ind > 0):
                act_str = ('*' if act_ind not in seen_acts else '') + hd.actions[act_ind]
                obj_str = ('*' if obj_ind not in seen_objs else '') + hd.objects[obj_ind]
                misses_str.append(f'{act_str} {obj_str} ({hoi_scores[i] * 100:.2f})')

        print(f'{img_id:5d} PRED {", ".join(hoi_str)}')
        print(f'{"":5s} MISS {", ".join(misses_str)}')

        if args.vis:
            img = Image.open(os.path.join(hd.get_img_dir(res_split), im_fn))
            visualizer = Visualizer(img)
            vis_output = visualizer.output
            plt.imshow(vis_output.get_image())
            plt.show()

        if 0 < args.num_imgs < idx:
            break


def main():
    funcs = {
        'zs': zs_stats,
        'expa': examine_part_action_predictions,
        'ppred': print_predictions,
    }
    analysis_hub(funcs)


if __name__ == '__main__':
    main()

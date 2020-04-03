import argparse
import os
import pickle
import sys
from typing import List, Dict

import matplotlib

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['stats', 'act', '--save_dir', 'output/base/zs4/nopart/2020-01-20_16-26-42_SINGLE']
    # sys.argv[1:] = ['stats', 'act', '--save_dir', 'output/part2hoi/zs4/standard/2020-01-20_16-27-47_SINGLE']
    # sys.argv[1:] = ['stats', 'act', '--save_dir', 'output/part2hoi/zs4/pbf/2020-01-20_19-06-50_SINGLE']
    # sys.argv[1:] = ['stats', 'act', '--save_dir', 'output/attp/zs4/pbf/2020-01-20_19-07-18_SINGLE']

    # sys.argv[1:] = ['ppred', '--save_dir', 'output/hoi/zs4/nopart/2020-01-23_19-23-29_SINGLE', '--hoi_thr', '0.02', '--vis']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/hoi/zs4/nopart_plus-box/2020-02-14_15-23-00_SINGLE', '--vis']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/frompscoatt/zs4/pbf/2020-02-17_13-34-24_SINGLE', '--vis', '--part']

    # sys.argv[1:] = ['ppred', '--save_dir', 'output/frompsgcnatt/zs4/pbf_gcn/2020-02-28_09-58-38_SINGLE', '--hoi_thr', '0.02', '--vis']
    # sys.argv[1:] = ['ps-h_att', '--save_dir', 'output/frompsgcnatt/zs4/pbf_gcn/2020-02-28_09-58-38_SINGLE']

    # sys.argv[1:] = ['stats', 'hoi', '--save_dir', 'output/frompsgcnatt/zs4/pbf_gcn_nocspos/2020-03-02_12-24-34_SINGLE', '--hoi_thr', '0.02']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/frompsgcnatt/zs4/pbf_gcn_nocspos/2020-03-02_12-24-34_SINGLE', '--hoi_thr', '0.02']
    # sys.argv[1:] = ['ppred', '--save_dir', 'output/frompsgcnatt/zs4/pbf_gcn_nocspos/2020-03-02_12-24-34_SINGLE', '--hoi_thr', '0.02', '--vis']
    sys.argv[1:] = ['ps-h_att', '--save_dir', 'output/frompsgcnatt/zs4/pbf_gcn_nocspos/2020-03-02_12-24-34_SINGLE']
except ImportError:
    pass

import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from matplotlib import pyplot as plt
from PIL import Image

from config import cfg
from lib.models.abstract_model import Prediction
from lib.dataset.hico_hake import HicoHakeSplit
from lib.dataset.utils import interactions_to_mat
from lib.dataset.hico_hake import HicoHake
from analysis.visualise_utils import Visualizer
from analysis.utils import analysis_hub, plot_mat
from analysis.show_embs import run_and_save


class Analyser:
    def __init__(self, dataset: HicoHakeSplit, hoi_score_thr):
        super().__init__()

        self.hoi_score_thr = hoi_score_thr
        print(f'Thr score: {hoi_score_thr}')

        self.dataset_split = dataset  # type: HicoHakeSplit
        self.full_dataset = self.dataset_split.full_dataset

        # self.ph_num_bins = 20
        # self.ph_bins = np.arange(self.ph_num_bins + 1) / self.ph_num_bins
        # self.prediction_act_hist = np.zeros((self.ph_bins.size, self.dataset_split.num_actions), dtype=np.int)

        gt_hoi_labels = self.full_dataset.split_labels[self.dataset_split.split]
        gt_hoi_labels[gt_hoi_labels < 0] = 0
        self.gt_hoi_labels = gt_hoi_labels

    def compute_act_stats(self, predictions: List[Dict], label_f):
        all_predictions = predictions
        assert len(all_predictions) == self.dataset_split.num_images, (len(all_predictions), self.dataset_split.num_images)

        predict_hoi_scores = np.full_like(self.gt_hoi_labels, fill_value=np.nan)
        for i, res in enumerate(all_predictions):
            prediction = Prediction(res)
            predict_hoi_scores[i, :] = prediction.hoi_scores
        assert not np.any(np.isnan(predict_hoi_scores)) and np.all(predict_hoi_scores >= 0)
        hoi_predictions = (predict_hoi_scores >= self.hoi_score_thr)

        predictions = label_f(hoi_predictions)
        gt_labels = label_f(self.gt_hoi_labels)

        conf_mat = multilabel_confusion_matrix(gt_labels, predictions)
        conf_mat = conf_mat.reshape(-1, 4)  # TN, FP, FN, TP
        conf_mat = conf_mat[:, [3, 1, 2, 0]]  # TP, FP, FN, TN
        conf_mat = conf_mat / conf_mat.sum(axis=1, keepdims=True)  # normalise

        with np.errstate(divide='ignore', invalid='ignore'):
            hits = (predictions > 0) & (gt_labels > 0)
            recall = np.sum(hits, axis=0) / np.sum(gt_labels, axis=0)
            precision = np.sum(hits, axis=0) / np.maximum(1, np.sum(predictions, axis=0))
            assert not np.any(np.isinf(recall)) and not np.any(np.isnan(recall)) and np.all(recall <= 1) and np.all(recall >= 0)
            assert not np.any(np.isinf(precision)) and not np.any(np.isnan(precision)) and np.all(precision <= 1) and np.all(precision >= 0)
        return conf_mat, ['TP', 'FP', 'FN', 'TN'], recall, precision


def _setup_and_load():
    cfg.parse_args(fail_if_missing=False, reset=True)

    with open(cfg.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results, 'test'


def stats():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str)
    parser.add_argument('--hoi_thr', type=float, default=0.05)
    parser.add_argument('--show', action='store_true', default=False)
    namespace = parser.parse_known_args()
    args = namespace[0]
    sys.argv = sys.argv[:1] + namespace[1]

    results, res_split = _setup_and_load()
    res_save_path = cfg.output_analysis_path
    os.makedirs(res_save_path, exist_ok=True)

    inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
    seen_act_inds = inds_dict['train']['act']
    seen_obj_inds = inds_dict['train']['obj']

    hh = HicoHake()
    test_split = HicoHakeSplit(split='test', full_dataset=hh)
    analyser = Analyser(dataset=test_split, hoi_score_thr=args.hoi_thr)

    def _hoi_labels_to_act_labels(hoi_labels):
        act_labels = np.max(interactions_to_mat(hoi_labels, hh, np2np=True), axis=1)
        assert np.all((act_labels == 0) | (act_labels == 1))
        return act_labels

    if args.type == 'act':
        label_f = _hoi_labels_to_act_labels
        class_labels = test_split.actions
        unseen_inds_set = np.setdiff1d(test_split.seen_actions, seen_act_inds)
    elif args.type == 'hoi':
        label_f = lambda x: x
        class_labels = hh.interactions_str
        seen_op_mat = hh.oa_pair_to_interaction[seen_obj_inds, :][:, seen_act_inds]
        seen_inds = np.setdiff1d(np.unique(seen_op_mat), -1).astype(np.int)
        unseen_inds_set = np.setdiff1d(test_split.seen_interactions, seen_inds)
    else:
        raise NotImplementedError

    conf_mat, conf_mat_labels, recall, precision = analyser.compute_act_stats(results, label_f=label_f)
    stat_mat = np.concatenate([conf_mat, recall[:, None], precision[:, None]], axis=1)

    gt_counts = np.sum(label_f(analyser.gt_hoi_labels), axis=0)
    sorted_inds = (np.argsort(gt_counts)[::-1]).tolist()
    sorted_unseen_inds = np.array([i for i in sorted_inds if i in unseen_inds_set])

    labels = [f'{class_labels[i]}{"*" if i in sorted_unseen_inds else ""}' for i in sorted_inds]
    zs_labels = [class_labels[i] for i in sorted_unseen_inds]
    for inds, lbls, name in [(sorted_inds, labels, 'stats'),
                             (sorted_unseen_inds, zs_labels, 'zs_stats')]:
        max_n = 120
        num_subplots = np.ceil(len(inds) / max_n).astype(np.int)
        for j in range(num_subplots):
            inds_j = inds[j * max_n: (j + 1) * max_n]
            lbls_j = lbls[j * max_n: (j + 1) * max_n]
            name_j = f'{name}{j if num_subplots > 1 else ""}'

            plot_mat(stat_mat[inds_j, :].T, lbls_j, conf_mat_labels + ['Rec', 'Prec'],
                     y_inds=np.arange(6), x_inds=inds_j, plot=False, alternate_labels=False, cbar=False, zero_color=[0.8, 0, 0.8, 1])
            plt.savefig(os.path.join(res_save_path, f'{args.type}_{name_j}.png'), dpi=300, bbox_inches='tight')
            print(f'Avg recall: {100 * np.mean(recall):.3f}%')
            print(f'Avg precision: {100 * np.mean(precision):.3f}%')

    if args.show:
        plt.show()


def print_predictions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--hoi_thr', type=float, default=0.05)
    parser.add_argument('--obj_thr', type=float, default=0.1)
    parser.add_argument('--part_thr', type=float, default=0.05)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--null', action='store_true', default=False)
    parser.add_argument('--part', action='store_true', default=False)
    parser.add_argument('--rnd', action='store_true', default=False)
    namespace = parser.parse_known_args()
    args = namespace[0]
    sys.argv = sys.argv[:1] + namespace[1]

    results, res_split = _setup_and_load()
    hh = HicoHake()
    hds_fns = hh.split_filenames[res_split]

    assert cfg.seenf >= 0
    inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
    obj_inds = sorted(inds_dict['train']['obj'].tolist())
    act_inds = sorted(inds_dict['train']['act'].tolist())

    print(f'Train objects ({len(obj_inds)}):', [hh.objects[i] for i in obj_inds])
    print(f'Train actions ({len(act_inds)}):', [hh.actions[i] for i in act_inds])
    seen_objs = set(obj_inds)
    seen_acts = set(act_inds)

    gt_part_action_labels = hh.split_part_annotations[res_split]
    gt_part_action_labels[gt_part_action_labels < 0] = 0
    gt_labels = hh.split_labels[res_split]

    n = args.num_imgs
    all_inds = list(range(n if n > 0 else len(hds_fns)))
    if args.rnd:
        seed = np.random.randint(1_000_000_000)
        print('Seed:', seed)
        np.random.seed(seed)
        np.random.shuffle(all_inds)
    for idx in all_inds:
        im_fn = hds_fns[idx]
        img_id = int(im_fn.split('_')[-1].split('.')[0])
        # if img_id != 14:
        #     continue
        prediction_dict = results[idx]
        prediction = Prediction(prediction_dict)

        print(f'{img_id:5d}')
        # Objects
        if prediction.obj_scores is not None:
            obj_anns = (gt_labels[idx, :] @ hh.interaction_to_object_mat) > 0
            obj_scores = np.squeeze(prediction.obj_scores, axis=0)
            inds = obj_scores.argsort()[::-1]
            obj_pred_str = []
            obj_misses_str = []
            for i, s in zip(inds, obj_scores[inds]):
                str_o = ('*' if i not in seen_objs else '') + hh.objects[i]
                if s >= args.obj_thr:
                    obj_pred_str.append(f'{str_o} ({s * 100:.2f}{"#" if obj_anns[i] else ""})')
                else:
                    if obj_anns[i]:
                        obj_misses_str.append(f'{str_o} ({obj_scores[i] * 100:.2f})')
            print(f'{"":5s} {"OBJ":10s} PRED {", ".join(obj_pred_str)}')
            if obj_misses_str:
                print(f'{"":5s} {"":10s} MISS {", ".join(obj_misses_str)}')

        # Interactions
        hoi_anns = (gt_labels[idx, :] > 0)
        hoi_scores = np.squeeze(prediction.hoi_scores, axis=0)
        inds = hoi_scores.argsort()[::-1]
        hoi_pred_str = []
        hoi_misses_str = []
        for i, s in zip(inds, hoi_scores[inds]):
            act_ind = hh.interactions[i, 0]
            obj_ind = hh.interactions[i, 1]
            act_str = ('*' if act_ind not in seen_acts else '') + hh.actions[act_ind]
            obj_str = ('*' if obj_ind not in seen_objs else '') + hh.objects[obj_ind]
            if args.null or act_ind > 0:
                if s >= args.hoi_thr:
                    hoi_pred_str.append(f'{act_str} {obj_str} ({s * 100:.2f}{"#" if hoi_anns[i] else ""})')
                else:
                    if hoi_anns[i]:
                        hoi_misses_str.append(f'{act_str} {obj_str} ({hoi_scores[i] * 100:.2f})')
        print(f'{"":5s} {"HOI":10s} PRED {", ".join(hoi_pred_str)}')
        if hoi_misses_str:
            print(f'{"":5s} {"":10s} MISS {", ".join(hoi_misses_str)}')

        # Parts
        if args.part:
            pred_i = prediction.part_state_scores.squeeze()
            gt_i = gt_part_action_labels[idx, :]
            for j, acts in enumerate(hh.states_per_part):
                part_pred_str = []
                part_miss_str = []
                pred_ij = pred_i[acts]
                inds = np.argsort(pred_ij)[::-1]
                for a, s in zip(acts[inds], pred_ij[inds]):
                    str_a = hh.bp_ps_pairs[a][1]
                    if args.null or str_a != hh.null_action:
                        str_a = 'NULL' if str_a == hh.null_action else str_a
                        if s >= args.part_thr:  # predicted
                            part_pred_str.append(f'{str_a} ({s * 100 :.2f}{"#" if gt_i[a] else ""})')
                        else:  # unpredicted
                            if gt_i[a]:  # miss
                                part_miss_str.append(f'{str_a} ({s * 100 :.2f})')
                if part_pred_str:
                    print(f'{"":5s} {"P:" + hh.parts[j].upper():10s} PRED {", ".join(part_pred_str)}')
                if part_miss_str:
                    print(f'{"":5s} {"":10s} MISS {", ".join(part_miss_str)}')

        # Visualise
        if args.vis:
            img = Image.open(hh.get_img_path(res_split, im_fn))
            visualizer = Visualizer(img)
            vis_output = visualizer.output
            plt.imshow(vis_output.get_image())
            plt.show()


def save_pstate_hoi_att():
    if torch.cuda.is_available():
        def func(model):
            model.eval()
            pstate_att, _ = model.branches['hoi']._get_pstate_att()
            return pstate_att.detach().cpu().numpy()

        run_and_save(func=func, fname='pstate-hoi_att_coeffs')
    else:
        _setup_and_load()
        hh = HicoHake()
        part_state_strs = [f'{p} {a}'
                           for p, a in hh.bp_ps_pairs
                           if a != hh.null_action
                           ]

        pstate_att = np.load(os.path.join(cfg.output_analysis_path, 'pstate-hoi_att_coeffs.npy'))

        # for i in range(6):
        #     plot_mat(pstate_att[:, i * 100:(i + 1) * 100],
        #              xticklabels=hh.interactions_str[i * 100:(i + 1) * 100],
        #              yticklabels=part_state_strs,
        #              zero_color=[1, 1, 1],
        #              vrange=(0, 1))

        assert cfg.seenf >= 0
        inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
        obj_inds = np.array(sorted(inds_dict['train']['obj'].tolist()))
        act_inds = np.array(sorted(inds_dict['train']['act'].tolist()))
        seen_interactions = np.setdiff1d(np.unique(hh.oa_pair_to_interaction[obj_inds, :][:, act_inds]), -1)

        # GT
        split = 'train'
        inters = hh.split_labels[split]
        part_states = hh.split_part_annotations[split]
        fg_part_states = np.concatenate([inds[:-1] for inds in hh.states_per_part])
        pstate_inters_cooccs = part_states[:, fg_part_states].T @ inters
        pstate_inters_cooccs /= np.maximum(1, pstate_inters_cooccs.sum(axis=0, keepdims=True))

        # Diff
        diff = pstate_inters_cooccs - pstate_att
        mse = np.sqrt((diff ** 2).sum(axis=0))
        sorted_interactions = np.argsort(mse)[::-1]

        unseen_inds = np.setdiff1d(sorted_interactions, seen_interactions, assume_unique=True)
        seen_inds = np.setdiff1d(sorted_interactions, unseen_inds, assume_unique=True)
        for inds in [seen_inds, unseen_inds]:
            print('#' * 100)
            for i in inds:
                abs_diff_i = np.abs(diff[:, i])
                js = np.flatnonzero(abs_diff_i > 0.01)
                inds_i = np.argsort(abs_diff_i[js])[::-1]
                js = js[inds_i]
                part_str = ', '.join([f'{part_state_strs[j]} ({diff[j, i] * 100:.2f}%)' for j in js])
                print(f'{hh.interactions_str[i]:>35s}: {part_str}')

        plot_mat(np.stack([np.mean(pstate_inters_cooccs, axis=1),
                           np.mean(pstate_att, axis=1),
                           np.mean(np.abs(diff), axis=1)
                           ], axis=0),
                 xticklabels=part_state_strs,
                 yticklabels=['GT', 'Pred', 'Diff'],
                 zero_color=[1, 1, 1],
                 alternate_labels=False,
                 vrange=(0, 1))
        print('Done.')


def main():
    funcs = {
        'stats': stats,
        'ppred': print_predictions,
        'ps-h_att': save_pstate_hoi_att,
    }
    analysis_hub(funcs)


if __name__ == '__main__':
    main()

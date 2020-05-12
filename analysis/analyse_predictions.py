import pickle
import sys

import matplotlib

try:
    matplotlib.use('Qt5Agg')

    sys.argv[1:] = ['act']
    # sys.argv[1:] = ['part']
    # sys.argv[1:] = ['part_dist']
except ImportError:
    pass

import numpy as np
from matplotlib import pyplot as plt

from config import cfg
from analysis.utils import analysis_hub, plot_mat
from lib.dataset.vcoco import VCoco, VCocoSplit
from lib.dataset.hicodet_hake import HicoDetHake, HicoDetHakeSplit
from lib.models.abstract_model import Prediction
from lib.eval.evaluator_vcoco_roi import EvaluatorVCocoROI
from lib.eval.evaluator_hicodethake_pstate import EvaluatorHicoDetHakePartROI


def _setup_and_load():
    cfg.parse_args(fail_if_missing=False, reset=True)

    with open(cfg.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results, cfg.eval_split


def part_dist_model():
    exp = 'output/part/hico/standard/2020-04-09_12-02-27_SINGLE'
    split = 'train'
    sys.argv[1:] = ['--save_dir', exp, '--eval_on', split]
    predictions, _ = _setup_and_load()

    hh = HicoDetHake()
    evaluator = EvaluatorHicoDetHakePartROI(dataset_split=HicoDetHakeSplit(split=split, full_dataset=hh))
    evaluator.evaluate_predictions(predictions=predictions)

    pred_scores = np.concatenate(evaluator.predict_scores, axis=0)
    gt_labels = np.concatenate(evaluator.all_gt_labels, axis=0)
    pred_assignment = np.concatenate(evaluator.pred_gt_assignment_per_ho_pair, axis=0)

    part = 1
    symstates_p = hh.symstates_per_sympart[part]
    S = symstates_p.size
    fig, axs = plt.subplots(int(np.ceil(np.sqrt(S))), int(np.floor(np.sqrt(S))), figsize=(22, 12), sharex='col')
    for i, syms in enumerate(symstates_p):
        neg, pos = [], []
        for s in hh.symstates_inds[syms]:
            print(hh.symstates[syms], hh.states[s])
            preds_s = pred_scores[:, s]
            pred_assignment_s = pred_assignment[:, s]
            labels_s = np.zeros_like(preds_s).astype(np.bool)
            labels_s[(pred_assignment_s >= 0) & (gt_labels[pred_assignment_s] == s)] = 1
            neg.append(preds_s[~labels_s])
            pos.append(preds_s[labels_s])
        neg = np.concatenate(neg)
        pos = np.concatenate(pos)

        ax = axs[i % axs.shape[0], i // axs.shape[0]]
        ax.hist(neg, bins=100, range=(0, 1), log=True, label='Neg', histtype='step')
        ax.hist(pos, bins=100, range=(0, 1), log=True, label='Pos', histtype='step')
        ax.legend()
        ax.set_title(hh.symstates[syms])

    plt.show()

    print('Done.')


def part_model():
    exp = 'output/part/hico/standard/2020-04-09_12-02-27_SINGLE'
    sys.argv[1:] = ['--save_dir', exp]
    predictions, split = _setup_and_load()

    hh = HicoDetHake()

    evaluator = EvaluatorHicoDetHakePartROI(dataset_split=HicoDetHakeSplit(split=split, full_dataset=hh))
    evaluator.evaluate_predictions(predictions=predictions)
    evaluator.compute_metrics()

    to_plot = evaluator.metrics['M-mAP'][:, None]
    to_plot = np.concatenate([to_plot,
                              -np.ones((1, to_plot.shape[1])),
                              to_plot.mean(axis=0, keepdims=True)
                              ], axis=0)
    plot_mat(to_plot.T,
             xticklabels=hh.states + ['Average'],
             yticklabels=['mAP'],
             neg_color=[1, 1, 1],
             zero_color=[1, 0, 1],
             alternate_labels=False,
             vrange=(0, 1),
             annotate=max(to_plot.shape) <= 60,
             figsize=(22, 12),
             )
    print('Done.')


def act_model():
    def _branch_att_per_action(_extra_info, _pred_gt_ho_assignment, _gt_labels, _ds):
        assert len(_extra_info) == 1
        branch_att = _extra_info[list(_extra_info.keys())[0]]
        print(len(branch_att))

        branch_att_per_act = {}
        for b_att, gt_assignment in zip(branch_att, _pred_gt_ho_assignment):
            for i, gt_idx in enumerate(gt_assignment):
                if gt_idx >= 0:
                    assert _gt_labels[gt_idx] == i
                    branch_att_per_act.setdefault(i, []).append(b_att)
        assert not set(branch_att_per_act.keys()) - set(range(_ds.num_actions))
        branch_att_per_act = [np.stack(branch_att_per_act[i], axis=0) if i in branch_att_per_act else np.zeros((0, 2, 2))
                              for i in range(_ds.num_actions)]
        print([v.shape[0] for v in branch_att_per_act])
        assert all([np.allclose(np.sum(x, axis=2), 1) for x in branch_att_per_act])

        aggr_branch_att_per_act = np.stack([x.mean(axis=0) for x in branch_att_per_act], axis=0)
        assert np.allclose(aggr_branch_att_per_act[np.any(~np.isnan(aggr_branch_att_per_act), axis=(1, 2)), :, :].sum(axis=2), 1)
        aggr_branch_att_per_act[np.isnan(aggr_branch_att_per_act)] = 0
        aggr_dir_branch_att_per_act = aggr_branch_att_per_act[:, 0, :]
        return aggr_dir_branch_att_per_act

    def _pstate_att_per_action(_extra_info, _pred_gt_ho_assignment, _gt_labels, _ds):
        pstate_att = _extra_info['act__pstate_att']
        print(len(pstate_att))

        pstate_att_per_act = {}
        for b_att, gt_assignment in zip(pstate_att, _pred_gt_ho_assignment):
            for i, gt_idx in enumerate(gt_assignment):
                if gt_idx >= 0:
                    assert _gt_labels[gt_idx] == i
                    pstate_att_per_act.setdefault(i, []).append(b_att)
        assert not set(pstate_att_per_act.keys()) - set(range(_ds.num_actions))
        pstate_att_per_act = [np.stack(pstate_att_per_act[i], axis=0) if i in pstate_att_per_act else np.zeros((0, hh.num_states))
                              for i in range(_ds.num_actions)]
        print([v.shape[0] for v in pstate_att_per_act])
        assert all([np.allclose(np.sum(x, axis=1), 1) for x in pstate_att_per_act])

        aggr_pstate_att_per_act = np.stack([x.mean(axis=0) for x in pstate_att_per_act], axis=0)
        assert np.allclose(aggr_pstate_att_per_act[np.any(~np.isnan(aggr_pstate_att_per_act), axis=1), :].sum(axis=1), 1)
        aggr_pstate_att_per_act[np.isnan(aggr_pstate_att_per_act)] = 0
        return aggr_pstate_att_per_act

    def _part_att_per_action(_extra_info, _pred_gt_ho_assignment, _gt_labels, _ds):
        part_att = _extra_info['act__part_att']
        print(len(part_att))

        part_att_per_act = {}
        for b_att, gt_assignment in zip(part_att, _pred_gt_ho_assignment):
            for i, gt_idx in enumerate(gt_assignment):
                if gt_idx >= 0:
                    assert _gt_labels[gt_idx] == i
                    part_att_per_act.setdefault(i, []).append(b_att)
        assert not set(part_att_per_act.keys()) - set(range(_ds.num_actions))
        part_att_per_act = [np.stack(part_att_per_act[i], axis=0) if i in part_att_per_act else np.zeros((0, hh.num_parts))
                            for i in range(_ds.num_actions)]
        print([v.shape[0] for v in part_att_per_act])
        assert all([np.allclose(np.sum(x, axis=1), 1) for x in part_att_per_act])

        aggr_part_att_per_act = np.stack([x.mean(axis=0) for x in part_att_per_act], axis=0)
        assert np.allclose(aggr_part_att_per_act[np.any(~np.isnan(aggr_part_att_per_act), axis=1), :].sum(axis=1), 1)
        aggr_part_att_per_act[np.isnan(aggr_part_att_per_act)] = 0
        return aggr_part_att_per_act

    def _pstate_weights(_extra_info, _pred_gt_ho_assignment, _gt_labels, _ds):
        pstate_weights = _extra_info['act__state_weights']
        print(len(pstate_weights))

        pstate_weights_per_act = {}
        for ps_w, gt_assignment in zip(pstate_weights, _pred_gt_ho_assignment):
            for i, gt_idx in enumerate(gt_assignment):
                if gt_idx >= 0:
                    assert _gt_labels[gt_idx] == i
                    pstate_weights_per_act.setdefault(i, []).append(ps_w[i, :])
        assert not set(pstate_weights_per_act.keys()) - set(range(_ds.num_actions))
        pstate_weights_per_act = [np.stack(pstate_weights_per_act[i], axis=0) if i in pstate_weights_per_act else None
                                  for i in range(_ds.num_actions)]
        shape1 = list({x.shape[1] for x in pstate_weights_per_act if x is not None})
        assert len(shape1) == 1
        pstate_weights_per_act = [x if x is not None else np.zeros((0, shape1[0])) for x in pstate_weights_per_act]
        print([x.shape[0] for x in pstate_weights_per_act])

        aggr_part_att_per_act = np.stack([x.mean(axis=0) for x in pstate_weights_per_act], axis=0)
        aggr_part_att_per_act[np.isnan(aggr_part_att_per_act)] = 0
        return aggr_part_att_per_act

    exps = [
        # 'output/act/vcoco_nozs/nopart/2020-04-16_15-25-24_SINGLE',
        # 'output/frompstate/vcoco_nozs/pbf/2020-04-21_11-13-05_SINGLE',
        'output/act/vcoco_nozs/pbf/2020-05-01_10-53-43_SINGLE',
        'output/logic/vcoco_nozs/pbf/2020-05-01_11-05-44_SINGLE',

        # 'output/act/vcoco_zs1/pbf_awsu1_oracle/2020-04-15_16-52-04_SINGLE',
        'output/act/vcoco_zs1/pbf_awsu1/2020-04-28_16-28-36_SINGLE',

        # 'output/pgcnact/vcoco_zs1/pbf_awsu1/2020-04-29_09-59-45_SINGLE',

        'output/late/vcoco_zs1/pbf_awsu1/2020-04-30_16-48-07_SINGLE',

        'output/lateatt/vcoco_zs1/pbf_awsu1/2020-04-30_17-51-42_SINGLE',

        'output/logic/vcoco_zs1/pbf_awsu1/2020-04-28_17-44-43_SINGLE',


    ]
    to_plot = []
    measure_labels = []

    hh = HicoDetHake()

    ds = VCoco()
    evaluator = EvaluatorVCocoROI(dataset_split=VCocoSplit(split='test', full_dataset=ds))
    seen_acts = None
    for exp in exps:
        sys.argv[1:] = ['--save_dir', exp]

        predictions, _split = _setup_and_load()
        if cfg.seenf >= 0:
            _seen_acts = pickle.load(open(cfg.seen_classes_file, 'rb'))['train']['act']
            assert seen_acts is None or np.all(seen_acts == _seen_acts)
            seen_acts = _seen_acts

        # Get prediction assignments
        evaluator.evaluate_predictions(predictions=predictions)
        gt_labels = np.concatenate(evaluator.gt_labels)
        pred_gt_ho_assignment = np.concatenate(evaluator.pred_gt_assignment_per_hoi, axis=0)
        print(gt_labels.size, pred_gt_ho_assignment.shape)

        # Get mAP
        evaluator.compute_metrics()
        mAP = evaluator.metrics['M-mAP']
        exp_to_plot = [mAP[:, None]]
        exp_ax_labels = ['mAP']

        extra_info = {}
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            einfo = prediction.extra_info
            if einfo is None:
                continue
            for k, v in einfo.items():
                extra_info.setdefault(k, []).append(v)

        # Get the attention coefficients per action
        try:
            raise KeyError
            if extra_info:

                extra_info = {k: np.concatenate(v, axis=0) for k, v in extra_info.items()}
                if cfg.model == 'att':
                    ei_ind = 2
                elif cfg.model == 'pgcnact':
                    ei_ind = 3
                else:
                    ei_ind = None

                if ei_ind == 0:
                    aggr_dir_branch_att_per_act = _branch_att_per_action(extra_info, pred_gt_ho_assignment, gt_labels, ds)
                    exp_to_plot.append(aggr_dir_branch_att_per_act)
                    exp_ax_labels.extend(['Vis', 'Part'])
                elif ei_ind == 1:
                    aggr_pstate_att_per_act = _pstate_att_per_action(extra_info, pred_gt_ho_assignment, gt_labels, ds)
                    exp_to_plot.append(aggr_pstate_att_per_act)
                    exp_ax_labels.extend(hh.states)
                elif ei_ind == 2:
                    aggr_part_att_per_act = _part_att_per_action(extra_info, pred_gt_ho_assignment, gt_labels, ds)
                    exp_to_plot.append(aggr_part_att_per_act)
                    exp_ax_labels.extend(hh.parts)
                elif ei_ind == 3:
                    aggr_psw_per_act = _pstate_weights(extra_info, pred_gt_ho_assignment, gt_labels, ds)
                    exp_to_plot.append(aggr_psw_per_act)
                    exp_ax_labels.extend(hh.symstates)
        except KeyError:
            print('No suitable extra info!')

        to_plot.extend(exp_to_plot + [-np.ones((ds.num_actions, 1))])
        measure_labels.extend(exp_ax_labels + [''])

    to_plot = np.concatenate(to_plot[:-1], axis=1)
    to_plot_ext = to_plot
    measure_labels = measure_labels[:-1]

    y_labels = ['', 'Average']
    to_plot_ext = np.concatenate([to_plot_ext,
                                  -np.ones((1, to_plot_ext.shape[1])),
                                  to_plot_ext.mean(axis=0, keepdims=True)
                                  ], axis=0)

    # Plot
    if seen_acts is None:
        act_labels = ds.actions
    else:
        act_labels = [f'{"" if i in seen_acts else "*"}{x}' for i, x in enumerate(ds.actions)]
        unseen_acts = np.setdiff1d(np.arange(ds.num_actions), seen_acts)
        to_plot_ext = np.concatenate([to_plot_ext,
                                      to_plot[unseen_acts, :].mean(axis=0, keepdims=True)
                                      ], axis=0)
        y_labels += ['Average unseen']
    y_labels = act_labels + y_labels

    plot_mat(to_plot_ext,
             xticklabels=measure_labels,
             yticklabels=y_labels,
             neg_color=[1, 0, 1],
             zero_color=[1, 1, 1],
             alternate_labels=False,
             vrange=(0, 1),
             annotate=max(to_plot_ext.shape) <= 60,
             figsize=(22, 12),
             )
    print('Done.')


def main():
    funcs = {
        'act': act_model,
        'part': part_model,
        'part_dist': part_dist_model,
    }
    analysis_hub(funcs)


if __name__ == '__main__':
    main()

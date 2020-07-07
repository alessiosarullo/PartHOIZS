import pickle
import sys
import os

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
from lib.eval.vsrl_eval import VCOCOeval, pkl_from_predictions


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
    def _w_per_action(_weights, _pred_gt_ho_assignment, _gt_labels, _ds):
        print(len(_weights))

        _weights_per_act = {}
        for ps_w, gt_assignment in zip(_weights, _pred_gt_ho_assignment):
            for i, gt_idx in enumerate(gt_assignment):
                if gt_idx >= 0:
                    assert _gt_labels[gt_idx] == i
                    _weights_per_act.setdefault(i, []).append(ps_w)
        assert not set(_weights_per_act.keys()) - set(range(_ds.num_actions))
        _weights_per_act = [np.stack(_weights_per_act[i], axis=0) if i in _weights_per_act else None
                            for i in range(_ds.num_actions)]
        shape1 = list({x.shape[1] for x in _weights_per_act if x is not None})
        assert len(shape1) == 1
        _weights_per_act = [x if x is not None else np.zeros((0, shape1[0])) for x in _weights_per_act]
        print([x.shape[0] for x in _weights_per_act])

        aggr_w_per_act = np.stack([x.mean(axis=0) for x in _weights_per_act], axis=0)
        aggr_w_per_act[np.isnan(aggr_w_per_act)] = 0
        return aggr_w_per_act

    exps = [
        # 'output/act/vcoco_nozs/nopart/2020-04-16_15-25-24_SINGLE',
        # 'output/frompstate/vcoco_nozs/pbf/2020-04-21_11-13-05_SINGLE',
        # 'output/act/vcoco_nozs/pbf/2020-05-01_10-53-43_SINGLE',
        # 'output/logic/vcoco_nozs/pbf/2020-05-01_11-05-44_SINGLE',

        'output/vcoco/act/zs1_nopart/noawsu/2020-06-10_16-55-00_SINGLE',
        'output/vcoco/act/zs1_pbf/noawsu/2020-06-10_16-38-37_SINGLE',
        'output/vcoco/logic/zs1_pbf/noawsu/2020-06-10_16-10-25_SINGLE',

        # 'output/vcoco/act/zs1_nopart/awsu1/2020-06-03_16-47-26_SINGLE',
        # 'output/vcoco/act/zs1_pbf/awsu1/2020-06-03_15-50-53_SINGLE',
        # 'output/vcoco/logic/zs1_pbf/awsu1_nopsf/2020-06-11_16-12-15_SINGLE',
        'output/vcoco/logic/zs1_pbf/awsu1/2020-06-03_16-00-59_SINGLE',

    ]
    to_plot = []
    measure_labels = []
    alternate_labels = False

    hh = HicoDetHake()

    use_vsrl_eval = False

    ds = VCoco()
    ds_split = VCocoSplit(split='test', full_dataset=ds)
    if use_vsrl_eval:
        evaluator = VCOCOeval(vsrl_annot_file=os.path.join(cfg.data_root, 'V-COCO', 'vcoco', 'vcoco_test.json'),
                              coco_annot_file=os.path.join(cfg.data_root, 'V-COCO', 'instances_vcoco_all_2014.json'),
                              split_file=os.path.join(cfg.data_root, 'V-COCO', 'splits', 'vcoco_test.ids')
                              )
    else:
        evaluator = EvaluatorVCocoROI(dataset_split=ds_split)
    seen_acts = None
    for exp in exps:
        sys.argv[1:] = ['--save_dir', exp]

        predictions, _split = _setup_and_load()
        if cfg.seenf >= 0:
            _seen_acts = pickle.load(open(cfg.seen_classes_file, 'rb'))['train']['act']
            assert seen_acts is None or np.all(seen_acts == _seen_acts)
            seen_acts = _seen_acts

        if use_vsrl_eval:
            det_file = os.path.join(cfg.output_path, 'vcoco_pred.pkl')
            pkl_from_predictions(dict_predictions=predictions, dataset=ds_split, filename=det_file)
            seen_acts_str = [ds_split.actions[i] for i in seen_acts] if seen_acts is not None else None
            map1, map2, actions_with_role = evaluator._do_eval(det_file, ovr_thresh=0.5, seen_acts_str=seen_acts_str)
            map1_inv = {ar: map1[i][j] for i, a_roles in enumerate(actions_with_role) for j, ar in enumerate(a_roles) if ar is not None}
            mAP = np.array([map1_inv.get(a, 0) for a in ds.actions])
        else:
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
            # raise KeyError
            if extra_info:

                extra_info = {k: np.concatenate(v, axis=0) for k, v in extra_info.items()}
                if cfg.model == 'att':
                    ei_ind = 1
                    # extra_info['act__mha_weights'] = extra_info['act__mha_weights'].mean(axis=1)
                elif cfg.model == 'lateatt':
                    ei_ind = 0
                else:
                    ei_ind = None

                if ei_ind == 0:
                    aggr_featatt_per_act = _w_per_action(extra_info['act__mha_weights'], pred_gt_ho_assignment, gt_labels, ds)
                    exp_to_plot.append(aggr_featatt_per_act)
                    exp_ax_labels.extend(['Img', 'Ex', 'Obj scores', 'State scores'])
                elif ei_ind == 1:
                    aggr_featatt_per_act = _w_per_action(extra_info['act__att_w'], pred_gt_ho_assignment, gt_labels, ds)
                    exp_to_plot.append(aggr_featatt_per_act)
                    exp_ax_labels.extend(['Img', 'Ex'] + hh.objects + hh.objects + hh.symstates)
                    alternate_labels = 'x'
        except KeyError:
            print('No suitable extra info!')

        to_plot.extend(exp_to_plot + [-np.ones((ds.num_actions, 1))])
        measure_labels.extend(exp_ax_labels + [''])

    to_plot = np.concatenate(to_plot[:-1], axis=1)
    to_plot_ext = to_plot
    measure_labels = measure_labels[:-1]

    y_labels = ['', 'Average all']
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

    print('\n'.join([f'{all*100:.2f} & {unseen*100:.2f}' for all, unseen in to_plot_ext[-2:, :].T if all >= 0 or unseen >= 0]))
    if True:  # plot to report:
        keep_along_x = np.flatnonzero(np.any(to_plot_ext > 0, axis=0))
        keep_along_y = np.flatnonzero(np.any(to_plot_ext > 0, axis=1))
        to_plot_ext = to_plot_ext[keep_along_y, :]
        to_plot_ext = to_plot_ext[:, keep_along_x]
        to_plot_ext = to_plot_ext.T
        act_labels = [y_labels[i] for i in keep_along_y]

        # Remove average
        act_labels = act_labels[:-2]
        to_plot_ext = to_plot_ext[:, :-2]

        exp_labels = []
        for i, e in enumerate(exps):
            s = 'Base'
            if 'pbf' in e:
                if 'nopsf' not in e:
                    s += '+P'
            else:
                assert 'nopart' in e
            if 'noawsu' not in e:
                assert 'awsu' in e
                s += '+A'
            if 'logic' in e:
                s += '+L'
            exp_labels.append(s)

        plot_mat(to_plot_ext,
                 xticklabels=act_labels,
                 yticklabels=exp_labels,
                 x_inds=[''] * len(act_labels),
                 y_inds=[''] * len(measure_labels),
                 alternate_labels=False,
                 vrange=(0, 1),
                 fix_cbar_height=True,
                 figsize=(22, 8),
                 fsize=28,
                 annotate=True,
                 ann_fsize=28,
                 perc=True,
                 plot=False
                 )
        filename = '_vs_'.join([e.lower().replace('base', '').strip('+_').replace('+', '-') for e in exp_labels])
        plt.savefig(f'/home/alex/Dropbox/PhD Docs/My stuff/Thesis/images/{filename}.png',
                    dpi=400, bbox_inches='tight', pad_inches=0, format='png')
        # plt.show()
    else:
        plot_mat(to_plot_ext,
                 xticklabels=measure_labels,
                 yticklabels=y_labels,
                 neg_color=[1, 0, 1],
                 zero_color=[1, 1, 1],
                 alternate_labels=alternate_labels,
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

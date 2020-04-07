from collections import Counter
from typing import List, Dict

import numpy as np


class Evaluator:
    def evaluate_predictions(self, predictions: List[Dict], **kwargs):
        raise NotImplementedError

    def output_metrics(self, to_keep=None, compute_pos=True, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class MetricFormatter:
    def __init__(self):
        super().__init__()

    def format_metric_and_gt_lines(self, gt_label_hist, metrics, class_inds, gt_str=None, verbose=False):
        num_gt = sum(gt_label_hist.values())

        pad = len(gt_str) if gt_str is not None else 0
        if metrics:
            pad = max(pad, max([len(k) for k in metrics.keys()]))

        p = 2
        lines = []
        for k, v in metrics.items():
            lines += [self.format_metric(k, v, metric_str_len=pad, verbose=verbose, precision=p)]
        format_str = '%{}s %{}s'.format(pad + 1, p + 8)
        if gt_str is not None and verbose:
            l1 = format_str % ('%s:' % gt_str, 'IDs')
            l1 += '[' + ' '.join([('%{:d}d '.format(p + 5)) % i for i in class_inds]) + ']'
            l2 = format_str % ('', '%')
            l2 += '[' + ' '.join([self._format_percentage(gt_label_hist[i] / num_gt, precision=p) for i in class_inds]) + ']'
            lines += [l1, l2]

        print('\n'.join(lines))

    def format_metric(self, metric_name, data, metric_str_len=None, verbose=False, **kwargs):
        metric_str_len = metric_str_len or len(metric_name)
        if verbose:
            per_class_str = ' @ [%s]' % ' '.join([self._format_percentage(x, **kwargs) for x in data]) if data.size > 1 else ''
        else:
            per_class_str = ''
        f_str = ('%{}s: %s%s'.format(metric_str_len)) % (metric_name, self._format_percentage(np.mean(data), **kwargs), per_class_str)
        return f_str

    @staticmethod
    def _format_percentage(value, precision=2):
        if -1 < value < 1:
            if value != 0:
                return ('% {}.{}f%%'.format(precision + 5, precision)) % (value * 100)
            else:
                return ('% {}.{}f%%'.format(precision + 5, 0)) % (value * 100)
        else:
            return ('% {}d%%'.format(precision + 5)) % (100 * np.sign(value))


def sort_and_filter(metrics, gt_labels, all_classes, sort=False, keep_inds=None, metric_prefix=''):
    gt_labels_hist = Counter(gt_labels)
    for c in all_classes:
        gt_labels_hist.setdefault(c, 0)

    if keep_inds:
        del_inds = set(gt_labels_hist.keys()) - set(keep_inds)
        for k in del_inds:
            del gt_labels_hist[k]

    if sort:
        class_inds = [p for p, num in gt_labels_hist.most_common()]
    else:
        class_inds = sorted(gt_labels_hist.keys())

    metrics = {metric_prefix + k: v[class_inds] if v.size > 1 else v for k, v in metrics.items()}
    return gt_labels_hist, metrics, class_inds

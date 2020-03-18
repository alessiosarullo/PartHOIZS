import argparse
import os

import numpy as np
import scipy.stats

from scripts.utils import get_runs_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('measure')
    parser.add_argument('baseline', type=float)
    args = parser.parse_args()
    dir = args.dir
    measure = args.measure
    baseline = args.baseline

    print(dir)
    runs = []
    for run_dir in os.listdir(dir):
        if 'RUN' in run_dir:
            runs.append(os.path.join(dir, run_dir))
    runs = sorted(runs)

    exp_data = get_runs_data(runs)

    # Result obtained at the lowest validation action loss.
    assert np.all(exp_data['Val']['steps'] == exp_data['Test']['steps'])
    best_val_loss_step_per_run = np.argmin(exp_data['Val']['values']['Act_loss'], axis=1)
    test_data = exp_data['Test']['values'][measure]
    test_accuracy_per_run = test_data[np.arange(test_data.shape[0]), best_val_loss_step_per_run]
    sp = max([len(r) for r in runs])
    for i, r in enumerate(runs):
        print(f'{r:{sp}s} {test_accuracy_per_run[i]:8.5f}')
    print(f'{"Mean":>{sp}s} {np.mean(test_accuracy_per_run):8.5f}')
    print(f'{"Std":>{sp}s} {np.std(test_accuracy_per_run):8.5f}')

    # Welch’s t-test
    baseline = np.atleast_1d(baseline)
    results = test_accuracy_per_run
    if baseline.shape[0] == 1:
        pvalue = scipy.stats.ttest_1samp(results, popmean=baseline[0])[1]
    else:
        pvalue = scipy.stats.ttest_ind(baseline, results, equal_var=False)[1]
    print(f'{measure:>15s}')
    print(f'p = {pvalue:11.2e}')


if __name__ == '__main__':
    main()

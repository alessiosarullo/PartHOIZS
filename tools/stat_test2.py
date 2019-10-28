import argparse
import os

import numpy as np
import scipy.stats

from lib.utils import get_runs_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument('measure')
    args = parser.parse_args()
    dir1 = args.dir1
    dir2 = args.dir2
    measure = args.measure

    print(dir1, dir2)
    runs = []
    for wd in [dir1, dir2]:
        runs_wd = []
        for run_dir in os.listdir(wd):
            if 'RUN' in run_dir:
                runs_wd.append(os.path.join(wd, run_dir))
        runs_wd = sorted(runs_wd)
        runs.append(runs_wd)

    all_exp_data = [get_runs_data(run) for run in runs]

    # Result obtained at the lowest validation action loss.
    test_accs = []
    for exp_data in all_exp_data:
        test_data = exp_data['Test']['values'][measure]
        if np.all(exp_data['Val']['steps'] == exp_data['Test']['steps']):
            best_val_loss_step_per_run = np.argmin(exp_data['Val']['values']['Act_loss'], axis=1)
        else:
            best_val_loss_step_per_run = -np.ones_like(test_data.shape[0])
        test_accuracy_per_run = test_data[np.arange(test_data.shape[0]), best_val_loss_step_per_run]
        sp = max([len(r) for r in runs])
        print(f'{"Mean":>{sp}s} {np.mean(test_accuracy_per_run):8.5f}')
        print(f'{"Std":>{sp}s} {np.std(test_accuracy_per_run):8.5f}')
        test_accs.append(test_accuracy_per_run)

    # Welch’s t-test
    pvalue = scipy.stats.ttest_ind(test_accs[0], test_accs[1], equal_var=False)[1]
    print(f'{measure:>15s}')
    print(f'p = {pvalue:11.2e}')


if __name__ == '__main__':
    main()

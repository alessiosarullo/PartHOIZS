import argparse
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from scripts.utils import get_runs_data


def aggregate_tb_runs(dirname, runs):
    print(dirname)
    print(runs)
    os.makedirs(dirname, exist_ok=True)

    runs_data = get_runs_data(runs)
    values_per_tag = runs_data['Test']['values']
    steps = runs_data['Test']['steps']

    aggr_ops = {'mean': np.mean,
                'std': np.std}
    for aggr_op_name, aggr_op in aggr_ops.items():
        tblogger = SummaryWriter(os.path.join(dirname, 'tboard/Test', aggr_op_name))
        for tag, values in values_per_tag.items():
            aggr_value_per_timestep = aggr_op(values, axis=0)
            for i, aggr_value in enumerate(aggr_value_per_timestep):
                tblogger.add_scalar(tag, aggr_value, global_step=steps[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname')
    parser.add_argument('runs', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    dirname = args.dirname
    runs = args.runs

    # try:  # PyCharm debugging
    #     print('Starting remote debugging (resume from debug server)')
    #     import pydevd_pycharm
    #     pydevd_pycharm.settrace('130.88.195.105', port=16004, stdoutToServer=True, stderrToServer=True)
    #     print('Remote debugging activated.')
    # except:
    #     print('Remote debugging failed.')
    #     raise

    aggregate_tb_runs(dirname=dirname, runs=runs)


if __name__ == '__main__':
    main()

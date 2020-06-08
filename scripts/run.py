import argparse
import datetime
import os
import sys
import ast
import io

from scripts.launch import Launcher
from ax.service.ax_client import AxClient
import numpy as np
from config import cfg


class MyLogger:
    def __init__(self, log_file_handler: io.TextIOWrapper):
        self.terminal = sys.stdout
        self.log = log_file_handler

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


def run_single_trial(launcher, output_dir, dataset, net, num_runs, arg_fname, run_args):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    runs = []
    losses = []
    for idx in range(1, num_runs + 1):
        date_time_s = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if num_runs > 1:
            run_name = f'{date_time_s}_RUN{idx}'
        else:
            run_name = f'{date_time_s}_SINGLE'
        exp_dir = os.path.join(output_dir, run_name)
        log_file = os.path.join(exp_dir, 'log.txt')

        runs.append(exp_dir)
        os.makedirs(exp_dir)

        # noinspection PyTypeChecker
        log_file_handler = open(log_file, 'a')  # type: io.TextIOWrapper
        sys.stdout = MyLogger(log_file_handler=log_file_handler)
        sys.stderr = MyLogger(log_file_handler=log_file_handler)
        print(f'Logging {exp_dir} to {log_file}.')

        script_args = arg_fname + ['--model', net, '--ds', dataset, '--save_dir', exp_dir] + run_args
        if num_runs > 1:
            script_args += ['--randomize']
        print(script_args)
        sys.argv = script_args

        lowest_val_loss = launcher.run()
        losses.append(lowest_val_loss)

        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    if num_runs > 1:
        from scripts.tb_utils import aggregate_tb_runs
        date_time_s = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = os.path.join(output_dir, f'{date_time_s}_AGGR{num_runs}')
        os.makedirs(exp_dir)
        aggregate_tb_runs(fname=exp_dir, runs=runs)

    return np.nanmean(np.array(losses)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('net', type=str)
    parser.add_argument('exp_name', type=str)
    parser.add_argument('variant_name', type=str)
    parser.add_argument('num_runs', type=int)
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('--hpn', type=int, default=0)
    parser.add_argument('--hpopt', type=str, default='')

    namespace = parser.parse_known_args()
    args = namespace[0]
    dataset = args.dataset
    net = args.net
    exp_name = args.exp_name
    variant_name = args.variant_name
    num_runs = args.num_runs
    gpu_id = args.gpu_id
    hpn = args.hpn
    hpopt = args.hpopt

    # os.environ['CUDA_LAUNCH_BLOCKING'] = True  # uncomment this to debug CUDA errors
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    arg_fname = sys.argv[:1]
    run_args = namespace[1]

    exp_output_dir = os.path.join('output', dataset, net, exp_name)
    launcher = Launcher()
    if hpn > 0:
        def float_to_str(x):
            return f'{x:.3e}'.replace('.', '').replace('e-0', 'e-').replace('e-', 'e')

        hpopt = ast.literal_eval(hpopt)
        assert isinstance(hpopt, dict) and len(hpopt) > 0

        hyperparameters_infos = []
        for param, bounds in hpopt.items():
            assert isinstance(bounds, list) and len(bounds) == 2
            hyperparameters_infos.append({'name': param,
                                          'type': 'range',
                                          'bounds': bounds,
                                          'value_type': cfg.__getattribute__(param).__class__.__name__,
                                          'log_scale': (bounds[1] / bounds[0] >= 10)
                                          })

        ax_client = AxClient()
        ax_client.create_experiment(name=f'Model: {net}. Dataset: {dataset}. Exp: {exp_name}.',
                                    parameters=hyperparameters_infos,
                                    objective_name='Validation loss',
                                    minimize=True,
                                    )

        tried_parameters = {k: [] for k in hpopt.keys()}
        trials = []
        hp_fname = variant_name + f'__hp{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        for i in range(hpn):
            parameters, trial_index = ax_client.get_next_trial()
            trials.append(parameters)

            current_variant_name = hp_fname + f'__trial{i:03d}'
            current_run_args = [x for x in run_args] + ['--randomize']  # make sure it's copied
            for param, value in parameters.items():
                tried_parameters[param].append(value)
                current_run_args = current_run_args + [f'--{param}', str(value)]
                current_variant_name = current_variant_name + f'_{param}{float_to_str(value)}'
            mean_loss = run_single_trial(launcher=launcher,
                                         output_dir=os.path.join(exp_output_dir, current_variant_name),
                                         dataset=dataset, net=net, num_runs=num_runs, arg_fname=arg_fname, run_args=current_run_args)
            ax_client.complete_trial(trial_index=trial_index, raw_data=mean_loss)

        best_parameters, metrics = ax_client.get_best_parameters()  # note: metrics does not return the actual value for the best parameters
        hp_result_file_path = os.path.join(exp_output_dir, hp_fname)
        with open(hp_result_file_path + '.txt', 'w') as f:
            res_str = [ax_client.experiment.name]
            res_str += ['#' * 50, '# Parameters', '#' * 50]
            for param, bounds in hpopt.items():
                res_str.append('=' * 10 + '\n' +
                               f'Infos: {ax_client.experiment.parameters[param]}\n' +
                               f'Tried: {np.sort(tried_parameters[param])}\n' +
                               f'Best: {best_parameters[param]}')
            res_str += ['#' * 50, '# Runs', '#' * 50]
            for i, t in enumerate(trials):
                best = all([np.isclose(best_parameters[k], v) for k, v in t.items()])
                res_str.append('=' * 10 + '\n' +
                               f'Trial {i:4d} {"(BEST)" if best else ""}\n' +
                               '\n'.join([f'\t{k:>10s}: {v}' for k, v in t.items()]))
            res_str = '\n'.join(res_str)
            print(res_str)
            f.write(res_str)
    else:
        run_single_trial(launcher=launcher,
                         output_dir=os.path.join(exp_output_dir, variant_name),
                         dataset=dataset, net=net, num_runs=num_runs, arg_fname=arg_fname, run_args=run_args)


def TEST():
    ax_client = AxClient()
    ax_client.create_experiment(
        name="test_experiment",
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [-5.0, 10.0],
                "value_type": "float",
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [0.0, 10.0],
            },
        ],
        objective_name="obj",
        minimize=True,
    )

    def f(hyperparameters):
        x1 = hyperparameters['x1']
        x2 = hyperparameters['x2']
        a = np.random.rand()
        b = np.random.randint(-1, 2)
        v = a * x1 + (1 - a) * b * x2
        print(f'{x1:8.3f} {x2:8.3f} {a:8.3f} {b:4d} {v:8.3f}')
        return v

    for _ in range(15):
        parameters, trial_index = ax_client.get_next_trial()
        v = f(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=v)

    best_parameters, metrics = ax_client.get_best_parameters()
    print(best_parameters)
    print(metrics)


if __name__ == '__main__':
    # TEST()
    main()

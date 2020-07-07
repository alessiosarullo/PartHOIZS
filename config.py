import argparse
import os
import pickle
import sys


class Configs:
    def __init__(self):

        ##########################################
        # General options                        #
        ##########################################

        # Program
        self.sync = False
        self.debug = False
        self.debug_port = 16008
        self.monitor = False
        self.nworkers = 0
        self.no_load_memory = False

        # Print
        self.print_interval = 50
        self.log_interval = 100  # Tensorboard
        self.eval_interval = 10
        self.verbose = False

        # Experiment
        self.randomize = False
        self.resume = False
        self.save_dir = ''
        self.seenf = -1
        self.vceval = False
        self.eval_split = 'test'
        self._eval_only = None

        ##########################################
        # Data options                           #
        ##########################################

        # Null/background
        self.include_bg_only = False
        self.hoi_bg_ratio = 3

        # Image
        self.min_ppl_score = 0.95  # this is very high because it's a threshold over KP estimation scores, which are on people only
        self.max_ppl = 3
        self.max_obj = 3

        # Features
        self.feat_norm = False

        # Dataset
        self.ds = ''
        self.val_ratio = 0.1

        # ZS graphs
        self.oracle = False

        ##########################################
        # Model options                          #
        ##########################################
        self.model = None

        # Architecture
        self.dropout = 0.5
        self.repr_dim0 = 1024
        self.repr_dim1 = 256
        self.swish = False
        self.wrap = False

        # General
        self.act = False
        self.train_null_act = False
        self.obj = False  # add object branch
        self.no_psf = False

        # Loss
        self.fl_gamma = 0.0  # gamma in focal loss
        self.meanc = False  # mean or sum over classes for BCE loss?
        self.cspc = 0.0  # Use cost-sensitive coefficients for positive examples
        self.cspbgc_p = 0.0  # Cost-sensitive coefficient for positive background (null) examples of part actions
        self.cspc_p = 0.0  # Cost-sensitive coefficient for positive examples of part actions
        self.att_sp_c = 1.0  # Coefficient for sparsity loss on attention weights
        # Action Weak Supervision loss (Unseen, Logic - Necessary, Logic - Sufficient, Logic - Threshold, Logic - apply to Unseen Only).
        self.awsu = 0.0
        self.awsln = 1.0
        self.awsln_neg = 5.0
        self.awsls = 1.0
        self.awsls_pos = 5.0
        self.awslt = 0.5
        self.awsl_uonly = False

        # Graph regularisation  # FIXME
        self.agreg = 0.0

        # Interactiveness (actually more like spatial configuration: interactiveness detection is one possible usage)
        self.tin = False
        self.ipsize = 32  # size of the interactiveness pattern feature map
        self.tin_dropout = 0.8

        # ZS specific
        self.merge_dir = False
        self.no_dir = False

        # ZS GCN specific
        self.gcswish = False
        self.gcldim = 1024
        self.gcrdim = 1024
        self.gcdropout = 0.5

        # GAT specific
        self.gat = False
        self.gat_heads = 4

        # Part
        self.pbf = ''  # part branch save file (load pre-trained part branch from here)
        self.no_part = False
        self.sbmar = 0.0  # small boxes min area ratio

        ##########################################
        # Optimiser options                      #
        ##########################################

        # Optimiser
        self.sgd = False
        self.radam = False
        self.adamb1 = 0.9
        self.adamb2 = 0.999
        self.momentum = 0.9
        self.l2_coeff = 5e-4
        self.grad_clip = 5.0
        self.num_epochs = 20

        # Learning rate. A value of 0 means that option is disabled.
        self.lr = 0.0
        self._def_lr_sgd = 1e-3
        self._def_lr_adam = 5e-5
        self.lr_gamma = 0.1
        self.lr_decay_period = 0
        self.lr_warmup = 0
        self.c_lr_gcn = 0.0

        # Batch
        self.batch_size = 64

    @property
    def output_root(self):
        return 'output'

    @property
    def data_root(self):
        return 'data'

    @property
    def cache_root(self):
        return 'cache'

    @property
    def embedding_dir(self):
        return os.path.join(self.data_root, 'embeddings')

    @property
    def detectron_pretrained_file_format(self):
        return os.path.join(self.data_root, 'pretrained_model', '%s.pkl')

    @property
    def precomputed_feats_format(self):
        return os.path.join(self.cache_root, 'precomputed_%s__%s_%s.h5')

    @property
    def output_path(self):
        return os.path.join(self.output_root, self.ds, self.model, self.save_dir)

    @property
    def config_file(self):
        return os.path.join(self.output_path, 'config.pkl')

    @property
    def checkpoint_file(self):
        return os.path.join(self.output_path, 'ckpt.tar')

    @property
    def watched_values_file(self):
        return os.path.join(self.output_path, 'watched.tar')

    @property
    def best_model_file(self):
        return os.path.join(self.output_path, 'best.tar')

    @property
    def prediction_file(self):
        return os.path.join(self.output_path, f'prediction_{self.eval_split}.pkl')

    @property
    def eval_res_file_format(self):
        return os.path.join(self.output_path, '%s_eval_test.pkl')

    @property
    def ds_inds_file(self):
        return os.path.join(self.output_path, 'ds_inds.pkl')

    @property
    def seen_classes_file(self):
        assert self.seenf >= 0
        return os.path.join('zero-shot_inds', f'{self.ds}_seen_inds_{self.seenf}.pkl.push')

    @property
    def tensorboard_dir(self):
        return os.path.join(self.output_path, 'tboard')

    @property
    def output_analysis_path(self):
        return os.path.join(self.output_path, 'analysis')

    @property
    def eval_only(self):
        return self._eval_only

    def parse_args(self, fail_if_missing=True, reset=False):
        args = sys.argv
        if reset:
            self.__init__()

        parser = argparse.ArgumentParser(description='Settings')
        for k, v in vars(self).items():
            self._add_argument(parser, k, v, fail_if_missing=fail_if_missing)
        namespace = parser.parse_known_args(args)
        self.__dict__.update({k: v for k, v in vars(namespace[0]).items() if v is not None})
        self._postprocess_args(fail_if_missing)

        args = namespace[1]
        if args[1:]:
            # Invalid options: either unknown or ones initialised as None, which should not be changed.
            raise ValueError('Invalid arguments: %s.' % ' '.join(args[1:]))
        sys.argv = sys.argv[:1]

    @staticmethod
    def _add_argument(parser, param_name, param_value, fail_if_missing=True):
        if param_name == 'model':
            from scripts.utils import get_all_models_by_name
            all_models_dict = get_all_models_by_name()
            all_models = set(all_models_dict.keys())
            parser.add_argument('--%s' % param_name, dest=param_name, type=str, choices=all_models)
        elif param_name == 'save_dir':
            parser.add_argument('--%s' % param_name, dest=param_name, type=str, required=fail_if_missing)
        else:
            if param_value is not None:
                parser_kwargs = {'dest': param_name}
                if type(param_value) == bool:
                    parser_kwargs['action'] = 'store_%s' % str(not param_value).lower()
                else:
                    parser_kwargs['type'] = type(param_value)
                parser.add_argument('--%s' % param_name, **parser_kwargs)

    def _postprocess_args(self, fail_if_missing):
        self.save_dir = self.save_dir.rstrip('/')
        if '/' in self.save_dir:
            save_dir_parts = self.save_dir.split('/')
            if save_dir_parts[0] == self.output_root:
                assert len(save_dir_parts) >= 4
                old_save_dir = self.save_dir
                self.save_dir = os.path.join(*save_dir_parts[3:])  # skip 'output', dataset, model
                self.ds = self.ds or save_dir_parts[1]
                self.model = self.model or save_dir_parts[2]
                assert old_save_dir == self.output_path
        if fail_if_missing:
            if not self.ds:
                raise ValueError('A dataset is required.')
            if self.model is None:
                raise ValueError('A model is required.')

        if self.lr == 0:
            self.lr = self._def_lr_sgd if self.sgd else self._def_lr_adam

        self._eval_only = os.path.exists(self.best_model_file) and not self.resume

    def save(self, file_path=None):
        file_path = file_path or self.config_file
        with open(file_path, 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, file_path=None):
        cfg_file_path = file_path or self.config_file
        with open(cfg_file_path, 'rb') as f:
            d = pickle.load(f)

        if file_path is None:
            # Save options that should not be loaded
            output_path = self.output_path
            save_dir = self.save_dir
            resume = self.resume
            num_epochs = self.num_epochs
            eval_split = self.eval_split
            vceval = self.vceval

            # Load
            self.__dict__.update(d)

            # Restore
            self.eval_split = eval_split
            self.vceval = vceval
            self.save_dir = save_dir
            self.resume = resume
            assert self.output_path.rstrip('/') == output_path.rstrip('/'), (self.output_path, output_path)
            if resume:
                self.num_epochs += num_epochs
        else:
            self.__dict__.update(d)
            self.resume = False

    def print(self):
        print(str(self), '\n')

    def __str__(self):
        s = []
        selfdict = self.__dict__
        for k in sorted(selfdict.keys()):
            v = selfdict[k]
            s += [f'{k:<35s} {str(v.__class__.__name__):10s} {str(v)}']
        return '{0}\nConfigs\n{0}\n{1}\n{0}\n{0}'.format('=' * 70, '\n'.join(s))


# Instantiate
cfg = Configs()  # type: Configs


def test():
    # print('Default configs')
    # Configs.print()

    # sys.argv += ['--sync', '--model', 'hicoall', '--save_dir', 'blabla/2019-24-12_param1/RUN1']
    # sys.argv += ['--save_dir', 'output/hicoall/blabla/2019-24-12_param1/RUN1']
    # sys.argv += ['--save_dir', 'hicoall/blabla/2019-24-12_param1/RUN1']  # this should fail
    sys.argv += ['--save_dir', 'output/hicoall/']  # this too
    cfg.parse_args()
    # print('Updated with args:', sys.argv)
    cfg.print()
    print(cfg.output_path)


if __name__ == '__main__':
    test()

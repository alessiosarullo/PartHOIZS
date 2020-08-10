import datetime
import os
import pickle
import random
import shutil
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, Minibatch
from lib.dataset.hicodet_hake import HicoDetHakeSplit
from lib.dataset.vcoco import VCocoSplit
from lib.dataset.cocoa import CocoaSplit
from lib.dataset.hico_cocoa import HicoCocoaSplit
from lib.eval.evaluator_hicodethake_pstate import EvaluatorHicoDetHakePartROI
from lib.eval.evaluator_vcoco_roi import EvaluatorVCocoROI
from lib.eval.evaluator_hoi_roi import EvaluatorHoiRoi
from lib.eval.vsrl_eval import VCOCOeval, pkl_from_predictions
from lib.models.abstract_model import AbstractModel, Prediction
from lib.radam import RAdam
from lib.timer import Timer
from scripts.running_stats import RunningStats
from scripts.utils import print_params, get_all_models_by_name


class Launcher:
    def __init__(self):
        torch.multiprocessing.set_start_method('spawn')
        self._reset()

    def _reset(self):
        self.detector = None  # type: Union[None, AbstractModel]
        self.train_split = None  # type: Union[None, HoiDatasetSplit]
        self.test_split = None  # type: Union[None, HoiDatasetSplit]
        self.curr_train_iter = 0
        self.start_epoch = 0

    def run(self):
        # Parse configs
        self._reset()
        cfg.parse_args(reset=True)
        Timer.gpu_sync = cfg.sync
        if cfg.debug:
            try:  # PyCharm debugging
                print('Starting remote debugging (resume from debug server)')
                import pydevd_pycharm
                pydevd_pycharm.settrace('130.88.195.105', port=cfg.debug_port, stdoutToServer=True, stderrToServer=True)
                print('Remote debugging activated.')
            except:
                print('Remote debugging failed.')
                raise
        if cfg.eval_only or cfg.resume:
            cfg.load()
        cfg.print()

        # Run the experiment
        if cfg.eval_only:
            self.setup()
            print('Start eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            predictions = self.evaluate_only()
            print('End eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            lowest_val_loss = None
        else:
            try:
                self.setup()
                os.makedirs(cfg.output_path, exist_ok=True)
                cfg.save()

                assert cfg.eval_split == 'test'
                try:
                    os.remove(cfg.prediction_file)
                except FileNotFoundError:
                    pass
                assert cfg.num_epochs > 0

                print('Start train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                predictions, lowest_val_loss = self.train()
                print('End train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            except:
                if not cfg.eval_only:
                    print(f'Exception raised. Removing "{cfg.output_path}".')
                    try:
                        shutil.rmtree(cfg.output_path)
                        os.removedirs(os.path.split(cfg.output_path.rstrip('/'))[0])
                    except OSError:
                        pass
                raise

        with open(cfg.prediction_file, 'wb') as f:
            pickle.dump(predictions, f)
        print('Wrote results to %s.' % cfg.prediction_file)

        return lowest_val_loss

    def setup(self):
        seed = 3 if not cfg.randomize else np.random.randint(1_000_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('RNG seed:', seed)

        # Data
        # Load inds from configs. Note that these might be None after this step, which means all possible indices will be used.
        inds = {k: None for k in ['obj', 'act', 'hoi']}
        if cfg.seenf >= 0:
            inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
            for k in ['obj', 'act', 'hoi']:
                try:
                    inds[k] = sorted(inds_dict['train'][k].tolist())
                except KeyError:
                    pass
        assert inds['hoi'] is None or (inds['obj'] is None and inds['act'] is None)

        if cfg.ds == 'hico' or cfg.ds == 'hh' or cfg.ds == 'hicodet':  # legacy support
            ds_class = HicoDetHakeSplit
        elif cfg.ds == 'vcoco':
            ds_class = VCocoSplit
        elif cfg.ds == 'cocoa':
            ds_class = CocoaSplit
        elif cfg.ds == 'hc':
            ds_class = HicoCocoaSplit
        else:
            raise ValueError('Unknown dataset.')
        splits = ds_class.get_splits(object_inds=inds['obj'], action_inds=inds['act'], interaction_inds=inds['hoi'],
                                     val_ratio=cfg.val_ratio, load_precomputed_data=True)
        self.train_split, self.test_split = splits['train'], splits['test']

        # Model
        self.detector = get_all_models_by_name()[cfg.model](self.train_split)  # type: AbstractModel
        if torch.cuda.is_available():
            self.detector.cuda()
        else:
            print('!!!!!!!!!!!!!!!!! Running on CPU!')
        print_params(self.detector, breakdown=False)

        if cfg.resume:
            ckpt = torch.load(cfg.checkpoint_file)
            self.detector.load_state_dict(ckpt['state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.curr_train_iter = ckpt['curr_iter'] + 1
            print(f'Continuing from epoch {self.start_epoch} @ iteration {self.curr_train_iter}.')
        elif cfg.eval_only:
            ckpt = torch.load(cfg.best_model_file)
            self.detector.load_state_dict(ckpt['state_dict'])

    def get_optim(self):
        params = self.detector.parameters()
        if cfg.resume:
            params = [{'params': p, 'initial_lr': cfg.lr} for p in self.detector.parameters() if p.requires_grad]

        if cfg.sgd:
            assert not cfg.radam
            optimizer = torch.optim.SGD(params, weight_decay=cfg.l2_coeff, lr=cfg.lr, momentum=cfg.momentum)
        elif cfg.radam:
            optimizer = RAdam(params, weight_decay=cfg.l2_coeff, lr=cfg.lr, betas=(cfg.adamb1, cfg.adamb2))
        else:
            assert not cfg.radam
            optimizer = torch.optim.Adam(params, weight_decay=cfg.l2_coeff, lr=cfg.lr, betas=(cfg.adamb1, cfg.adamb2))

        lr_decay = cfg.lr_decay_period
        lr_warmup = cfg.lr_warmup
        if lr_warmup > 0:
            assert lr_decay == 0
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_warmup], gamma=cfg.lr_gamma,
                                                             last_epoch=self.start_epoch - 1)
        elif lr_decay > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_period, gamma=cfg.lr_gamma,
                                                        last_epoch=self.start_epoch - 1)
        else:
            scheduler = None
        return optimizer, scheduler

    def train(self):
        optimizer, scheduler = self.get_optim()
        train_loader = self.train_split.get_loader(batch_size=cfg.batch_size, num_workers=cfg.nworkers)
        val_loader = self.train_split.get_loader(batch_size=cfg.batch_size, holdout_set=True)
        test_loader = self.test_split.get_loader(batch_size=cfg.batch_size)

        training_stats = RunningStats(split='train', num_batches=len(train_loader),
                                      batch_size=train_loader.batch_size or train_loader.batch_sampler.batch_size)
        val_stats = RunningStats(split='val', num_batches=len(val_loader),
                                 batch_size=val_loader.batch_size or val_loader.batch_sampler.batch_size, history_window=len(val_loader))
        test_stats = RunningStats(split='test', num_batches=len(test_loader), batch_size=cfg.batch_size, history_window=len(test_loader))

        try:
            lowest_val_metric = np.inf
            last_epoch = cfg.num_epochs
            for epoch in range(self.start_epoch, last_epoch):
                print('Epoch %d start.' % epoch)
                self.detector.train()
                self.loss_epoch(epoch, train_loader, training_stats, optimizer)

                self.detector.train()  # needed to save the correct state
                torch.save({'epoch': epoch,
                            'curr_iter': self.curr_train_iter,
                            'state_dict': self.detector.state_dict()},
                           cfg.checkpoint_file)

                self.detector.eval()
                val_loss = self.loss_epoch(epoch, val_loader, val_stats)
                if scheduler is not None:
                    try:
                        scheduler.step(metrics=val_loss)
                    except TypeError:
                        # Scheduler default behaviour is wrong: it gets called with epoch=0 twice, both at the beginning and after the first epoch
                        scheduler.step(epoch=epoch + 1)

                val_metric = val_loss.cpu().numpy()
                if val_metric < lowest_val_metric:  # save best model only
                    lowest_val_metric = val_metric
                    self.detector.train()  # needed to save the correct state
                    torch.save({'epoch': epoch,
                                'curr_iter': self.curr_train_iter,
                                'state_dict': self.detector.state_dict()},
                               cfg.best_model_file)

                if epoch % cfg.eval_interval == 0 or epoch + 1 == last_epoch:
                    test_predictions = self.compute_predictions(epoch_idx=epoch, data_loader=test_loader, stats=test_stats)
                    self.eval_epoch(predictions=test_predictions, epoch_idx=epoch, eval_split=self.test_split, stats=test_stats)
                    if epoch + 1 < last_epoch:
                        del test_predictions

                # if any([pg['lr'] <= 1e-6 for pg in optimizer.param_groups]):
                #     print('Exiting training early.', flush=True)
                #     break
            assert test_predictions is not None
            Timer.get().print()
        finally:
            training_stats.close_tensorboard_logger()
            val_stats.close_tensorboard_logger()
            test_stats.close_tensorboard_logger()

        # noinspection PyUnboundLocalVariable
        return test_predictions, lowest_val_metric

    def loss_epoch(self, epoch_idx, data_loader, stats: RunningStats, optimizer=None):
        stats.epoch_tic()
        epoch_loss = 0
        for batch_idx, batch in self.data_loader_generator(data_loader, epoch_idx):
            stats.batch_tic()
            batch_loss = self.loss_batch(batch, stats, optimizer)
            if optimizer is None:
                epoch_loss += batch_loss.detach()
            stats.batch_toc()

            verbose = (batch_idx % (cfg.print_interval * (100 if optimizer is None else 1)) == 0)
            if optimizer is not None:  # training
                if batch_idx % cfg.log_interval == 0:
                    stats.log_stats(self.curr_train_iter, verbose=verbose,
                                    lr=optimizer.param_groups[0]['lr'], epoch=epoch_idx, batch=batch_idx)
                self.curr_train_iter += 1
            else:
                if verbose:
                    stats.print_times(epoch_idx, batch=batch_idx, curr_iter=self.curr_train_iter)

            # torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.

        if optimizer is None:
            stats.log_stats(self.curr_train_iter, epoch_idx)
        epoch_loss /= len(data_loader)
        stats.epoch_toc()
        return epoch_loss

    def loss_batch(self, batch, stats: RunningStats, optimizer=None):
        """ :arg `optimizer` should be None on validation batches. """

        losses = self.detector(batch, inference=False)
        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        losses['total_loss'] = loss

        batch_stats = {'losses': losses}
        # values_to_monitor = {k: v.detach().cpu() for k, v in self.detector.values_to_monitor.items()}
        # if values_to_monitor:
        #     batch_stats['hist'] = values_to_monitor

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_([p for p in self.detector.parameters() if p.grad is not None], max_norm=cfg.grad_clip)

            if cfg.monitor:
                batch_stats['grads'] = {k + '_gradnorm': v.grad.detach().cpu().norm() for k, v in self.detector.named_parameters()
                                        if v.requires_grad and 'bias' not in k}

            optimizer.step()

        stats.update_stats(batch_stats)

        return loss

    def compute_predictions(self, epoch_idx, data_loader: DataLoader, stats: RunningStats, do_toc=False):
        self.detector.eval()
        all_predictions = []
        stats.epoch_tic()

        for batch_idx, batch in self.data_loader_generator(data_loader, epoch_idx):
            stats.batch_tic()
            if batch is None:
                predictions = Prediction()
            else:
                predictions = self.detector(batch)  # type: Prediction
                extra_info = self.detector.extra_infos
                if extra_info:
                    predictions.extra_info = {k: v.detach().cpu().numpy() for k, v in extra_info.items()}
            if data_loader.batch_size > 1:
                predictions_vars = vars(predictions)
                for i in range(data_loader.batch_size):
                    pred_vars_i = {}
                    for k, v in predictions_vars.items():
                        if v is not None:
                            if isinstance(v, np.ndarray):
                                pred_vars_i[k] = v[[i]]
                            else:
                                assert isinstance(v, dict)
                                pred_vars_i[k] = {k1: v1[i] for k1, v1 in v.items()}
                    all_predictions.append(pred_vars_i)
            else:
                all_predictions.append(vars(predictions))
            stats.batch_toc()

            # if batch_idx % 20 == 0:
            #     torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.
            if batch_idx % 500 == 0:
                stats.print_times(epoch_idx, batch=batch_idx, curr_iter=self.curr_train_iter)

        if do_toc:
            stats.epoch_toc()
        return all_predictions

    def eval_epoch(self, predictions, epoch_idx, eval_split: HoiDatasetSplit, stats: RunningStats, metric_prefix='', do_tic=False):
        self.detector.eval()
        if do_tic:
            stats.epoch_tic()

        do_part_eval = do_hoi_eval = False
        for p in predictions:
            pr = Prediction(p)
            do_part_eval = do_part_eval or pr.part_state_scores is not None
            do_hoi_eval = do_hoi_eval or pr.output_scores is not None
            if do_part_eval and do_hoi_eval:
                break

        metric_dict = {}
        if do_part_eval and isinstance(eval_split, HicoDetHakeSplit):
            print('Part states:')
            part_evaluator = EvaluatorHicoDetHakePartROI(eval_split)
            part_evaluator.evaluate_predictions(predictions)
            part_metric_dict = {f'Part_{k}': v for k, v in part_evaluator.output_metrics().items()}
            metric_dict.update(part_metric_dict)

            null_classes = np.flatnonzero([states[-1] for states in eval_split.full_dataset.states_per_part]).tolist()
            part_interactiveness_metric_dict = part_evaluator.output_metrics(compute_pos=False, to_keep=null_classes)
            part_interactiveness_metric_dict = {f'Part_interactiveness_{k}': v for k, v in part_interactiveness_metric_dict.items()}
            assert not (set(part_interactiveness_metric_dict.keys()) & set(metric_dict.keys()))
            metric_dict.update(part_interactiveness_metric_dict)

        EVAL_TYPES = ['interactions', 'actions']
        eval_type = 'actions' if eval_split.full_dataset.labels_are_actions else 'interactions'
        assert eval_type in EVAL_TYPES

        if cfg.seenf >= 0:
            if eval_type == 'interactions':
                seen = self.train_split.seen_interactions
                unseen = np.setdiff1d(np.arange(self.train_split.full_dataset.num_interactions), seen).tolist()
            elif eval_type == 'actions':
                seen = self.train_split.seen_actions
                unseen = np.setdiff1d(np.arange(self.train_split.full_dataset.num_actions), seen).tolist()
            else:
                raise ValueError(f'Eval type must be one of f{EVAL_TYPES}, but it is {eval_type}.')
        else:
            seen = unseen = None

        if do_hoi_eval:
            if cfg.vceval:
                assert isinstance(eval_split, VCocoSplit)
                evaluator = VCOCOeval(vsrl_annot_file=os.path.join(cfg.data_root, 'V-COCO', 'vcoco', 'vcoco_test.json'),
                                      coco_annot_file=os.path.join(cfg.data_root, 'V-COCO', 'instances_vcoco_all_2014.json'),
                                      split_file=os.path.join(cfg.data_root, 'V-COCO', 'splits', 'vcoco_test.ids')
                                      )
                det_file = os.path.join(cfg.output_path, 'vcoco_pred.pkl')
                pkl_from_predictions(dict_predictions=predictions, dataset=eval_split, filename=det_file)
                evaluator._do_eval(det_file, ovr_thresh=0.5, seen_acts_str=[eval_split.actions[i] for i in seen] if seen is not None else None)
            else:
                print('All:')
                if eval_type == 'interactions':
                    evaluator = EvaluatorHoiRoi(eval_split)
                    null_classes = np.flatnonzero(self.test_split.full_dataset.interactions[:, 0] == 0).tolist()
                elif eval_type == 'actions':
                    evaluator = EvaluatorVCocoROI(eval_split)
                    null_classes = [0]
                else:
                    raise ValueError(f'Eval type must be one of f{EVAL_TYPES}, but it is "{eval_type}".')
                evaluator.evaluate_predictions(predictions)
                hoi_metric_dict = evaluator.output_metrics()
                hoi_metric_dict = {f'{k}': v for k, v in hoi_metric_dict.items()}
                assert not (set(hoi_metric_dict.keys()) & set(metric_dict.keys()))
                metric_dict.update(hoi_metric_dict)

                interactiveness_metric_dict = evaluator.output_metrics(compute_pos=False, to_keep=null_classes)
                interactiveness_metric_dict = {f'HOI_interactiveness_{k}': v for k, v in interactiveness_metric_dict.items()}
                assert not (set(interactiveness_metric_dict.keys()) & set(metric_dict.keys()))
                metric_dict.update(interactiveness_metric_dict)

                if seen is not None:
                    assert unseen is not None
                    detailed_metric_dicts = []

                    print('Seen:')
                    detailed_metric_dicts.append({f'tr_{k}': v for k, v in evaluator.output_metrics(to_keep=sorted(seen)).items()})

                    print('Unseen:')
                    detailed_metric_dicts.append({f'zs_{k}': v for k, v in evaluator.output_metrics(to_keep=sorted(unseen)).items()})

                    if isinstance(eval_split, HicoCocoaSplit):
                        # Sanity check
                        assert sorted(self.train_split.objects) == eval_split.full_dataset.hico.objects
                        assert self.train_split.actions == eval_split.full_dataset.hico.actions
                        assert [f'{self.train_split.full_dataset.actions[a]} {self.train_split.full_dataset.objects[o]}'
                                for a, o in self.train_split.interactions] == sorted(eval_split.full_dataset.hico.interactions_str)

                        unseen_acts = np.setdiff1d(np.arange(self.train_split.full_dataset.num_actions), self.train_split.seen_actions)
                        unseen_w_unseen_acts = np.setdiff1d(self.train_split.full_dataset.oa_to_interaction[:, unseen_acts], [-1])
                        assert np.all(unseen_w_unseen_acts >= 0) and np.setdiff1d(unseen_w_unseen_acts, unseen).size == 0

                        print('Unseen (with unseen actions):')
                        detailed_metric_dicts.append({f'zs_unseen_acts_{k}': v
                                                      for k, v in evaluator.output_metrics(to_keep=sorted(unseen_w_unseen_acts)).items()})

                    for d in detailed_metric_dicts:
                        for k, v in d.items():
                            assert k not in metric_dict
                            metric_dict[k] = v
                else:
                    assert unseen is None

        mean_metric_dict = {k: np.mean(v) for k, v in metric_dict.items()}

        if metric_prefix:
            metric_prefix += '__'
        stats.update_stats({'metrics': {f'{metric_prefix}{k}': v for k, v in mean_metric_dict.items()}})
        stats.log_stats(self.curr_train_iter, epoch_idx)

        stats.epoch_toc()
        return mean_metric_dict

    def evaluate_only(self):
        if cfg.eval_split == 'train':
            ds_split = self.train_split
            data_loader = ds_split.get_loader(batch_size=cfg.batch_size, evaluation=True)
        elif cfg.eval_split == 'test':
            ds_split = self.test_split
            data_loader = ds_split.get_loader(batch_size=cfg.batch_size)
        else:
            raise ValueError(f'Possible splits to evaluate on are "train" or "test". Picked: {cfg.eval_split}.')
        stats = RunningStats(split=cfg.eval_split, num_batches=len(data_loader), batch_size=1, history_window=len(data_loader),
                             tboard_log=False)

        try:
            with open(cfg.prediction_file, 'rb') as f:
                predictions = pickle.load(f)
            print('Results loaded from %s.' % cfg.prediction_file)
        except FileNotFoundError:
            predictions = self.compute_predictions(epoch_idx=None, data_loader=data_loader, stats=stats)

        self.eval_epoch(predictions=predictions, epoch_idx=None, eval_split=ds_split, stats=stats)
        return predictions

    def data_loader_generator(self, data_loader, epoch_idx):
        for batch_idx, batch in enumerate(data_loader):
            batch = batch  # type: Union[None, Minibatch]
            if batch is not None:
                batch = batch._replace(epoch=epoch_idx, iter=self.curr_train_iter)
            yield batch_idx, batch


def main():
    Launcher().run()


if __name__ == '__main__':
    main()

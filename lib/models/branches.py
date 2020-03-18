from typing import List, NamedTuple, Callable

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hico_hake import HicoHakeKPSplit, Minibatch
from lib.dataset.utils import interactions_to_mat
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.gcns import BipartiteGCN
from lib.models.misc import bce_loss, weighted_binary_cross_entropy_with_logits, LIS, MemoryEfficientSwish as Swish


class LabelData(NamedTuple):
    tag: str
    num_classes: int
    seen_classes: np.array
    unseen_classes: np.array
    label_f: Callable
    all_classes_str: List[str]


class Cache:
    def __init__(self, word_embs: WordEmbeddings):
        self.word_embs = word_embs
        self.batch_cache = {}
        self.reset()

    def reset(self):
        self.batch_cache = {}

    def __getitem__(self, item):
        return self.batch_cache[item]

    def __setitem__(self, key, value):
        assert key not in self.batch_cache.keys()
        self.batch_cache[key] = value


class AbstractModule(nn.Module):
    def __init__(self, dataset: HicoHakeKPSplit, cache: Cache, repr_dim, **kwargs):
        super().__init__()
        self.dataset = dataset  # type: HicoHakeKPSplit
        self.dims = self.dataset.dims
        self.cache = cache
        self.repr_dim = repr_dim
        self.repr_mlps = nn.ModuleDict()
        self.linear_predictors = nn.ParameterDict()

    def _add_mlp(self, name, input_dim, output_dim, num_classes=None, hidden_dim=1024, dropout_p=None):
        assert name not in self.repr_mlps
        if dropout_p is None:
            dropout_p = cfg.dropout
        self.repr_mlps[name] = nn.Sequential(*([nn.Linear(input_dim, hidden_dim),
                                                (Swish() if cfg.swish else nn.ReLU(inplace=True))
                                                ] +
                                               ([nn.Dropout(p=dropout_p)] if dropout_p > 0 else []) +
                                               [nn.Linear(hidden_dim, output_dim)
                                                ]
                                               ))
        nn.init.xavier_normal_(self.repr_mlps[name][0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.repr_mlps[name][3 if dropout_p > 0 else 2].weight, gain=torch.nn.init.calculate_gain('linear'))
        if num_classes is not None:
            self._add_linear_layer(name=name, input_dim=output_dim, output_dim=num_classes)

    def _add_linear_layer(self, name, input_dim, output_dim):
        assert name not in self.linear_predictors
        self.linear_predictors[name] = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.linear_predictors[name], gain=1.0)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AbstractBranch(AbstractModule):
    def __init__(self, enable_zs=True, wrapped_branch=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zs_enabled = (cfg.seenf >= 0) & enable_zs
        self.wrapped_branch = wrapped_branch

    def _check_dims(self, x: Minibatch = None):
        if x is None:
            ex_data = im_data = person_data = obj_data = None
        else:
            ex_data, im_data, person_data, obj_data = x.ex_data, x.im_data, x.person_data, x.obj_data
        dims = self.dims
        P, M, K_coco, K_hake, O, F_kp, F_obj, D = dims.P, dims.M, dims.K_coco, dims.K_hake, dims.O, dims.F_kp, dims.F_obj, dims.D

        N = None
        if ex_data is not None:
            feats = ex_data[1]
            N = feats.shape[0]

        if im_data is not None:
            assert im_data[1].shape[0] == N

        if person_data is not None:
            ppl_boxes, ppl_feats, coco_kps, kp_boxes, kp_feats = person_data[:5]
            N = N if N is not None else kp_feats.shape[0]
            assert ppl_boxes.shape == (N, P, 4)
            assert ppl_feats.shape == (N, P, F_kp)
            assert coco_kps.shape == (N, P, K_coco, 3)
            assert kp_boxes.shape == (N, P, K_hake, 5)
            assert kp_feats.shape == (N, P, K_hake, F_kp)

        if obj_data is not None:
            obj_boxes, obj_scores, obj_feats = obj_data[:3]
            N = N if N is not None else obj_boxes.shape[0]
            assert obj_boxes.shape == (N, M, 4)
            assert obj_scores.shape == (N, M, O)
            assert obj_feats.shape == (N, M, F_obj)

        if cfg.tin:
            ipatterns = person_data[5]
            assert ipatterns.shape[:3] == (N, P, M) and ipatterns.shape[3:5] == (D, D) and len(ipatterns.shape) == 6

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            self._check_dims(x)
            output = self.forward_wrap(x)
            if not isinstance(output, tuple):
                output = (output,)
            output = tuple((sum(o) / len(o)) if isinstance(o, list) else o for o in output)
            if isinstance(output, tuple) and len(output) == 1:
                output = output[0]
            if not inference:
                return self._get_losses(x, output)
            else:
                return self._predict(x, output)

    def forward_wrap(self, x):
        output = self._forward(x)
        if self.wrapped_branch is not None:
            if not isinstance(output, tuple):
                output = (output,)
            output_lists = tuple([o] if not isinstance(o, list) else o for o in output)
            wrapped_output = self.wrapped_branch.forward_wrap(x)
            if not isinstance(wrapped_output, tuple):
                wrapped_output = (wrapped_output,)
            assert len(wrapped_output) == len(output) == len(output_lists)
            for olist, wrapped_o in zip(output_lists, wrapped_output):
                olist += (wrapped_o if isinstance(wrapped_o, list) else [wrapped_o])
            output = output_lists
        return output

    def _get_losses(self, x: Minibatch, output, **kwargs):
        raise NotImplementedError()

    def _predict(self, x: Minibatch, output, **kwargs):
        raise NotImplementedError()

    def _forward(self, x: Minibatch):
        raise NotImplementedError()


class SpatialConfigurationModule(AbstractModule):
    def __init__(self, sp_ch=64, pose_ch=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        K_hake, F_kp, F_obj, D = self.dims.K_hake, self.dims.F_kp, self.dims.F_obj, self.dims.D
        d = D // 4
        self.repr_dim = d * d * (sp_ch // 2 + pose_ch // 2)

        self.sp_map_module = nn.Sequential(
            nn.Conv2d(in_channels=2 + K_hake, out_channels=sp_ch, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=sp_ch, out_channels=sp_ch // 2, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
            # Output is (N, D/4, D/4, sp_ch/2)
        )

        self.pose_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=pose_ch, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=pose_ch, out_channels=pose_ch // 2, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Output is (N, D/4, D/4, pose_ch/2)
        )

    def forward(self, x: Minibatch):
        with torch.set_grad_enabled(self.training):
            ipatterns = x.person_data[5]
            P, M, D = self.dims.P, self.dims.M, self.dims.D
            N = ipatterns.shape[0]

            ipatterns = ipatterns.view(N * P * M, D, D, -1).permute(0, 3, 1, 2)
            sp_feats = self.sp_map_module(ipatterns[:, :-1, ...]).view(N * P * M, -1).view(N, P, M, -1)
            pose_feats = self.pose_module(ipatterns[:, -1:, ...]).view(N * P * M, -1).view(N, P, M, -1)

            feats = torch.cat([sp_feats, pose_feats], dim=-1)
            return feats


class SpatialConfigurationBranch(AbstractBranch):
    def __init__(self, sp_ch=64, pose_ch=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spconf = SpatialConfigurationModule(sp_ch=sp_ch, pose_ch=pose_ch, *args, **kwargs)
        F_kp, F_obj = self.dims.F_kp, self.dims.F_obj
        repr_dim = self.repr_dim

        self.label_inds = nn.Parameter(torch.from_numpy(self._get_label_inds()), requires_grad=False)
        self.predictor = nn.Linear(in_features=repr_dim, out_features=self.label_inds.shape[0])

        # Keeping the same names as the official TIN implementation (although this is a simplified version).
        self.fc8_binary1 = nn.Sequential(nn.Linear(in_features=self.spconf.repr_dim + F_kp, out_features=repr_dim),
                                         nn.Dropout(p=cfg.tin_dropout))
        self.fc8_binary2 = nn.Sequential(nn.Linear(in_features=F_obj + self.dims.O, out_features=repr_dim),
                                         nn.Dropout(p=cfg.tin_dropout))
        self.fc9_binary = nn.Sequential(nn.Linear(in_features=2 * repr_dim, out_features=repr_dim),
                                        nn.Dropout(p=cfg.tin_dropout))

    def _get_label_inds(self):
        raise NotImplementedError

    def _get_losses(self, x: Minibatch, output, **kwargs):
        raise NotImplementedError

    def _predict(self, x: Minibatch, output, **kwargs):
        return torch.sigmoid(output).cpu().numpy()

    def _forward(self, x: Minibatch):
        ppl_boxes, ppl_feats, coco_kps, kp_boxes, kp_feats, ipatterns = x.person_data[:6]
        obj_boxes, obj_scores, obj_feats = x.obj_data[:3]
        P, M, D = self.dims.P, self.dims.M, self.dims.D
        N = ppl_boxes.shape[0]

        spconf_feats = self.spconf(x)

        interactiveness_feats_H = torch.cat([ppl_feats.unsqueeze(dim=2).expand(N, P, M, -1),
                                             spconf_feats
                                             ], dim=-1)
        interactiveness_feats_H = self.fc8_binary1(interactiveness_feats_H)

        interactiveness_feats_O = self.fc8_binary2(torch.cat([obj_feats, obj_scores], dim=-1))

        interactiveness_feats_HO = torch.cat([interactiveness_feats_H, interactiveness_feats_O.unsqueeze(dim=1).expand(N, P, M, -1)], dim=-1)
        interactiveness_feats = self.fc9_binary(interactiveness_feats_HO)

        logits = self.predictor(interactiveness_feats)  # N x P x M x C
        logits = self._reduce_logits(logits)  # N x C
        return logits

    def _reduce_logits(self, logits):
        return logits.max(dim=2)[0].max(dim=1)[0]


class HoiUninteractivenessBranch(SpatialConfigurationBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_label_inds(self):
        return np.flatnonzero(self.dataset.full_dataset.interactions[:, 0] == 0)  # null interactions

    def _get_losses(self, x: Minibatch, output, **kwargs):
        labels = x.ex_labels[:, self.label_inds]
        losses = {f'hoi_bg_loss': bce_loss(output, labels)}
        return losses

    def _reduce_logits(self, logits):
        return logits.min(dim=2)[0].min(dim=1)[0]


class PartUninteractivenessBranch(SpatialConfigurationBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_label_inds(self):
        return np.array([states[-1] for states in self.dataset.full_dataset.states_per_part])  # null part states

    def _get_losses(self, x: Minibatch, output, **kwargs):
        labels = x.pstate_labels[:, self.label_inds]
        losses = {f'part_bg_loss': bce_loss(output, labels)}
        return losses

    def _reduce_logits(self, logits):
        return logits.min(dim=2)[0].min(dim=1)[0]


class PartStateBranch(AbstractBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hicohake = self.dataset.full_dataset

        self.final_part_repr_dim = None
        if cfg.tin:
            self.uninteractiveness_branch = PartUninteractivenessBranch(dataset=self.dataset, cache=self.cache,
                                                                        repr_dim=256, sp_ch=32, pose_ch=16)
        for i, p_states in enumerate(hicohake.states_per_part):
            part_name = hicohake.parts[i]
            self._add_mlp(name=f'img_part_{part_name}', input_dim=self.dims.F_img, output_dim=self.repr_dim)
            self._add_mlp(name=f'kp_part_{part_name}', input_dim=self.dims.F_kp, output_dim=self.repr_dim)
            logit_input_dim = self.repr_dim + self.dims.O + self.repr_dim  # direct repr + obj scores + aggr people repr

            if self.final_part_repr_dim is None:
                self.final_part_repr_dim = logit_input_dim
            else:
                assert self.final_part_repr_dim == logit_input_dim
            self._add_linear_layer(name=f'img_part_{part_name}', input_dim=logit_input_dim, output_dim=p_states.size)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        hh = self.dataset.full_dataset
        orig_part_state_labels = x.pstate_labels
        part_dir_logits = output
        bg_part_states = torch.from_numpy(np.array([app[-1] for app in hh.states_per_part])).to(device=part_dir_logits.device)
        fg_part_states = torch.from_numpy(np.concatenate([app[:-1] for app in hh.states_per_part])).to(device=part_dir_logits.device)
        losses = {'fg_part_loss': bce_loss(part_dir_logits[:, fg_part_states], orig_part_state_labels[:, fg_part_states],
                                           pos_weights=cfg.cspc_p if cfg.cspc_p > 0 else None),
                  'bg_part_loss': bce_loss(part_dir_logits[:, bg_part_states], orig_part_state_labels[:, bg_part_states],
                                           pos_weights=cfg.cspbgc_p if cfg.cspbgc_p > 0 else None)
                  }
        if cfg.tin:
            losses.update(self.uninteractiveness_branch(x, inference=False))
        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        part_dir_logits = output
        part_state_scores = np.zeros((x.ex_data[1].shape[0], self.dims.S))
        if part_dir_logits is not None:
            part_state_scores = torch.sigmoid(part_dir_logits).cpu().numpy()
            if cfg.tin:
                part_state_scores[:, self.uninteractiveness_branch.label_inds] = self.uninteractiveness_branch(x, inference=True)
        return part_state_scores

    def _forward(self, x: Minibatch):
        im_part_reprs = self._get_repr(x)
        im_part_logits_list = []
        for part_idx, p in enumerate(self.dataset.full_dataset.parts):
            p_logits = im_part_reprs[..., part_idx, :] @ self.linear_predictors[f'img_part_{p}']  # N x P x M x S_part
            im_p_logits = p_logits.max(dim=2)[0].max(dim=1)[0]  # N x S_part
            im_p_logits[:, -1] = p_logits[..., -1].min(dim=2)[0].min(dim=1)[0]  # if ANYONE is interacting, NULL should not be predicted (thus min)
            im_part_logits_list.append(im_p_logits)
        im_part_dir_logits = torch.cat(im_part_logits_list, dim=-1)  # N x S
        self.cache['part_dir_logits'] = im_part_dir_logits  # N x S
        return im_part_dir_logits

    def _get_repr(self, x: Minibatch):
        im_feats = x.im_data[1]
        ppl_boxes, ppl_feats, coco_kps, kp_boxes, kp_feats = x.person_data[:5]
        obj_boxes, obj_scores, obj_feats = x.obj_data[:3]

        hh = self.dataset.full_dataset
        P, M, B = self.dims.P, self.dims.M, self.dims.B
        N = ppl_boxes.shape[0]

        im_part_reprs_dir = []
        for p in hh.parts:
            im_part_reprs_dir.append(self.repr_mlps[f'img_part_{p}'](im_feats))
        im_part_reprs_dir = torch.stack(im_part_reprs_dir, dim=1)
        assert im_part_reprs_dir.shape[:-1] == (N, B)
        part_reprs = [im_part_reprs_dir.view(N, 1, 1, B, -1).expand(N, P, M, B, -1)]

        # All of the following representations are computed for each person and then aggregated to obtain one per image.
        # Part representation from keypoint bounding boxes visual features.
        ppl_part_reprs = []
        for p, kps in enumerate(hh.part_to_kp):
            # Lower representation's contribution based on score. Soft threshold is at around 0.1 (LIS with w=96 and k=10).
            kp_repr = self.repr_mlps[f'kp_part_{hh.parts[p]}'](kp_feats[:, :, kps, :])
            kp_weights = LIS(kp_boxes[:, :, kps, -1:], w=96, k=10)
            part_repr = (kp_repr * kp_weights).mean(dim=2)
            ppl_part_reprs.append(part_repr)
        ppl_part_reprs = torch.stack(ppl_part_reprs, dim=2)
        assert ppl_part_reprs.shape[:-1] == (N, P, B)
        part_reprs.append(ppl_part_reprs.unsqueeze(dim=2).expand(N, P, M, B, -1))
        self.cache['part_reprs'] = ppl_part_reprs

        # Object detection's scores
        part_reprs.append(obj_scores.unsqueeze(dim=1).unsqueeze(dim=3).expand(N, P, M, B, -1))

        part_reprs = torch.cat(part_reprs, dim=-1)
        assert part_reprs.shape == (N, P, M, B, self.final_part_repr_dim)

        return part_reprs


class FrozenPartStateBranch(PartStateBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.part_cache = {}
        self.starting_epoch = None

    def _forward(self, x: Minibatch):
        ex_ids = x.ex_data[0]
        epoch_idx = x.other[0]
        if self.starting_epoch is None:
            self.starting_epoch = epoch_idx
        if epoch_idx > self.starting_epoch and all([exid in self.part_cache for exid in ex_ids]):
            # The second condition might not be true in epochs after the first because of the drop_last option of the data loader.
            dir_logits, part_reprs = zip(*[self.part_cache[exid] for exid in ex_ids])
            im_part_dir_logits = torch.stack(dir_logits, dim=0)
            all_part_reprs = torch.stack(part_reprs, dim=0)
            self.cache['part_dir_logits'] = im_part_dir_logits
            self.cache['part_reprs'] = all_part_reprs
        else:
            im_part_dir_logits = super()._forward(x)
            all_part_reprs = self.cache['part_reprs']

            # Detach
            all_part_reprs.detach_()  # This HAS to be done in place because this is saved into the cache
            im_part_dir_logits.detach_()

            # Save for later
            for i, exid in enumerate(ex_ids):
                assert exid not in self.part_cache or epoch_idx > self.starting_epoch
                dir_logits = im_part_dir_logits[i, :]
                part_reprs = all_part_reprs[i, :, :]
                self.part_cache[exid] = [dir_logits, part_reprs]
        return im_part_dir_logits


class ZSBranch(AbstractBranch):
    def __init__(self, label_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ld = self._get_label_data(label_type=label_type)

        self._add_mlp(name='repr', input_dim=self.dims.F_img, output_dim=self.repr_dim)
        self._add_linear_layer(name='dir', input_dim=self.repr_dim, output_dim=self.ld.num_classes)

        if self.zs_enabled:
            self.seen_inds = nn.Parameter(torch.tensor(self.ld.seen_classes), requires_grad=False)
            self.unseen_inds = nn.Parameter(torch.tensor(self.ld.unseen_classes), requires_grad=False)
            self._init_predictor_mlps()
        self.train_seen_inds = self._get_train_seen_inds()

        self.unseen_loss_coeff = 0 if self.ld.tag == 'obj' else cfg.awsu
        if self.unseen_loss_coeff > 0:  # using weak supervision for unseen actions/interactions
            wembs = self.cache.word_embs.get_embeddings(self.ld.all_classes_str, retry='avg')
            self.wemb_sim = nn.Parameter(torch.from_numpy(wembs @ wembs.T).clamp(min=0), requires_grad=False)
            self.obj_act_feasibility = nn.Parameter(self.cache['oa_adj'], requires_grad=False)

    def _get_train_seen_inds(self):
        if self.zs_enabled:
            train_seen_inds = self.seen_inds
        else:
            train_seen_inds = None

        if not cfg.train_null_act:
            if self.ld.tag == 'hoi':
                fg_interactions = np.flatnonzero(self.dataset.full_dataset.interactions[:, 0] > 0)
                train_seen_inds = nn.Parameter(torch.from_numpy(np.intersect1d(self.ld.seen_classes, fg_interactions)), requires_grad=False)
                assert len(train_seen_inds) == self.dataset.num_interactions - self.dims.O  # #seen - #objects
            elif self.ld.tag == 'act':
                train_seen_inds = nn.Parameter(train_seen_inds[1:], requires_grad=False)

        return train_seen_inds

    def _get_label_data(self, label_type=None) -> LabelData:
        if label_type == 'obj':
            label_trasf_mat = torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat)
            return LabelData(tag=label_type,
                             seen_classes=self.dataset.seen_objects,
                             unseen_classes=np.setdiff1d(np.arange(self.dims.O), self.dataset.seen_objects),
                             label_f=lambda hoi_labels: (hoi_labels @ label_trasf_mat.to(hoi_labels)).clamp(min=0, max=1).detach(),
                             num_classes=self.dims.O,
                             all_classes_str=self.dataset.full_dataset.objects)
        elif label_type == 'act':
            label_trasf_mat = torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat)
            return LabelData(tag='act',
                             seen_classes=self.dataset.seen_actions,
                             unseen_classes=np.setdiff1d(np.arange(self.dims.A), self.dataset.seen_actions),
                             label_f=lambda hoi_labels: (hoi_labels @ label_trasf_mat.to(hoi_labels)).clamp(min=0, max=1).detach(),
                             num_classes=self.dims.A,
                             all_classes_str=self.dataset.full_dataset.actions)
        elif label_type == 'hoi':
            return LabelData(tag='hoi',
                             seen_classes=self.dataset.seen_interactions,
                             unseen_classes=np.setdiff1d(np.arange(self.dims.C), self.dataset.seen_interactions),
                             label_f=lambda hoi_labels: hoi_labels.clamp(min=0, max=1).detach(),
                             num_classes=self.dims.C,
                             all_classes_str=self.dataset.full_dataset.interactions_str)
        else:
            raise NotImplementedError

    def _init_predictor_mlps(self):
        pass

    def _get_losses(self, x: Minibatch, output, **kwargs):
        interaction_labels = x.ex_labels
        dir_logits, zs_logits = output
        labels = self.ld.label_f(interaction_labels)
        assert dir_logits.shape[1] == labels.shape[1]
        train_seen_inds = self.train_seen_inds
        if self.zs_enabled:
            assert train_seen_inds is not None
            assert zs_logits.shape[1] == labels.shape[1]
            losses = {f'{self.ld.tag}_loss_seen': bce_loss(dir_logits[:, train_seen_inds], labels[:, train_seen_inds]) +
                                                  bce_loss(zs_logits[:, train_seen_inds], labels[:, train_seen_inds])}
            if self.unseen_loss_coeff > 0:
                unseen_class_labels = self.get_unseen_labels(interaction_labels)
                losses[f'{self.ld.tag}_loss_unseen'] = self.unseen_loss_coeff * bce_loss(zs_logits[:, self.unseen_inds], unseen_class_labels)
        else:
            if train_seen_inds is not None:
                dir_logits = dir_logits[:, train_seen_inds]
                labels = labels[:, train_seen_inds]
            losses = {f'{self.ld.tag}_loss': bce_loss(dir_logits, labels)}
        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        dir_logits, zs_logits = output
        logits = dir_logits
        if self.zs_enabled:
            logits[:, self.unseen_inds] = zs_logits[:, self.unseen_inds]
        if cfg.merge_dir:
            logits[:, self.seen_inds] = (dir_logits[:, self.seen_inds] + zs_logits[:, self.seen_inds]) / 2
        return torch.sigmoid(logits).cpu().numpy()

    def _forward(self, x: Minibatch):
        repr = self._get_repr(x)
        self.cache[f'{self.ld.tag}_repr'] = repr
        dir_logits = self._reduce_logits(repr @ self._get_dir_predictor())
        if self.zs_enabled:
            zs_predictor = self._get_zs_predictor()
            zs_logits = self._reduce_logits(repr @ zs_predictor)
        else:
            zs_logits = None
        return dir_logits, zs_logits

    def _reduce_logits(self, logits):
        return logits

    def _get_repr(self, x: Minibatch):
        img_feats = x.ex_data[1]
        repr = self.repr_mlps['repr'](img_feats)
        return repr

    def _get_dir_predictor(self):
        return self.linear_predictors['dir']

    def _get_zs_predictor(self):
        pass

    def get_unseen_labels(self, interaction_labels: torch.Tensor):
        assert self.unseen_loss_coeff > 0
        assert not self.ld.tag == 'obj'

        batch_size = interaction_labels.shape[0]
        extended_inter_mat = interactions_to_mat(interaction_labels, hico=self.dataset.full_dataset)  # N x I -> N x O x A

        similar_acts_per_obj = torch.bmm(extended_inter_mat, self.wemb_sim.unsqueeze(dim=0).expand(batch_size, -1, -1))
        similar_acts_per_obj = similar_acts_per_obj / extended_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
        feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)

        if self.ld.tag == 'hoi':
            all_interactions = self.dataset.full_dataset.interactions
            ws_labels = feasible_similar_acts_per_obj[:, all_interactions[:, 1], all_interactions[:, 0]]
        else:
            assert self.ld.tag == 'act'
            ws_labels = feasible_similar_acts_per_obj.max(dim=1)[0]
        unseen_class_labels = ws_labels[:, self.unseen_inds]
        return unseen_class_labels.detach()


class ZSGCBranch(ZSBranch):
    def __init__(self, enable_gcn=True, *args, **kwargs):
        super().__init__(enable_zs=enable_gcn, *args, **kwargs)

    def _init_predictor_mlps(self):
        gc_latent_dim = cfg.gcldim
        self._add_mlp(name=f'oa_gcn_predictor', input_dim=gc_latent_dim, hidden_dim=(gc_latent_dim + self.repr_dim) // 2,
                      output_dim=self.repr_dim, dropout_p=cfg.gcdropout)
        if self.ld.tag == 'obj':
            word_embs = self.cache.word_embs
            obj_wembs = word_embs.get_embeddings(self.dataset.full_dataset.objects, retry='avg')
            self.word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
            self._add_mlp(name='wemb_predictor', input_dim=word_embs.dim, hidden_dim=600, output_dim=self.repr_dim)

    def _get_zs_predictor(self):
        gcn_class_embs = self.cache[f'oa_gcn_{self.ld.tag}_class_embs']
        zs_predictor = self.repr_mlps['oa_gcn_predictor'](gcn_class_embs)
        if self.ld.tag == 'obj':
            zs_predictor = zs_predictor + self.repr_mlps['wemb_predictor'](self.word_embs)
        return zs_predictor.t()


class SeenRecBranch(AbstractBranch):
    def __init__(self, hoi_repr_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.zs_enabled and cfg.no_filter_bg_only
        self._add_mlp(name='seen_rec', input_dim=hoi_repr_dim, output_dim=1)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        interaction_labels = x.ex_labels
        labels = interaction_labels[:, self.dataset.seen_interactions].sum(dim=1, keepdim=True).clamp(max=1)
        assert output.shape[1] == labels.shape[1]
        losses = {f'seen_loss': bce_loss(output, labels)}
        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        return torch.sigmoid(output).cpu().numpy()

    def _forward(self, x: Minibatch):
        repr = self.cache['hoi_repr']
        seen_logits = self.repr_mlps['seen_rec'](repr)
        return seen_logits


class FromObjBranch(ZSGCBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_mlp(name='obj_to_hoi', input_dim=self.dataset.num_objects, output_dim=self.repr_dim,
                      num_classes=self.ld.num_classes)

    def _get_repr(self, x: Minibatch):
        obj_boxes, obj_scores, obj_feats = x.obj_data[:3]
        return self.repr_mlps['obj_to_hoi'](obj_scores)

    def _reduce_logits(self, logits):
        return logits.max(dim=1)[0]


class FromPoseBranch(ZSGCBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert cfg.tin
        D = self.dims.D
        d = D // 4
        pose_ch = 32

        self.pose_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=pose_ch, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=pose_ch, out_channels=pose_ch // 2, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Output is (N, D/4, D/4, pose_ch/2)
        )
        self._add_mlp(name='from_pose', input_dim=d * d * pose_ch // 2, output_dim=self.repr_dim)

    def _get_repr(self, x: Minibatch):
        ppl_boxes, ppl_feats, coco_kps, kp_boxes, kp_feats, ipatterns = x.person_data[:6]
        P, M, D = self.dims.P, self.dims.M, self.dims.D
        N = ppl_boxes.shape[0]
        ipatterns = ipatterns.view(N * P * M, D, D, -1).permute(0, 3, 1, 2)
        pose_feats = self.pose_module(ipatterns[:, -1:, ...]).view(N * P * M, -1).view(N, P, M, -1)
        return self.repr_mlps['from_pose'](pose_feats)

    def _reduce_logits(self, logits):
        return logits.max(dim=2)[0].max(dim=1)[0]


class FromBoxesBranch(ZSGCBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_mlp(name='from_boxes', input_dim=self.dims.F_kp + self.dims.F_obj, output_dim=self.repr_dim)

    def _get_repr(self, x: Minibatch):
        ppl_boxes, ppl_feats, coco_kps, kp_boxes, kp_feats = x.person_data[:5]
        obj_boxes, obj_scores, obj_feats = x.obj_data[:3]
        P, M, D, F_kp, F_obj = self.dims.P, self.dims.M, self.dims.D, self.dims.F_kp, self.dims.F_obj
        interaction_feats = torch.cat([ppl_feats.unsqueeze(dim=2).expand(-1, P, M, F_kp),
                                       obj_feats.unsqueeze(dim=1).expand(-1, P, M, F_obj),
                                       ], dim=-1)
        return self.repr_mlps['from_boxes'](interaction_feats)

    def _reduce_logits(self, logits):
        return logits.max(dim=2)[0].max(dim=1)[0]


class FromSpConfBranch(ZSGCBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spconf = SpatialConfigurationModule(*args, **kwargs)
        self._add_mlp(name='from_spconf', input_dim=self.spconf.repr_dim, output_dim=self.repr_dim)

    def _get_repr(self, x: Minibatch):
        spconf_feats = self.spconf(x)
        return self.repr_mlps['from_spconf'](spconf_feats)

    def _reduce_logits(self, logits):
        return logits.max(dim=2)[0].max(dim=1)[0]


class FromPartStateLogitsCooccAttBranch(ZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hh = self.dataset.full_dataset
        fg_part_states = np.concatenate([inds[:-1] for inds in hh.states_per_part])
        inters = hh.split_img_labels[self.dataset.split]
        part_states = hh.split_part_annotations[self.dataset.split]

        part_states = part_states[:, fg_part_states]
        pstate_inters_cooccs = part_states.T @ inters
        pstate_inters_cooccs /= np.maximum(1, pstate_inters_cooccs.sum(axis=0, keepdims=True))

        self.fg_part_states = nn.Parameter(torch.from_numpy(fg_part_states).long(), requires_grad=False)
        self.pstate_inters_cooccs = nn.Parameter(torch.from_numpy(pstate_inters_cooccs).float(), requires_grad=False)

    def _forward(self, x: Minibatch):
        part_dir_logits = self.cache['part_dir_logits']
        logits = part_dir_logits[:, self.fg_part_states] @ self.pstate_inters_cooccs
        dir_logits = logits
        if self.zs_enabled:
            zs_logits = logits
        else:
            zs_logits = None
        return dir_logits, zs_logits


class FromPartStateLogitsAttBranch(ZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fg_part_states = np.concatenate([inds[:-1] for inds in self.dataset.full_dataset.states_per_part])
        self.fg_part_states = nn.Parameter(torch.from_numpy(fg_part_states).long(), requires_grad=False)
        self.S_pos = self.fg_part_states.shape[0]

        self._add_mlp(name='pstate_att_from_boxes', input_dim=self.dims.F_kp + self.dims.F_obj, output_dim=self.repr_dim,
                      num_classes=self.S_pos * self.dims.C)

    def _forward(self, x: Minibatch):
        pstate_logits = self.cache['part_dir_logits']  # N x S

        ppl_boxes, ppl_feats, coco_kps, kp_boxes, kp_feats = x.person_data[:5]
        obj_boxes, obj_scores, obj_feats = x.obj_data[:3]
        P, M, C, S, F_kp, F_obj = self.dims.P, self.dims.M, self.dims.C, self.dims.S, self.dims.F_kp, self.dims.F_obj
        interaction_feats = torch.cat([ppl_feats.unsqueeze(dim=2).expand(-1, P, M, F_kp),
                                       obj_feats.unsqueeze(dim=1).expand(-1, P, M, F_obj)
                                       ], dim=-1)

        pstate_att_coeffs = self.repr_mlps['pstate_att_from_boxes'](interaction_feats) @ self.linear_predictors['pstate_att_from_boxes']
        pstate_att = nn.functional.softmax(pstate_att_coeffs.view(-1, P, M, self.S_pos, C), dim=-2)

        pstate_logits = pstate_logits.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, P, M, S).unsqueeze(dim=-2)  # N x P x M x 1 x S
        logits = pstate_logits[..., self.fg_part_states] @ pstate_att  # N x P x M x 1 x C
        logits = logits.squeeze(dim=-2).max(dim=2)[0].max(dim=1)[0]  # N x C

        dir_logits = logits
        if self.zs_enabled:
            zs_logits = logits
        else:
            zs_logits = None
        return dir_logits, zs_logits


class FromPartStateLogitsGCNAttBranch(ZSGCBranch):
    def __init__(self, part_branch: PartStateBranch, att_repr_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.part_branch = part_branch
        hh = self.dataset.full_dataset

        fg_part_states = np.concatenate([inds[:-1] for inds in hh.states_per_part])
        self.fg_part_states = nn.Parameter(torch.from_numpy(fg_part_states).long(), requires_grad=False)
        self.S_pos = self.fg_part_states.shape[0]

        # Define state-interaction attention matrix based on dataset's co-occurrences
        inters = hh.split_img_labels[self.dataset.split]
        part_states = hh.split_part_annotations[self.dataset.split]
        part_states = part_states[:, fg_part_states]
        pstate_inters_cooccs = part_states.T @ inters  # S_pos
        pstate_inters_cooccs /= np.maximum(1, pstate_inters_cooccs.sum(axis=0, keepdims=True))
        pstate_inters_affordance = (torch.from_numpy(pstate_inters_cooccs) > 0).float()
        self.pstate_inters_affordance = nn.Parameter(pstate_inters_affordance, requires_grad=False)

        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.state_hoi_gcn = BipartiteGCN(adj_block=pstate_inters_affordance, input_dim=gc_emb_dim, gc_dims=gc_dims)

        self._add_mlp(name='pstate_decoder', input_dim=gc_latent_dim, output_dim=att_repr_dim)
        self._add_mlp(name='hoi_decoder', input_dim=gc_latent_dim, output_dim=att_repr_dim)
        self._add_mlp(name='pstate_att', input_dim=2 * att_repr_dim, output_dim=1)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        losses = super()._get_losses(x, output=output[:2])
        state_hoi_logit_att_mat = output[2]
        if cfg.gat_instance:
            interaction_labels, part_state_labels = x.ex_labels, x.pstate_labels
            labels = self.ld.label_f(interaction_labels)[:, self.train_seen_inds]  # N x C_seen
            N = labels.shape[0]
            gt_state_hoi_att_mat = (part_state_labels[:, self.fg_part_states].unsqueeze(dim=2) * labels.unsqueeze(dim=1))  # N x S_pos x C_seen
            pred_state_hoi_att_mat = state_hoi_logit_att_mat[:, self.train_seen_inds].unsqueeze(dim=0).expand(N, -1, -1)  # N x S_pos x C_seen
            losses[f'state-hoi_att_loss'] = bce_loss(pred_state_hoi_att_mat.view(N, -1), gt_state_hoi_att_mat.view(N, -1))
        else:
            pred_state_hoi_att_mat = state_hoi_logit_att_mat[:, self.train_seen_inds]  # S_pos x C_seen
            gt_state_hoi_att_mat = self.pstate_inters_affordance[:, self.train_seen_inds]  # S_pos x C_seen
            if cfg.nocspos:
                pos_weights = None
            else:
                pos_weights = 1 / gt_state_hoi_att_mat.sum(dim=1).clamp(min=1)
            losses[f'state-hoi_att_loss'] = bce_loss(pred_state_hoi_att_mat.t(), gt_state_hoi_att_mat.t(), pos_weights=pos_weights)
        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        return super()._predict(x, output=output[:2])

    def _forward(self, x: Minibatch):
        pstate_logits = self.cache['part_dir_logits']  # N x S
        P, M, S = self.dims.P, self.dims.M, self.dims.S

        pstate_att, pstate_att_unbounded = self._get_pstate_att(x)
        pstate_logits = pstate_logits.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, P, M, S).unsqueeze(dim=-2)  # N x P x M x 1 x S
        logits = pstate_logits[..., self.fg_part_states] @ pstate_att  # N x P x M x 1 x C
        logits = logits.squeeze(dim=-2).max(dim=2)[0].max(dim=1)[0]  # N x C

        dir_logits = logits
        if self.zs_enabled:
            zs_logits = logits
        else:
            zs_logits = None
        return dir_logits, zs_logits, pstate_att_unbounded

    def _get_pstate_att(self, x: Minibatch):
        pstate_repr, hoi_repr = self.state_hoi_gcn()
        pstate_att_repr = self.repr_mlps['pstate_decoder'](pstate_repr)
        hoi_att_repr = self.repr_mlps['hoi_decoder'](hoi_repr)
        att_repr = torch.cat([pstate_att_repr.unsqueeze(dim=1).expand(-1, self.dims.C, -1),
                              hoi_att_repr.unsqueeze(dim=0).expand(self.S_pos, -1, -1)
                              ], dim=-1)
        pstate_att_unbounded = self.repr_mlps['pstate_att'](att_repr).squeeze(dim=-1)  # S_pos x C
        pstate_att = nn.functional.softmax(pstate_att_unbounded, dim=0)
        return pstate_att, pstate_att_unbounded

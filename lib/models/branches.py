from typing import List, Tuple, Dict, NamedTuple, Callable, Union

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet_hake import HicoDetHake, HicoDetHakeSplit
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, Minibatch, Labels
from lib.dataset.utils import Dims, interactions_to_mat
from lib.dataset.vcoco import VCoco
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.gcns import BipartiteGCN
from lib.models.misc import bce_loss, LIS, MemoryEfficientSwish as Swish


class LabelData(NamedTuple):
    tag: str
    num_classes: int
    seen_classes: np.array
    unseen_classes: np.array
    label_f: Callable
    all_classes_str: List[str]


class Cache:
    def __init__(self):
        self.word_embs = None  # type: Union[None, WordEmbeddings]
        self.oa_adj = None
        self.aos_cooccs = None
        self.num_ao_pairs = None
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
    def __init__(self, dataset: HoiDatasetSplit, cache: Cache, repr_dims, **kwargs):
        super().__init__()
        assert isinstance(repr_dims, (Tuple, List))
        self.dataset = dataset  # type: HoiDatasetSplit
        self.part_dataset = self.dataset.full_dataset if isinstance(self.dataset, HicoDetHakeSplit) else HicoDetHake()
        self.dims = self.dataset.dims._replace(B=self.part_dataset.num_parts,
                                               B_sym=self.part_dataset.num_symparts,
                                               S=self.part_dataset.num_states,
                                               S_sym=self.part_dataset.num_symstates)  # type: Dims
        self.cache = cache
        self.repr_dims = repr_dims
        self.repr_mlps = nn.ModuleDict()
        self.linear_predictors = nn.ParameterDict()
        self.extra_infos = {}  # type: Dict[str, torch.Tensor]

    def _add_mlp(self, name, input_dim, output_dim, num_classes=None, hidden_dim=1024, dropout_p=None, override=False):
        if override:
            assert name in self.repr_mlps
        else:
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
            self._add_linear_layer(name=name, input_dim=output_dim, output_dim=num_classes, override=override)

    def _add_linear_layer(self, name, input_dim, output_dim, override=False):
        if override:
            assert name in self.linear_predictors
        else:
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
        P, M, K, B, O, F_ex, F_kp, F_obj, D = dims.P, dims.M, dims.K, dims.B, dims.O, dims.F_ex, dims.F_kp, dims.F_obj, dims.D

        N = None
        if ex_data is not None:
            ex_feats = ex_data[1]
            N = ex_feats.shape[0]
            assert ex_feats.shape == (N, F_ex)
            self.dims = self.dims._replace(N=N)  # type: Dims

        if im_data is not None:
            assert im_data[1].shape[0] == N

        if person_data is not None:
            ppl_boxes, ppl_feats, coco_kps, kp_boxes = person_data[:4]
            N = N if N is not None else ppl_boxes.shape[0]
            assert ppl_boxes.shape == (N, P, 4), ppl_boxes.shape
            assert ppl_feats.shape == (N, P, F_kp), ppl_feats.shape
            assert coco_kps.shape == (N, P, K, 3), coco_kps.shape
            assert kp_boxes.shape == (N, P, B, 5), kp_boxes.shape

            kp_feats = person_data[4]
            if kp_feats is None:
                assert cfg.no_part
            else:
                assert kp_feats.shape == (N, P, B, F_kp), kp_feats.shape

        if obj_data is not None:
            obj_boxes, obj_scores, obj_feats = obj_data[:3]
            N = N if N is not None else obj_boxes.shape[0]
            assert obj_boxes.shape == (N, M, 4), obj_boxes.shape
            assert obj_scores.shape == (N, M, O), obj_scores.shape
            assert obj_feats.shape == (N, M, F_obj), obj_feats.shape

        if cfg.tin:
            ipatterns = person_data[5]
            assert ipatterns.shape[:3] == (N, P, M) and ipatterns.shape[3:5] == (D, D) and len(ipatterns.shape) == 6

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            self._check_dims(x)
            output = self.forward_wrap(x)
            if not isinstance(output, tuple):
                output = (output,)
            output = self.aggregate_outputs(x, output)
            if isinstance(output, tuple) and len(output) == 1:
                output = output[0]
            if not inference:
                return self._get_losses(x, output)
            else:
                return self._predict(x, output)

    def aggregate_outputs(self, x: Minibatch, outputs: Tuple):
        return tuple((sum(o) / len(o)) if isinstance(o, list) else o for o in outputs)

    def forward_wrap(self, x):
        output = self._forward(x)
        if self.wrapped_branch is not None:
            # Get output lists
            if not isinstance(output, tuple):
                output = (output,)
            output_lists = tuple([o] if not isinstance(o, list) else o for o in output)  # (x, y, z) -> ([x], [y], [z])

            # Append wrapped branch's output
            wrapped_output = self.wrapped_branch.forward_wrap(x)
            if not isinstance(wrapped_output, tuple):
                wrapped_output = (wrapped_output,)
            assert len(wrapped_output) == len(output) == len(output_lists)
            for olist, wrapped_o in zip(output_lists, wrapped_output):
                olist += (wrapped_o if isinstance(wrapped_o, list) else [wrapped_o])  # append, i.e., the wrapped branch goes last

            # Remove None and substitute empty lists with None
            output_lists = tuple([o for o in ol if o is not None] for ol in output_lists)
            assert all(isinstance(ol, list) for ol in output_lists)
            output = tuple(ol if ol else None for ol in output_lists)
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
        B, F_kp, F_obj, D = self.dims.B, self.dims.F_kp, self.dims.F_obj, self.dims.D
        d = D // 4
        self.output_dim = d * d * (sp_ch // 2 + pose_ch // 2)

        self.sp_map_module = nn.Sequential(
            nn.Conv2d(in_channels=2 + B, out_channels=sp_ch, kernel_size=(5, 5), padding=2),
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
        repr_dim = self.repr_dims[0]

        self.label_inds = nn.Parameter(torch.from_numpy(self._get_label_inds()), requires_grad=False)
        self.predictor = nn.Linear(in_features=repr_dim, out_features=self.label_inds.shape[0])

        # Keeping the same names as the official TIN implementation (although this is a simplified version).
        self.fc8_binary1 = nn.Sequential(nn.Linear(in_features=self.spconf.output_dim + F_kp, out_features=repr_dim),
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
        ppl_feats = x.person_data[1]
        obj_scores, obj_feats = x.obj_data[1:3]
        P, M, D = self.dims.P, self.dims.M, self.dims.D
        N = ppl_feats.shape[0]

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


class PartUninteractivenessBranch(SpatialConfigurationBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self.dataset  # type: HicoDetHakeSplit

    def _get_label_inds(self):
        return np.array([states[-1] for states in self.dataset.full_dataset.states_per_part])  # null part states

    def _get_losses(self, x: Minibatch, output, **kwargs):
        labels = x.labels.pstate[:, self.label_inds]
        losses = {f'part_bg_loss': bce_loss(output, labels)}
        return losses

    def _reduce_logits(self, logits):
        return logits.min(dim=2)[0].min(dim=1)[0]


class PartStateBranch(AbstractBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parts = self.part_dataset.parts
        self._states_per_part = self.part_dataset.states_per_part

        self.final_part_repr_dim = None
        if cfg.tin:
            self.uninteractiveness_branch = PartUninteractivenessBranch(dataset=self.dataset, cache=self.cache,
                                                                        repr_dim=256, sp_ch=32, pose_ch=16)
        for i, p_states in enumerate(self._states_per_part):
            part_name = self._parts[i]
            self._add_mlp(name=f'ex_{part_name}', input_dim=self.dims.F_ex, output_dim=self.repr_dims[1])
            self._add_mlp(name=f'part_{part_name}', input_dim=self.dims.F_kp, output_dim=self.repr_dims[1])
            logit_input_dim = self.repr_dims[1] + self.dims.O + self.repr_dims[1]  # direct repr + obj scores + aggr people repr

            if self.final_part_repr_dim is None:
                self.final_part_repr_dim = logit_input_dim
            else:
                assert self.final_part_repr_dim == logit_input_dim
            self._add_linear_layer(name=f'dir_ex_{part_name}', input_dim=logit_input_dim, output_dim=p_states.size)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        if not self.training:
            return {}
        orig_part_state_labels = x.labels.pstate
        part_dir_logits = output
        bg_part_states = torch.from_numpy(np.array([app[-1] for app in self._states_per_part])).to(device=part_dir_logits.device)
        fg_part_states = torch.from_numpy(np.concatenate([app[:-1] for app in self._states_per_part])).to(device=part_dir_logits.device)
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
        part_state_scores = np.zeros((x.ex_data[1].shape[0], self.part_dataset.num_states))  # this has to be like this! Don't use dims.N
        if part_dir_logits is not None:
            part_state_scores = torch.sigmoid(part_dir_logits).cpu().numpy()
            if cfg.tin:
                part_state_scores[:, self.uninteractiveness_branch.label_inds] = self.uninteractiveness_branch(x, inference=True)
        return part_state_scores

    def _forward(self, x: Minibatch):
        part_reprs = self._get_repr(x)
        part_logits_list = []
        for part_idx, p in enumerate(self._parts):
            p_logits_per_pair = part_reprs[..., part_idx, :] @ self.linear_predictors[f'dir_ex_{p}']  # N x P x M x S_part
            p_logits = p_logits_per_pair.max(dim=2)[0].max(dim=1)[0]  # N x S_part
            p_logits[:, -1] = p_logits_per_pair[..., -1].min(dim=2)[0].min(dim=1)[0]  # if ANYONE is interacting, NULL shouldn't be predicted (= min)
            part_logits_list.append(p_logits)
        im_part_dir_logits = torch.cat(part_logits_list, dim=-1)  # N x S
        self.cache['part_dir_logits'] = im_part_dir_logits  # N x S
        return im_part_dir_logits

    def _get_repr(self, x: Minibatch):
        ex_feats = x.ex_data[1]
        ppl_boxes, _, _, kp_boxes, kp_feats = x.person_data[:5]
        obj_scores = x.obj_data[1]
        assert kp_feats is not None

        P, M, B = self.dims.P, self.dims.M, self.dims.B
        N = ppl_boxes.shape[0]

        part_reprs_dir = []
        for p in self._parts:
            part_reprs_dir.append(self.repr_mlps[f'ex_{p}'](ex_feats))
        part_reprs_dir = torch.stack(part_reprs_dir, dim=1)
        assert part_reprs_dir.shape[:-1] == (N, B)
        part_reprs = [part_reprs_dir.view(N, 1, 1, B, -1).expand(N, P, M, B, -1)]

        # All of the following representations are computed for each person and then aggregated to obtain one per image.
        ppl_part_reprs = []
        for p, p_str in enumerate(self._parts):  # a different MLP per part
            # Lower representation's contribution based on score. Soft threshold is at around 0.1 (LIS with w=96 and k=10).
            p_repr = self.repr_mlps[f'part_{p_str}'](kp_feats[:, :, p, :])
            p_weights = LIS(kp_boxes[:, :, p, -1:], w=96, k=10)
            ppl_part_reprs.append(p_repr * p_weights)
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

    def train(self, mode=True):
        super().train(mode=False)

    def _forward(self, x: Minibatch):
        ex_ids = x.ex_data[0]
        epoch_idx = x.epoch
        if self.starting_epoch is None:
            self.starting_epoch = epoch_idx
        try:
            if epoch_idx is not None and epoch_idx > self.starting_epoch:
                raise KeyError  # hack

            try:
                dir_logits, part_reprs = zip(*[self.part_cache[exid] for exid in ex_ids])

                # dir_logits, part_reprs = [], []
                # for exid in ex_ids:
                #     entry = self.part_cache[exid]
                #     dir_logits.append(entry[0])
                #     part_reprs.append(entry[1])
            except KeyError:
                # Keys might be missing in epochs after the first because of the drop_last option of the data loader.
                raise

            im_part_dir_logits = torch.stack(dir_logits, dim=0)
            all_part_reprs = torch.stack(part_reprs, dim=0)
            self.cache['part_dir_logits'] = im_part_dir_logits
            self.cache['part_reprs'] = all_part_reprs
        except KeyError:
            im_part_dir_logits = super()._forward(x)
            all_part_reprs = self.cache['part_reprs']

            # Detach
            all_part_reprs.detach_()  # This HAS to be done in place because this is saved into the cache
            im_part_dir_logits.detach_()

            # Save for later
            for i, exid in enumerate(ex_ids):
                assert exid not in self.part_cache or epoch_idx is None or epoch_idx > self.starting_epoch
                dir_logits = im_part_dir_logits[i, :]
                part_reprs = all_part_reprs[i, :, :]
                self.part_cache[exid] = [dir_logits, part_reprs]
        return im_part_dir_logits


class ActZSBranch(AbstractBranch):
    def __init__(self, use_pstates=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_pstates = use_pstates
        self.ld = self._get_label_data()

        self.input_vec_dim, self.input_vec_f = self._get_input_vec_func()
        self._add_mlp(name='act_repr', input_dim=self.input_vec_dim, output_dim=self.repr_dims[0])
        self._add_linear_layer(name='dir', input_dim=self.repr_dims[0], output_dim=self.ld.num_classes)

        if self.zs_enabled:
            self.seen_inds = nn.Parameter(torch.tensor(self.ld.seen_classes), requires_grad=False)
            self.unseen_inds = nn.Parameter(torch.tensor(self.ld.unseen_classes), requires_grad=False)

            self.unseen_loss_coeff = cfg.awsu
            if self.unseen_loss_coeff > 0:  # using weak supervision for unseen actions/interactions
                wembs = self.cache.word_embs.get_embeddings(self.ld.all_classes_str, retry='avg')
                self.wemb_sim = nn.Parameter(torch.from_numpy(wembs @ wembs.T).clamp(min=0), requires_grad=False)
                self.obj_act_feasibility = nn.Parameter(self.cache.oa_adj, requires_grad=False)

    def _get_input_vec_func(self):
        input_vec_dim = self.dims.F_img + self.dims.F_ex + 2 * self.dims.O
        if self.use_pstates:
            input_vec_dim = input_vec_dim + self.dims.S_sym

        def input_vec_f(x: Minibatch):
            img_feats = x.im_data[1]
            ex_feats = x.ex_data[1]
            obj_scores = x.obj_data[1].squeeze(dim=1)
            im_obj_scores = x.im_data[3]
            input_vec = torch.cat([img_feats, ex_feats, obj_scores, im_obj_scores], dim=1)
            if self.use_pstates:
                symstate_logits = self._get_symstate_logits()  # N x S_sym
                input_vec = torch.cat([input_vec, symstate_logits], dim=1)
            return input_vec

        return input_vec_dim, input_vec_f

    def _get_symstate_logits(self):
        state_logits = self.cache['part_dir_logits']
        symstate_logits = state_logits.new_zeros((state_logits.shape[0], self.part_dataset.num_symstates))
        for i, js in enumerate(self.part_dataset.symstates_inds):
            symstate_logits[:, i] = state_logits[:, js].max(dim=1)[0]  # OR -> max
        return symstate_logits

    def _get_label_data(self) -> LabelData:
        label_trasf_mat = torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat)
        act_str = self.dataset.full_dataset.actions
        if isinstance(self.dataset.full_dataset, VCoco):
            act_str = act_str[:1] + ['_'.join(a.split('_')[:-1]) for a in act_str[1:]]
        return LabelData(tag='act',
                         seen_classes=self.dataset.seen_actions,
                         unseen_classes=np.setdiff1d(np.arange(self.dims.A), self.dataset.seen_actions),
                         label_f=lambda labels: labels.act if labels.act is not None else
                         (labels.hoi @ label_trasf_mat.to(labels.hoi)).clamp(min=0, max=1).detach(),
                         num_classes=self.dims.A,
                         all_classes_str=act_str)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        logits = output
        labels = self.ld.label_f(x.labels)
        assert logits.shape[1] == labels.shape[1]

        losses = {}
        if self.zs_enabled:
            seen_inds = self.seen_inds
            if not cfg.train_null_act:
                seen_inds = seen_inds[1:]
            losses[f'act_loss_seen'] = bce_loss(logits[:, seen_inds], labels[:, seen_inds])

            if self.unseen_loss_coeff > 0:
                unseen_class_labels = self.get_unseen_labels(x.labels)
                losses[f'act_loss_unseen'] = self.unseen_loss_coeff * bce_loss(logits[:, self.unseen_inds], unseen_class_labels)
        else:  # no ZS
            if cfg.train_null_act:
                losses[f'act_loss_seen'] = bce_loss(logits, labels)
            else:
                losses[f'act_loss_seen'] = bce_loss(logits[:, 1:], labels[:, 1:])
        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        return torch.sigmoid(output).cpu().numpy()

    def _forward(self, x: Minibatch):
        vrepr = self.repr_mlps['act_repr'](self.input_vec_f(x))
        logits = vrepr @ self.linear_predictors['dir']
        return logits

    def get_unseen_labels(self, labels: Labels):
        assert self.unseen_loss_coeff > 0

        if labels.hoi is not None:
            extended_inter_mat = interactions_to_mat(labels.hoi, hico=self.dataset.full_dataset)  # N x I -> N x O x A
        else:
            assert labels.act is not None and labels.obj is not None
            extended_inter_mat = labels.obj.unsqueeze(dim=2) * labels.act.unsqueeze(dim=1)

        similar_acts_per_obj = torch.bmm(extended_inter_mat, self.wemb_sim.unsqueeze(dim=0).expand(self.dims.N, -1, -1))
        similar_acts_per_obj = similar_acts_per_obj / extended_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
        feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(self.dims.N, -1, -1)

        ws_labels = feasible_similar_acts_per_obj.max(dim=1)[0]
        unseen_class_labels = ws_labels[:, self.unseen_inds]
        assert torch.numel(unseen_class_labels) > 0
        return unseen_class_labels.detach()


class ActZSGCNBranch(ActZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(use_pstates=False, *args, **kwargs)
        self._add_mlp(name='ho_subj_repr', input_dim=self.dims.F_kp, output_dim=self.repr_dims[0])
        self._add_mlp(name='ho_obj_repr', input_dim=self.dims.F_kp + self.dims.O, output_dim=self.repr_dims[0])
        self._add_mlp(name='act_repr', input_dim=self.dims.F_ex, output_dim=self.repr_dims[0], override=True)
        # Note: if ZS is not enabled, the GCN is not used, so it is a bit of a misnomer...
        if self.zs_enabled:
            gc_emb_dim = 1024
            gc_latent_dim = 1024
            self.affordance_gcn = BipartiteGCN(adj_block=self.cache.oa_adj, input_dim=gc_emb_dim,
                                               gc_dims=[(gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim])
            self._add_mlp(name=f'oa_gcn_predictor', input_dim=gc_latent_dim, hidden_dim=(gc_latent_dim + self.repr_dims[0]) // 2,
                          output_dim=self.repr_dims[0], dropout_p=0.5)

            self.vv_adj = nn.Parameter((self.cache.oa_adj.t() @ self.cache.oa_adj).clamp(max=1).bool(), requires_grad=False)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        dir_logits = output[0]
        labels = self.ld.label_f(x.labels)
        assert dir_logits.shape[1] == labels.shape[1]

        losses = {}
        if self.zs_enabled:
            gcn_logits = output[1]
            assert gcn_logits.shape[1] == labels.shape[1]
            seen_inds = self.seen_inds
            if not cfg.train_null_act:
                seen_inds = seen_inds[1:]
            losses[f'act_loss_seen'] = bce_loss(dir_logits[:, seen_inds], labels[:, seen_inds]) + \
                                       bce_loss(gcn_logits[:, seen_inds], labels[:, seen_inds])

            if self.unseen_loss_coeff > 0:
                unseen_class_labels = self.get_unseen_labels(x.labels)
                losses[f'act_loss_unseen'] = self.unseen_loss_coeff * bce_loss(gcn_logits[:, self.unseen_inds], unseen_class_labels)

            try:
                losses['act_reg_loss'] = self.cache['reg_loss']
            except KeyError:
                pass
        else:
            if cfg.train_null_act:
                losses[f'act_loss_seen'] = bce_loss(dir_logits, labels)
            else:
                losses[f'act_loss_seen'] = bce_loss(dir_logits[:, 1:], labels[:, 1:])
        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        dir_logits, gcn_logits = output
        logits = dir_logits
        if self.zs_enabled:
            logits[:, self.unseen_inds] = gcn_logits[:, self.unseen_inds]
        return torch.sigmoid(logits).cpu().numpy()

    def _forward(self, x: Minibatch):
        ho_subj_repr = self.repr_mlps['ho_subj_repr'](x.person_data[1].squeeze(dim=1))
        ho_obj_repr = self.repr_mlps['ho_obj_repr'](torch.cat([x.obj_data[2].squeeze(dim=1),  x.obj_data[1].squeeze(dim=1)], dim=1))
        act_repr = self.repr_mlps['act_repr'](x.ex_data[1])
        hoi_repr = ho_subj_repr + ho_obj_repr + act_repr

        dir_logits = hoi_repr @ self.linear_predictors['dir']

        if self.zs_enabled and cfg.agreg > 0:
            _, gcn_act_class_embs = self.affordance_gcn()
            gcn_predictors = self.repr_mlps[f'oa_gcn_predictor'](gcn_act_class_embs).t()
            gcn_logits = hoi_repr @ gcn_predictors
            if cfg.agreg > 0:
                margin = 0.3  # FIXME?
                adj = self.vv_adj
                seen = self.seen_inds
                unseen = self.unseen_inds
                predictors = self.linear_predictors['dir'].t()

                # # Detach seen classes predictors
                # all_predictors = predictors
                # predictors_seen = predictors[seen, :].detach()
                # predictors_unseen = predictors[unseen, :]
                # predictors = torch.cat([predictors_seen, predictors_unseen], dim=0)[torch.sort(torch.cat([seen, unseen]))[1]]
                # assert (all_predictors[seen] == predictors[seen]).all() and (all_predictors[unseen] == predictors[unseen]).all()

                predictors_norm = nn.functional.normalize(predictors, dim=1)
                predictors_sim = predictors_norm @ predictors_norm.t()
                null = ~adj.any(dim=1)
                arange = torch.arange(predictors_sim.shape[0])

                predictors_sim_diff = predictors_sim.unsqueeze(dim=2) - predictors_sim.unsqueeze(dim=1)
                reg_loss_mat = (margin - predictors_sim_diff).clamp(min=0)
                reg_loss_mat[~adj.unsqueeze(dim=2).expand_as(reg_loss_mat)] = 0
                reg_loss_mat[adj.unsqueeze(dim=1).expand_as(reg_loss_mat)] = 0
                reg_loss_mat[arange, arange, :] = 0
                reg_loss_mat[arange, :, arange] = 0
                reg_loss_mat[:, arange, arange] = 0
                reg_loss_mat[null, :, :] = 0
                reg_loss_mat[:, null, :] = 0
                reg_loss_mat[:, :, null] = 0

                reg_loss_mat = reg_loss_mat[unseen, :, :]
                reg_loss = reg_loss_mat.sum() / (reg_loss_mat != 0).sum().item()
                self.cache['reg_loss'] = cfg.agreg * reg_loss
        else:
            gcn_logits = None
        return dir_logits, gcn_logits


class FromPartStateLogitsBranch(ActZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(use_pstates=True, *args, **kwargs)  # use_pstates here is actually irrelevant
        self.input_vec_dim = self.input_vec_f = None  # these should not be used
        self._add_mlp(name='from_state', input_dim=self.dims.S_sym, output_dim=self.repr_dims[1], num_classes=self.ld.num_classes)

    def _forward(self, x: Minibatch):
        symstate_logits = self._get_symstate_logits()  # N x S_sym
        dir_logits = self.repr_mlps['from_state'](symstate_logits) @ self.linear_predictors['from_state']
        if self.zs_enabled:
            zs_logits = dir_logits
        else:
            zs_logits = None
        return dir_logits, zs_logits


class LogicBranch(ActZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_mlp(name='necessary', input_dim=self.input_vec_dim, output_dim=self.repr_dims[0], num_classes=self.ld.num_classes)
        self._add_mlp(name='sufficient', input_dim=self.input_vec_dim, output_dim=self.repr_dims[0], num_classes=self.ld.num_classes)

        thr = cfg.awslt
        assert 0 <= thr < 1
        cooccs = self.cache.aos_cooccs.sum(dim=1)  # A x S_sym
        num = self.cache.num_ao_pairs.sum(dim=1)  # A

        nec_label_mat = cooccs / num.unsqueeze(dim=1).clamp(min=1).double()
        nec_label_mat_norm = (nec_label_mat > thr).float()
        nec_label_mat_norm /= nec_label_mat_norm.sum(dim=1, keepdim=True)  # this is so all necessary conditions have to be met to give 1
        self.nec_label_mat = nn.Parameter(nec_label_mat_norm, requires_grad=False)

        suf_label_mat = cooccs / cooccs.sum(dim=0, keepdim=True).clamp(min=1)
        self.suf_label_mat = nn.Parameter((suf_label_mat > thr).float(), requires_grad=False)

        actions_with_nec_labels = (nec_label_mat > thr).any(dim=1)
        actions_with_suf_labels = (suf_label_mat > thr).any(dim=1)
        if cfg.awsl_uonly:
            actions_with_nec_labels[self.seen_inds] = 0
            actions_with_suf_labels[self.seen_inds] = 0
        self.actions_with_nec_labels = nn.Parameter(actions_with_nec_labels, requires_grad=False)
        self.actions_with_suf_labels = nn.Parameter(actions_with_suf_labels, requires_grad=False)

    def _get_losses(self, x: Minibatch, output, **kwargs):
        logits, nec_logits, suf_logits = output

        losses = super()._get_losses(x=x, output=logits)

        symstate_labels = torch.sigmoid(self._get_symstate_logits())  # N x S_sym
        nec_mask = self.actions_with_nec_labels
        suf_mask = self.actions_with_suf_labels

        w_nec = cfg.awsln
        w_nec_neg = cfg.awsln_neg
        assert w_nec_neg >= 1
        nec_labels = (symstate_labels @ self.nec_label_mat.t()).clamp(max=1)
        # w * 1/w in the following is a hack to weigh negative examples without reducing the weight on positive examples
        losses[f'act_loss_nec'] = w_nec * w_nec_neg * bce_loss(nec_logits[:, nec_mask], nec_labels[:, nec_mask], pos_weights=1 / w_nec_neg)

        w_suf = cfg.awsls
        w_suf_pos = cfg.awsls_pos
        assert w_suf_pos >= 1
        suf_labels = (symstate_labels @ self.suf_label_mat.t()).clamp(max=1)
        # suf_labels = (symstate_labels.unsqueeze(dim=1) * self.suf_label_mat.unsqueeze(dim=0)).max(dim=2)[0].clamp(max=1)
        losses[f'act_loss_suf'] = w_suf * bce_loss(suf_logits[:, suf_mask], suf_labels[:, suf_mask], pos_weights=w_suf_pos)

        return losses

    def _predict(self, x: Minibatch, output, **kwargs):
        logits, nec_logits, suf_logits = output
        logits[:, self.actions_with_nec_labels] += nec_logits[:, self.actions_with_nec_labels]
        logits[:, self.actions_with_suf_labels] += suf_logits[:, self.actions_with_suf_labels]
        return torch.sigmoid(logits).cpu().numpy()

    def _forward(self, x: Minibatch):
        dir_logits = self.repr_mlps['act_repr'](self.input_vec_f(x)) @ self.linear_predictors['dir']
        nec_logits = self.repr_mlps['necessary'](self.input_vec_f(x)) @ self.linear_predictors['necessary']
        suf_logits = self.repr_mlps['sufficient'](self.input_vec_f(x)) @ self.linear_predictors['sufficient']
        return dir_logits, nec_logits, suf_logits


class AttBranch(ActZSBranch):
    def __init__(self, part_repr_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_input_vec_func(self):
        input_vec_dim, _input_vec_f = super()._get_input_vec_func()
        self._add_mlp(name='self_att', input_dim=input_vec_dim, output_dim=input_vec_dim)

        def input_vec_f(x: Minibatch):
            base_vec = _input_vec_f(x)
            att_w = torch.sigmoid(self.repr_mlps['self_att'](base_vec))
            att_repr = base_vec * att_w
            self.cache['att_w'] = att_w

            # Note: this depends on how the input feature vector is structured, of course.
            cd = np.cumsum([0, self.dims.F_img, self.dims.F_ex, self.dims.O, self.dims.O, self.dims.S_sym])
            assert cd[0] == 0 and cd[5] == base_vec.shape[1]
            self.extra_infos['att_w'] = torch.cat([att_w[:, cd[0]:cd[1]].mean(dim=1, keepdim=True),
                                                   att_w[:, cd[1]:cd[2]].mean(dim=1, keepdim=True),
                                                   att_w[:, cd[2]:cd[3]],
                                                   att_w[:, cd[3]:cd[4]],
                                                   att_w[:, cd[4]:cd[5]],
                                                   ], dim=1)
            return att_repr

        return input_vec_dim, input_vec_f

    def _get_losses(self, x: Minibatch, output, **kwargs):
        losses = super()._get_losses(x=x, output=output)
        att_w = self.cache['att_w']

        # norm-∞ maximization. Added 1 only to get the loss between 0 and 1.
        losses[f'act_att_w_sparsity'] = cfg.att_sp_c * (1 - att_w.max(dim=1)[0].mean())
        return losses

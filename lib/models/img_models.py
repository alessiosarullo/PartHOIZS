from typing import List

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico_hake import HicoHakeSplit, HicoHakeKPSplit
from lib.models.abstract_model import AbstractModel
from lib.models.misc import bce_loss, LIS


class PartZSModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'partzs'

    def __init__(self, dataset: HicoHakeSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.repr_dim = cfg.repr_dim

        # Base model
        self.repr_mlps = nn.ModuleDict()
        for k in ['obj', 'act']:
            self.repr_mlps[k] = nn.Sequential(*[nn.Linear(self.dataset.precomputed_visual_feat_dim, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(1024, self.repr_dim),
                                                ])
            nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.repr_mlps[k][3].weight, gain=torch.nn.init.calculate_gain('linear'))

        # Predictors
        self.linear_predictors = nn.ParameterDict()
        for k, d in [('obj', dataset.full_dataset.num_objects),
                     ('act', dataset.full_dataset.num_actions)]:
            self.linear_predictors[k] = nn.Parameter(torch.empty(d, self.repr_dim), requires_grad=True)
            torch.nn.init.xavier_normal_(self.linear_predictors[k], gain=1.0)

        # Part branch
        if not cfg.no_part:
            for k in self.dataset.full_dataset.actions_per_part.keys():
                k = f'part_{k}'
                self.repr_mlps[k] = nn.Sequential(*[nn.Linear(self.dataset.precomputed_visual_feat_dim, 1024),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=cfg.dropout),
                                                    nn.Linear(1024, self.repr_dim),
                                                    ])
                nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(self.repr_mlps[k][3].weight, gain=torch.nn.init.calculate_gain('linear'))
            for k, acts in self.dataset.full_dataset.actions_per_part.items():
                k = f'part_{k}'
                self.linear_predictors[k] = nn.Parameter(torch.empty(len(acts), self.repr_dim), requires_grad=True)
                torch.nn.init.xavier_normal_(self.linear_predictors[k], gain=1.0)
            self.part_act_to_human_act = nn.Parameter(torch.empty(self.dataset.num_part_actions, self.dataset.num_actions), requires_grad=True)
            torch.nn.init.xavier_normal_(self.part_act_to_human_act, gain=1.0)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, orig_labels, orig_part_labels, other = x
            if not inference:
                epoch, iter = other
            all_logits, all_labels, all_part_labels = self._forward(feats, orig_labels, orig_part_labels)
            assert not set(all_labels.keys()) & set(all_part_labels.keys())
            for k, v in all_part_labels.items():
                all_labels[k] = v

            if not inference:
                losses = {}
                for k in all_logits.keys():
                    logits = all_logits[k]
                    labels = all_labels[k]

                    if not cfg.train_null:
                        if k == 'act':
                            labels = labels[:, 1:]
                            logits = logits[:, 1:]
                    losses[f'{k}_loss'] = bce_loss(logits, labels)

                return losses
            else:
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(all_logits['obj']).cpu().numpy()
                act_scores = torch.sigmoid(all_logits['act']).cpu().numpy()
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]]
                return prediction

    def _forward(self, feats, labels, part_labels):
        # Instance representation
        instance_repr = {k: self.repr_mlps[k](feats) for k in self.repr_mlps.keys()}
        all_logits = {k: instance_repr[k] @ self.linear_predictors[k].t() for k in self.linear_predictors.keys()}
        if not cfg.no_part:
            all_part_logits = torch.cat([all_logits[f'part_{k}'] for k in range(self.dataset.num_parts)], dim=1)
            if cfg.part_only:
                all_logits['act'] = all_part_logits @ nn.functional.softmax(self.part_act_to_human_act, dim=0)
            else:
                all_logits['act'] += all_part_logits @ nn.functional.softmax(self.part_act_to_human_act, dim=0)
                # all_logits['act'] += all_part_logits @ self.part_act_to_human_act

        # Labels
        if labels is not None:
            obj_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(labels)).clamp(max=1).detach()
            act_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(labels)).clamp(max=1).detach()
            part_labels = {f'part_{k}': part_labels[:, torch.from_numpy(v)] for k, v in self.dataset.full_dataset.actions_per_part.items()}
        else:
            obj_labels = act_labels = None
            part_labels = {f'part_{k}': None for k in self.dataset.full_dataset.actions_per_part.keys()}
        labels = {'obj': obj_labels, 'act': act_labels}
        return all_logits, labels, part_labels


class BodyPartModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        if not cfg.part_only:
            raise NotImplementedError

        self.dataset = dataset
        self.repr_dim = cfg.repr_dim
        img_feats_dim = self.dataset.precomputed_visual_feat_dim
        kp_feats_dim = self.dataset.pc_kp_feats_dim

        hicohake = self.dataset.full_dataset
        bg_part_actions = np.array([app[-1] for app in hicohake.actions_per_part])
        fg_part_actions = np.concatenate([app[:-1] for app in hicohake.actions_per_part])
        self.bg_part_actions = nn.Parameter(torch.from_numpy(bg_part_actions), requires_grad=False)
        self.fg_part_actions = nn.Parameter(torch.from_numpy(fg_part_actions), requires_grad=False)
        # objs = np.minimum(1, np.maximum(0, self.dataset.labels) @ hicohake.interaction_to_object_mat)
        # part_object_cooccs = self.dataset.part_labels.T @ objs

        # Part branches
        self.repr_mlps = nn.ModuleDict()
        self.linear_predictors = nn.ParameterDict()
        for i, p_acts in enumerate(hicohake.actions_per_part):
            part_name = hicohake.parts[i]
            self._add_repr_branch(branch_name=f'img_part_{part_name}', input_dim=img_feats_dim, repr_dim=self.repr_dim)
            logit_input_dim = self.repr_dim
            if not cfg.no_kp:
                self._add_repr_branch(branch_name=f'kp_part_{part_name}', input_dim=kp_feats_dim, repr_dim=self.repr_dim)
                logit_input_dim += self.repr_dim
            self._add_logit_head(branch_name=f'img_part_{part_name}', input_dim=logit_input_dim, num_classes=p_acts.size - 1)
            self._add_logit_head(branch_name=f'img_part_{part_name}_null', input_dim=logit_input_dim, num_classes=1)

        # Other branches
        branches_output_dims = [('obj', dataset.full_dataset.num_objects)]
        self.img_branches = sorted([k for k, d in branches_output_dims])
        for k, n_classes in branches_output_dims:
            self._add_repr_branch(branch_name=k, input_dim=img_feats_dim, repr_dim=self.repr_dim, num_classes=n_classes)

    def _add_repr_branch(self, branch_name, input_dim, repr_dim, num_classes=None):
        k = branch_name
        self.repr_mlps[k] = nn.Sequential(*[nn.Linear(input_dim, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(1024, repr_dim),
                                            ])
        nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.repr_mlps[k][3].weight, gain=torch.nn.init.calculate_gain('linear'))
        if num_classes is not None:
            self._add_logit_head(branch_name=branch_name, input_dim=repr_dim, num_classes=num_classes)

    def _add_logit_head(self, branch_name, input_dim, num_classes):
        k = branch_name
        self.linear_predictors[k] = nn.Parameter(torch.empty(input_dim, num_classes), requires_grad=True)
        torch.nn.init.xavier_normal_(self.linear_predictors[k], gain=1.0)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            # Prepare the data
            feats, person_data, obj_data, orig_labels, orig_part_labels, other = x
            if not inference:
                epoch, iter = other

            # Compute logits
            part_logits, obj_logits = self._forward(feats, person_data, obj_data, orig_labels, orig_part_labels)
            if part_logits is None:
                assert inference

            if not inference:
                if not cfg.train_null:
                    raise NotImplementedError()
                obj_labels = (orig_labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(orig_labels)
                              ).clamp(min=0, max=1).detach()

                losses = {'fg_part_loss': bce_loss(part_logits[:, self.fg_part_actions], orig_part_labels[:, self.fg_part_actions],
                                                   pos_weights=cfg.cspc if cfg.cspc > 0 else None),
                          'bg_part_loss': bce_loss(part_logits[:, self.bg_part_actions], orig_part_labels[:, self.bg_part_actions],
                                                   pos_weights=cfg.cspbgc if cfg.cspbgc > 0 else None),
                          'obj_loss': bce_loss(obj_logits, obj_labels),
                          }
                return losses
            else:
                prediction = Prediction()
                if part_logits is None:
                    prediction.part_action_scores = np.zeros((feats.shape[0], self.dataset.full_dataset.num_part_actions))
                else:
                    prediction.part_action_scores = torch.sigmoid(part_logits).cpu().numpy()
                assert prediction.part_action_scores.shape[1] == self.dataset.full_dataset.num_part_actions
                return prediction

    def _forward(self, feats, person_data, obj_data, labels, part_labels):
        im_inds, person_boxes, kp_boxes, kp_feats = person_data
        hh = self.dataset.full_dataset

        obj_logits = self.repr_mlps['obj'](feats) @ self.linear_predictors['obj']

        part_reprs = []
        for p in hh.parts:
            part_reprs.append(self.repr_mlps[f'img_part_{p}'](feats))
        part_reprs = torch.stack(part_reprs, dim=1)

        if not cfg.no_kp:
            if im_inds:
                part_kp_reprs_per_person = []
                for p, kps in enumerate(hh.part_to_kp):
                    # Lower representation's contribution based on score. Soft threshold is at around 0.1 (LIS with w=96 and k=10).
                    weights = LIS(kp_boxes[:, kps, -1:], w=96, k=10)
                    kp_reprs = (self.repr_mlps[f'kp_part_{hh.parts[p]}'](kp_feats[:, kps, :]) * weights).mean(dim=1)
                    part_kp_reprs_per_person.append(kp_reprs)
                part_kp_reprs_per_person = torch.stack(part_kp_reprs_per_person, dim=1)
                part_kp_reprs = torch.stack([part_kp_reprs_per_person[iids].mean(dim=0) for iids in im_inds], dim=0)
            else:
                part_kp_reprs = part_reprs.new_zeros(part_reprs.shape)
            part_reprs = torch.cat([part_reprs, part_kp_reprs], dim=2)
        assert part_reprs.shape[0] == feats.shape[0] and part_reprs.shape[1] == hh.num_parts

        part_logits = []
        for i, p in enumerate(hh.parts):
            repr = part_reprs[:, i, :]
            p_logits = repr @ self.linear_predictors[f'img_part_{p}']
            p_interactiveness_logits = repr @ self.linear_predictors[f'img_part_{p}_null']
            part_logits.append(torch.cat([p_logits, p_interactiveness_logits], dim=1))
        im_part_logits = torch.cat(part_logits, dim=1)

        # # Make NULL exclusive with other interactions (in a soft way).
        # for p, acts in enumerate(hh.actions_per_part):
        #     im_part_logits[:, acts[:-1]] -= im_part_logits[:, [acts[-1]]]

        return im_part_logits, obj_logits

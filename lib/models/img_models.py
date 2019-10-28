from typing import List

import torch
import torch.nn as nn

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico_hake import HicoHakeSplit
from lib.models.abstract_model import AbstractModel
from lib.models.misc import bce_loss


class SKZSMultiModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'partzs'

    def __init__(self, dataset: HicoHakeSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset = dataset

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
        for k in self.dataset.full_dataset.part_action_dict.keys():
            k = f'part_{k}'
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
        for k, acts in self.dataset.full_dataset.part_action_dict.items():
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
                obj_scores = torch.sigmoid(all_logits['obj']).cpu().numpy() ** cfg.osc
                act_scores = torch.sigmoid(all_logits['act']).cpu().numpy() ** cfg.asc
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]]
                return prediction

    def _forward(self, feats, labels, part_labels):
        # Instance representation
        instance_repr = {k: self.repr_mlps[k](feats) for k in self.repr_mlps.keys()}
        all_logits = {k: instance_repr[k] @ self.linear_predictors[k].t() for k in self.linear_predictors.keys()}
        all_part_logits = torch.cat([all_logits[f'part_{k}'] for k in range(self.dataset.num_parts)], dim=1)
        all_logits['act'] += all_part_logits @ self.part_act_to_human_act

        # Labels
        if labels is not None:
            obj_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(labels)).clamp(max=1).detach()
            act_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(labels)).clamp(max=1).detach()
            part_labels = {f'part_{k}': part_labels[:, torch.from_numpy(v)] for k, v in self.dataset.full_dataset.part_action_dict.items()}
        else:
            obj_labels = act_labels = None
            part_labels = {f'part_{k}': None for k in self.dataset.full_dataset.part_action_dict.keys()}
        labels = {'obj': obj_labels, 'act': act_labels}
        return all_logits, labels, part_labels

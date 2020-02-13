from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hico_hake import HicoHakeKPSplit
from lib.dataset.utils import interactions_to_mat
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.misc import bce_loss, LIS, MemoryEfficientSwish as Swish



class PartToHoiAttGcnActionBranch(GcnActionBranch):
    def __init__(self, part_repr_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for part_name in self.dataset.full_dataset.parts:
            self._add_mlp(name=f'img_part_to_act_{part_name}',
                          input_dim=part_repr_dim,
                          hidden_dim=2048,
                          output_dim=self.repr_dim)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        im_ids, feats, orig_img_wh = img_data

        im_part_reprs = self.cache['im_part_reprs']
        repr = self.repr_mlps['repr'](feats)  # N x D
        x = torch.stack([self.repr_mlps[f'img_part_to_act_{p_n}'](im_part_reprs[:, p_i, :])
                         for p_i, p_n in enumerate(self.dataset.full_dataset.parts)], dim=2)  # N x D x P

        att_coeffs = nn.functional.softmax(torch.bmm(repr.unsqueeze(dim=1), x).squeeze(dim=1), dim=-1)  # N x P
        repr = repr + (att_coeffs.unsqueeze(dim=1) * x).sum(dim=2)

        im_act_dir_logits = repr @ self.linear_predictors['dir']
        if self.zs_enabled:
            predictor = self.repr_mlps['oa_gcn_predictor'](self.cache[f'oa_gcn_{self.ld.tag}_class_embs'])
            im_act_zs_logits = repr @ predictor.t()
        else:
            im_act_zs_logits = None

        return im_act_dir_logits, im_act_zs_logits


class PartToHoiDualGcnActionBranch(PartToHoiGcnActionBranch):
    def __init__(self, part_repr_dim, *args, **kwargs):
        super().__init__(part_repr_dim, use_ap_gcn=True, *args, **kwargs)

    def _get_losses(self, x, output, **kwargs):
        img_data, person_data, obj_data, orig_labels, orig_part_labels, other = x
        dir_logits, zs_logits_from_obj, zs_logits_from_part = output

        act_labels = (orig_labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(orig_labels)
                      ).clamp(min=0, max=1).detach()
        if self.zs_enabled:
            seen_inds = self.seen_inds
            if not cfg.train_null_act:
                seen_inds = seen_inds[1:]
            losses = {f'{self.ld.tag}_loss_seen': bce_loss(dir_logits[:, seen_inds], act_labels[:, seen_inds]) +
                                                  bce_loss(zs_logits_from_obj[:, seen_inds], act_labels[:, seen_inds]) +
                                                  bce_loss(zs_logits_from_part[:, seen_inds], act_labels[:, seen_inds])
                      }
        else:
            if not cfg.train_null_act:  # for part actions null is always trained, because a part needs not be relevant in an image.
                dir_logits = dir_logits[:, 1:]
                act_labels = act_labels[:, 1:]
            losses = {f'{self.ld.tag}_loss': bce_loss(dir_logits, act_labels)}
        return losses

    def _predict(self, x, output, **kwargs):
        dir_logits, zs_logits_from_obj, zs_logits_from_part = output
        zs_logits = zs_logits_from_obj + zs_logits_from_part
        logits = dir_logits
        if self.zs_enabled:
            logits[:, self.unseen_inds] = zs_logits[:, self.unseen_inds] / 2
        if cfg.merge_dir:
            logits[:, self.seen_inds] = (dir_logits[:, self.seen_inds] + zs_logits[:, self.seen_inds]) / 3
        return torch.sigmoid(logits).cpu().numpy()

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        im_ids, feats, orig_img_wh = img_data

        im_part_reprs = self.cache['im_part_reprs']
        act_repr = self.repr_mlps['repr'](feats)
        x = torch.cat([act_repr.unsqueeze(dim=1).expand(-1, im_part_reprs.shape[1], -1), im_part_reprs], dim=2)
        act_repr += torch.stack([self.repr_mlps[f'img_part_to_act_{p_n}'](x[:, p_i, :]) for p_i, p_n in enumerate(self.dataset.full_dataset.parts)],
                                dim=1).mean(dim=1)

        act_dir_logits = act_repr @ self.linear_predictors['dir']
        if self.zs_enabled:
            predictor_from_obj = self.repr_mlps['oa_gcn_predictor'](self.cache[f'oa_gcn_{self.ld.tag}_class_embs'])
            act_zs_logits_from_obj = act_repr @ predictor_from_obj.t()

            predictor_from_part = self.repr_mlps['ap_gcn_predictor'](self.cache[f'ap_gcn_{self.ld.tag}_class_embs'])
            act_zs_logits_from_part = act_repr @ predictor_from_part.t()
        else:
            act_zs_logits_from_obj = act_zs_logits_from_part = None

        return act_dir_logits, act_zs_logits_from_obj, act_zs_logits_from_part

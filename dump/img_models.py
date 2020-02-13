from typing import List

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico_hake import HicoHakeKPSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel
from lib.models.branches import Cache, \
    PartActionBranch, FrozenPartActionBranch, \
    GcnObjectBranch, PretrainedObjectBranch, \
    GcnActionBranch, PartToHoiGcnActionBranch, PartToHoiDualGcnActionBranch, PartToHoiAttGcnActionBranch, ActionFromPartWrapperBranch, \
    PartInteractivenessActionWrapperBranch
from lib.models.gcns import BipartiteGCN


class AttPart2HoiModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'attp'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = PartToHoiAttGcnActionBranch(part_repr_dim=self.branches['part'].final_part_repr_dim,
                                                           dataset=self.dataset,
                                                           cache=self.cache,
                                                           repr_dim=cfg.repr_dim)


class DualGcnPart2HoiModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'dual'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_part_branch(self):
        self.branches['part'] = PartActionBranch(enable_gcn=True, dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_act_branch(self):
        if cfg.trihead:
            self.branches['act'] = PartToHoiDualGcnActionBranch(part_repr_dim=self.branches['part'].final_part_repr_dim,
                                                                dataset=self.dataset,
                                                                cache=self.cache,
                                                                repr_dim=cfg.repr_dim)
        else:
            self.branches['act'] = PartToHoiGcnActionBranch(part_repr_dim=self.branches['part'].final_part_repr_dim,
                                                            use_ap_gcn=True,
                                                            dataset=self.dataset,
                                                            cache=self.cache,
                                                            repr_dim=cfg.repr_dim)

    def _init_gcn(self):
        super()._init_gcn()
        all_interaction_labels = self.dataset.full_dataset.split_annotations[self.dataset._data_split]  # FIXME? But how? Oracle
        all_action_labels = np.minimum(1, all_interaction_labels @ self.dataset.full_dataset.interaction_to_action_mat)
        cooccs = self.dataset.part_labels.T @ all_action_labels
        if not cfg.apgcn_link_null:
            bg_part_actions = np.array([app[-1] for app in self.dataset.full_dataset.actions_per_part])
            cooccs[bg_part_actions, :] = 0
            cooccs[:, 0] = 0
        num_occs_per_part_action = np.maximum(1, cooccs.sum(axis=0, keepdims=True))
        num_occs_per_action = np.maximum(1, cooccs.sum(axis=1, keepdims=True))
        part_actions_actions_norm_cooccs = torch.from_numpy(cooccs / np.sqrt(num_occs_per_part_action * num_occs_per_action))

        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.ap_gcn = BipartiteGCN(adj_block=part_actions_actions_norm_cooccs, input_dim=gc_emb_dim, gc_dims=gc_dims)

    def _forward(self, x: List[torch.Tensor], inference=True):
        if not cfg.part_only and self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'] = self.oa_gcn()
            self.cache['ap_gcn_part_class_embs'], self.cache['ap_gcn_act_class_embs'] = self.ap_gcn()

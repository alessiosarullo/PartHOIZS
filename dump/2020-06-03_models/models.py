from typing import Union

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.cocoa import CocoaSplit
from lib.dataset.hicodet_hake import HicoDetHake
from lib.dataset.hoi_dataset_split import Minibatch, HoiDatasetSplit
from lib.dataset.vcoco import VCocoSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel, Prediction
from lib.models.branches import Cache, AbstractModule, \
    PartStateBranch, FrozenPartStateBranch, \
    ZSGCBranch, FromPartStateLogitsBranch, AttBranch, PartStateInReprBranch, PartWeightedZSGCBranch, LogicBranch, \
    LateFusionBranch, LateFusionAttBranch
from lib.models.gcns import BipartiteGCN
from lib.models.graphs import get_vcoco_graphs, get_cocoa_graphs


class AbstractTriBranchModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError

    def __init__(self, dataset: HoiDatasetSplit, part_only=False, **kwargs):
        super().__init__(dataset, **kwargs)
        assert not (cfg.no_part and part_only)
        self.dataset = dataset
        self.part_dataset = None if cfg.no_part else HicoDetHake()  # type: Union[None, HicoDetHake]
        self.zs_enabled = (cfg.seenf >= 0)
        self.part_only = part_only
        self.predict_act = False

        self.repr_dims = [cfg.repr_dim0, cfg.repr_dim1]

        self.cache = Cache()
        self.cache.word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        if isinstance(self.dataset, (VCocoSplit, CocoaSplit)):
            if isinstance(self.dataset, VCocoSplit):
                get_graphs = get_vcoco_graphs
            else:
                assert isinstance(self.dataset, CocoaSplit)
                get_graphs = get_cocoa_graphs

            if cfg.no_part:
                oa_adj, _, _ = get_graphs(self.dataset, source_ds=HicoDetHake(), to_torch=True,
                                          ext_interactions=not cfg.oracle, sym=True)
            else:
                oa_adj, aos_cooccs, num_ao_pairs = get_graphs(self.dataset, source_ds=self.part_dataset, to_torch=True,
                                                              ext_interactions=not cfg.oracle, sym=True)
                self.cache.aos_cooccs = aos_cooccs
                self.cache.num_ao_pairs = num_ao_pairs
            self.cache.oa_adj = oa_adj

        self.branches = nn.ModuleDict()
        if not cfg.no_part:
            self._init_part_branch()
            if cfg.pbf:
                ckpt = torch.load(cfg.pbf)
                state_dict = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('branches.part.')}
                try:
                    self.load_state_dict(state_dict)
                except RuntimeError:
                    raise RuntimeError('Use --tin option.')  # FIXME
        if not self.part_only:
            if self.zs_enabled:
                print('Zero-shot enabled.')
                self._init_gcn()
            if cfg.obj:
                self._init_obj_branch()
            self._init_act_branch()
            self._init_hoi_branch()

    def _init_part_branch(self):
        if cfg.pbf:
            self.branches['part'] = FrozenPartStateBranch(dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)
        else:
            self.branches['part'] = PartStateBranch(dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)

    def _init_obj_branch(self):
        self.branches['obj'] = ZSGCBranch(label_type='obj', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)

    def _init_act_branch(self):
        pass

    def _init_hoi_branch(self):
        pass

    def _init_gcn(self):
        pass

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            self.cache.reset()
            self._forward(x, inference=inference)
            if not inference:
                losses = {}
                for branch_name, branch in self.branches.items():
                    branch_losses = branch(x, inference)
                    assert isinstance(branch_losses, dict) and not (set(losses.keys()) & set(branch_losses.keys()))
                    losses.update(branch_losses)
                return_value = losses
            else:
                return_value = self._predict(x, inference)
            for branch_name, branch in self.branches.items():
                branch = branch  # type: AbstractModule
                for k, v in branch.extra_infos.items():
                    self.extra_infos[f'{branch_name}__{k}'] = v
            return return_value

    def _predict(self, x: Minibatch, inference) -> Prediction:
        prediction = Prediction()
        if not cfg.no_part:
            prediction.part_state_scores = self.branches['part'](x, inference)
        if not self.part_only:
            interactions = self.dataset.full_dataset.interactions
            hoi_scores = []

            if 'obj' in self.branches:
                obj_scores = self.branches['obj'](x, inference)
                prediction.hoi_obj_scores = obj_scores
                hoi_scores.append(obj_scores[:, interactions[:, 1]])

            if 'act' in self.branches:
                act_scores = self.branches['act'](x, inference)
                hoi_scores.append(act_scores[:, interactions[:, 0]])
            else:
                act_scores = None

            if 'hoi' in self.branches:
                hoi_scores.append(self.branches['hoi'](x, inference))

            assert hoi_scores
            if self.predict_act:
                assert act_scores is not None
                prediction.output_scores = act_scores
            else:
                prediction.output_scores = np.prod(np.stack(hoi_scores, axis=0), axis=0)

        if len(x.ex_data) > 2:
            prediction.ho_pairs = x.ex_data[2]
            prediction.obj_boxes = x.ex_data[3]
            if prediction.hoi_obj_scores is None:
                obj_scores = x.ex_data[4]
                prediction.hoi_obj_scores = obj_scores[prediction.ho_pairs[:, 1], :]
            assert prediction.output_scores is None or prediction.ho_pairs.shape[0] == prediction.output_scores.shape[0]
            assert prediction.part_state_scores is None or prediction.ho_pairs.shape[0] == prediction.part_state_scores.shape[0]
        return prediction

    def _forward(self, x: Minibatch, inference=True):
        pass


class PartModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'part'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, part_only=True, **kwargs)


class ActModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'act'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.predict_act = True

    def _init_act_branch(self):
        if cfg.no_part:
            branch_class = ZSGCBranch
        else:
            branch_class = PartStateInReprBranch
        self.branches['act'] = branch_class(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                            use_dir_repr=not cfg.no_dir)

    def _init_gcn(self):
        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.affordance_gcn = BipartiteGCN(adj_block=self.cache.oa_adj, input_dim=gc_emb_dim, gc_dims=gc_dims)

    def _forward(self, x: Minibatch, inference=True):
        if self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'] = self.affordance_gcn()


class LogicActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'logic'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.predict_act = True

    def _init_act_branch(self):
        self.branches['act'] = LogicBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                           use_dir_repr=not cfg.no_dir)


class PartGcnActModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'pgcnact'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.predict_act = True

    def _init_act_branch(self):
        self.branches['act'] = PartWeightedZSGCBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)


class FromPStateActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompstate'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = FromPartStateLogitsBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)


class LateActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'late'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = LateFusionBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                                use_dir_repr=not cfg.no_dir)


class LateAttActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'lateatt'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = LateFusionAttBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                                   use_dir_repr=not cfg.no_dir)


class AttActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'att'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = AttBranch(part_repr_dim=self.branches['part'].repr_dims[1],
                                         label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                         use_dir_repr=not cfg.no_dir)

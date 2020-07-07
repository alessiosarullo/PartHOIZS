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
    ActZSBranch, FromPartStateLogitsBranch, \
    AttBranch, LogicBranch
from lib.models.graphs import get_vcoco_graphs, get_cocoa_graphs


class AbstractTriBranchModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError

    def __init__(self, dataset: HoiDatasetSplit, part_only=False, **kwargs):
        super().__init__(dataset, **kwargs)
        assert not (cfg.no_part and part_only)
        self.dataset = dataset
        self.part_dataset = HicoDetHake()  # type: Union[None, HicoDetHake]
        self.zs_enabled = (cfg.seenf >= 0)
        self.part_only = part_only
        self.predict_act = True

        self.repr_dims = [cfg.repr_dim0, cfg.repr_dim1]

        self.cache = Cache()
        self.cache.word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        if isinstance(self.dataset, (VCocoSplit, CocoaSplit)):
            if isinstance(self.dataset, VCocoSplit):
                get_graphs = get_vcoco_graphs
            else:
                assert isinstance(self.dataset, CocoaSplit)
                get_graphs = get_cocoa_graphs

            oa_adj, aos_cooccs, num_ao_pairs = get_graphs(self.dataset, source_ds=self.part_dataset, to_torch=True,
                                                          ext_interactions=not cfg.oracle, sym=True)
            self.cache.aos_cooccs = aos_cooccs
            self.cache.num_ao_pairs = num_ao_pairs
            self.cache.oa_adj = oa_adj

        self.branches = nn.ModuleDict()
        if not cfg.no_part:
            if cfg.pbf:
                self.branches['part'] = FrozenPartStateBranch(dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)
                ckpt = torch.load(cfg.pbf)
                state_dict = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('branches.part.')}
                try:
                    self.load_state_dict(state_dict)
                except RuntimeError:
                    raise RuntimeError('Use --tin option.')  # FIXME
            else:
                self.branches['part'] = PartStateBranch(dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)
        if not self.part_only:
            if self.zs_enabled:
                print('Zero-shot enabled.')
            self._init_act_branch()

    def _init_act_branch(self):
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

    def _init_act_branch(self):
        self.branches['act'] = ActZSBranch(use_pstates=not (cfg.no_part or cfg.no_psf),
                                           dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)


class LogicActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'logic'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        assert not cfg.no_part
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = LogicBranch(use_pstates=not cfg.no_psf,
                                           dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)


class FromPStateActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompstate'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = FromPartStateLogitsBranch(dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)


class AttActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'att'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = AttBranch(use_pstates=not (cfg.no_part or cfg.no_psf),
                                         part_repr_dim=self.branches['part'].repr_dims[1],
                                         dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)

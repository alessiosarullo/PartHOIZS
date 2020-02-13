import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico_hake import HicoHakeKPSplit, Minibatch
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel
from lib.models.branches import Cache, \
    PartActionBranch, FrozenPartActionBranch, PartStateBranch, FrozenPartStateBranch, \
    GcnObjectBranch, PretrainedObjectBranch, \
    GcnActionBranch, PartToHoiGcnActionBranch, ActionFromPartWrapperBranch, \
    GcnHoiBranch, \
    HoiUninteractivenessBranch, \
    PartInteractivenessActionWrapperBranch, TriheadActionBranch
from lib.models.gcns import BipartiteGCN, HoiGCN


class AbstractTriBranchModel(AbstractModel):
    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert not (cfg.no_part and cfg.part_only)
        self.dataset = dataset
        self.zs_enabled = (cfg.seenf >= 0)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        self.cache = Cache(word_embs=word_embs)

        self.branches = nn.ModuleDict()
        if not cfg.no_part:
            self._init_part_branch()
            if cfg.pbf:
                ckpt = torch.load(cfg.pbf)
                state_dict = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('branches.part.')}
                self.load_state_dict(state_dict)
        if not cfg.part_only:
            if self.zs_enabled:
                print('Zero-shot enabled.')
                self._init_gcn()
            self._init_obj_branch()
            self._init_act_branch()
            self._init_hoi_branch()

    def _init_part_branch(self):
        if cfg.pbf:
            self.branches['part'] = FrozenPartActionBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)
        else:
            self.branches['part'] = PartActionBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_obj_branch(self):
        if cfg.ptos:
            self.branches['obj'] = PretrainedObjectBranch(dataset=self.dataset, cache=self.cache)
        else:
            self.branches['obj'] = GcnObjectBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

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
                return losses
            else:
                return self._predict(x, inference)

    def _predict(self, x: Minibatch, inference) -> Prediction:
        prediction = Prediction()
        if not cfg.no_part:
            prediction.part_state_scores = self.branches['part'](x, inference)
            assert prediction.part_state_scores.shape[1] == self.dataset.full_dataset.num_part_states
        if not cfg.part_only:
            interactions = self.dataset.full_dataset.interactions

            obj_scores = self.branches['obj'](x, inference)
            prediction.hoi_scores = obj_scores[:, interactions[:, 1]]

            if 'act' in self.branches:
                act_scores = self.branches['act'](x, inference)
                prediction.hoi_scores *= act_scores[:, interactions[:, 0]]

            if 'hoi' in self.branches:
                hoi_scores = self.branches['hoi'](x, inference)
                prediction.hoi_scores *= hoi_scores
        return prediction

    def _forward(self, x: Minibatch, inference=True):
        pass


class BaseModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        if not cfg.part_only:
            if cfg.ptres:
                self.branches['act'] = ActionFromPartWrapperBranch(base_branch=self.branches['act'],
                                                                   dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)
            if cfg.pint:
                self.branches['act'] = PartInteractivenessActionWrapperBranch(base_branch=self.branches['act'],
                                                                              dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_act_branch(self):
        self.branches['act'] = GcnActionBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_gcn(self):
        interactions = self.dataset.full_dataset.interactions  # FIXME only "oracle" is supported ATM, even for zero-shot
        oa_adj = np.zeros([self.dataset.full_dataset.num_objects, self.dataset.full_dataset.num_actions], dtype=np.float32)
        oa_adj[interactions[:, 1], interactions[:, 0]] = 1
        oa_adj[:, 0] = 0
        oa_adj = torch.from_numpy(oa_adj)
        self.cache['oa_adj'] = oa_adj

        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.hoi_gcn = BipartiteGCN(adj_block=oa_adj, input_dim=gc_emb_dim, gc_dims=gc_dims)

    def _forward(self, x: Minibatch, inference=True):
        if not cfg.part_only and self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'] = self.hoi_gcn()


class HoiModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoi'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = GcnHoiBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_gcn(self):
        interactions = self.dataset.full_dataset.interactions  # FIXME only "oracle" is supported ATM, even for zero-shot
        oa_adj = np.zeros([self.dataset.full_dataset.num_objects, self.dataset.full_dataset.num_actions], dtype=np.float32)
        oa_adj[interactions[:, 1], interactions[:, 0]] = 1
        oa_adj[:, 0] = 0
        oa_adj = torch.from_numpy(oa_adj)
        self.cache['oa_adj'] = oa_adj

        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.hoi_gcn = HoiGCN(interactions=interactions, input_dim=gc_emb_dim, gc_dims=gc_dims)

    def _forward(self, x: Minibatch, inference=True):
        if not cfg.part_only and self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'], self.cache['oa_gcn_hoi_class_embs'] = self.hoi_gcn()


class TinModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'tin'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert cfg.tin
        super().__init__(dataset, **kwargs)
        self.no_interaction_mask = (self.dataset.full_dataset.interactions[:, 0] == 0)

    def _init_hoi_branch(self):
        super()._init_hoi_branch()
        self.branches['hoi_interactiveness'] = HoiUninteractivenessBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _predict(self, x: Minibatch, inference) -> Prediction:
        prediction = super()._predict(x, inference)
        if not cfg.part_only:
            hoi_interactiveness_scores = self.branches['hoi_interactiveness'](x, inference)
            prediction.hoi_scores[:, self.no_interaction_mask] = hoi_interactiveness_scores
        return prediction


class TinPartModel(TinModel):
    @classmethod
    def get_cline_name(cls):
        return 'tinpart'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_part_branch(self):
        if cfg.pbf:
            self.branches['part'] = FrozenPartStateBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.part_repr_dim)
        else:
            self.branches['part'] = PartStateBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.part_repr_dim)


class PTHoiModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'pt'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert not cfg.no_part
        assert cfg.pbf
        super().__init__(dataset, **kwargs)

    def _init_obj_branch(self):
        if cfg.ptos:
            self.branches['obj'] = PretrainedObjectBranch(dataset=self.dataset, cache=self.cache)
        else:
            self.branches['obj'] = GcnObjectBranch(enable_gcn=False, dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_act_branch(self):
        self.branches['act'] = ActionFromPartWrapperBranch(base_branch=None, dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _forward(self, x: Minibatch, inference=True):
        pass


class Part2HoiModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'part2hoi'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert not cfg.no_part
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = PartToHoiGcnActionBranch(part_repr_dim=self.branches['part'].final_part_repr_dim,
                                                        use_ap_gcn=False,
                                                        dataset=self.dataset,
                                                        cache=self.cache,
                                                        repr_dim=cfg.repr_dim)


class TriModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'tri'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = TriheadActionBranch(part_repr_dim=self.branches['part'].final_part_repr_dim,
                                                   use_ap_gcn=False,
                                                   dataset=self.dataset,
                                                   cache=self.cache,
                                                   repr_dim=cfg.repr_dim)

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hico_hake import HicoHakeKPSplit, Minibatch
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel, Prediction
from lib.models.branches import Cache, \
    ZSGCBranch, SeenRecBranch, \
    PartStateBranch, FrozenPartStateBranch, \
    FromObjBranch, FromPoseBranch, FromBoxesBranch, FromSpConfBranch, \
    FromPartStateLogitsAttBranch, FromPartStateLogitsCooccAttBranch, FromPartStateLogitsGCNAttBranch, \
    HoiUninteractivenessBranch
from lib.models.gcns import BipartiteGCN, HoiGCN


class AbstractTriBranchModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError

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
                try:
                    self.load_state_dict(state_dict)
                except RuntimeError:
                    raise RuntimeError('Use --tin option.')  # FIXME
        if not cfg.part_only:
            if self.zs_enabled:
                print('Zero-shot enabled.')
                self._init_gcn()
            if not cfg.no_obj:
                self._init_obj_branch()
            self._init_act_branch()
            self._init_hoi_branch()
            self._init_seen_branch()

    def _init_part_branch(self):
        if cfg.pbf:
            self.branches['part'] = FrozenPartStateBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.part_repr_dim)
        else:
            self.branches['part'] = PartStateBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.part_repr_dim)

    def _init_obj_branch(self):
        self.branches['obj'] = ZSGCBranch(label_type='obj', dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_act_branch(self):
        pass

    def _init_hoi_branch(self):
        pass

    def _init_seen_branch(self):
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
            assert prediction.part_state_scores.shape[1] == self.dataset.dims.S
        if not cfg.part_only:
            interactions = self.dataset.full_dataset.interactions
            hoi_scores = []

            if 'obj' in self.branches:
                obj_scores = self.branches['obj'](x, inference)
                prediction.obj_scores = obj_scores
                hoi_scores.append(obj_scores[:, interactions[:, 1]])

            if 'act' in self.branches:
                act_scores = self.branches['act'](x, inference)
                hoi_scores.append(act_scores[:, interactions[:, 0]])

            if 'hoi' in self.branches:
                hoi_scores.append(self.branches['hoi'](x, inference))

            assert hoi_scores
            prediction.hoi_scores = np.prod(np.stack(hoi_scores, axis=0), axis=0)
        if 'seen' in self.branches:
            prediction.seen_scores = self.branches['seen'](x, inference)
        return prediction

    def _forward(self, x: Minibatch, inference=True):
        pass


class BaseModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        self.branches['act'] = ZSGCBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)

    def _init_gcn(self):
        interactions = self.dataset.full_dataset.interactions  # FIXME only "oracle" is supported ATM, even for zero-shot
        oa_adj = np.zeros([self.dataset.dims.O, self.dataset.dims.A], dtype=np.float32)
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
        branch = ZSGCBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)
        # branch = FromPartStateLogitsBranch(label_type='hoi', wrapped_branch=branch,
        #                                    dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)
        # branch = FromBoxesBranch(label_type='hoi', wrapped_branch=branch,
        #                          dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)
        self.branches['hoi'] = branch

    def _init_seen_branch(self):
        if cfg.seen_pred:
            self.branches['seen'] = SeenRecBranch(dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim, hoi_repr_dim=cfg.repr_dim)

    def _init_gcn(self):
        interactions = self.dataset.full_dataset.interactions  # FIXME only "oracle" is supported ATM, even for zero-shot
        oa_adj = np.zeros([self.dataset.dims.O, self.dataset.dims.A], dtype=np.float32)
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


class FromObjHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'fromobj'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromObjBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromPoseHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompose'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromPoseBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromBoxesHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'frombox'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromBoxesBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromSpconfHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'fromsp'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromSpConfBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromPartStateLogitsCooccAttHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompscoatt'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert not cfg.no_part
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromPartStateLogitsCooccAttBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromPartStateAttHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompsatt'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert not cfg.no_part
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromPartStateLogitsAttBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromPartStateGCNAttHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompsgcnatt'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert not cfg.no_part
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromPartStateLogitsGCNAttBranch(part_branch=self.branches['part'],
                                                               label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)


class FromPartStateGCNAttVisHoiModel(HoiModel):
    @classmethod
    def get_cline_name(cls):
        return 'frompsgcnattvis'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        assert not cfg.no_part
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        self.branches['hoi'] = FromPartStateLogitsGCNAttVisBranch(part_branch=self.branches['part'],
                                                                  label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hico_hake import HicoHakeSplit
from lib.dataset.hoi_dataset_split import Minibatch
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel, Prediction
from lib.models.branches import Cache, \
    ZSGCBranch, \
    PartStateBranch, FrozenPartStateBranch
from lib.models.gcns import HoiGCN, BipartiteGCN


class AbstractTriBranchModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError

    def __init__(self, dataset: HicoHakeSplit, part_only=False, **kwargs):
        super().__init__(dataset, **kwargs)
        assert not (cfg.no_part and part_only)
        self.dataset = dataset
        self.zs_enabled = (cfg.seenf >= 0)
        self.part_only = part_only

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
        if not self.part_only:
            if self.zs_enabled:
                print('Zero-shot enabled.')
                self._init_gcn()
            if not cfg.no_obj:
                self._init_obj_branch()
            self._init_act_branch()
            self._init_hoi_branch()

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
        if not self.part_only:
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

            if len(x.ex_data) > 2:
                prediction.obj_boxes = x.ex_data[2]
                prediction.ho_pairs = x.ex_data[3]
                assert prediction.ho_pairs.shape[0] == prediction.hoi_scores.shape[0]
        return prediction

    def _forward(self, x: Minibatch, inference=True):
        pass


class PartModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'part'

    def __init__(self, dataset: HicoHakeSplit, **kwargs):
        super().__init__(dataset, part_only=True, **kwargs)


class ActModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'act'

    def __init__(self, dataset: HicoHakeSplit, **kwargs):
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
        if self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'] = self.hoi_gcn()


class HoiModel(AbstractTriBranchModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoi'

    def __init__(self, dataset: HicoHakeSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_hoi_branch(self):
        branch = ZSGCBranch(label_type='hoi', dataset=self.dataset, cache=self.cache, repr_dim=cfg.repr_dim)
        # branch = FromPartStateLogitsBranch(label_type='hoi', wrapped_branch=branch,
        #                                    dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)
        # branch = FromBoxesBranch(label_type='hoi', wrapped_branch=branch,
        #                          dataset=self.dataset, cache=self.cache, repr_dim=cfg.hoi_repr_dim)
        self.branches['hoi'] = branch

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
        if self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'], self.cache['oa_gcn_hoi_class_embs'] = self.hoi_gcn()

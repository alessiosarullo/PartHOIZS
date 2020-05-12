from typing import Union

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet_hake import HicoDetHake
from lib.dataset.hoi_dataset_split import Minibatch, HoiDatasetSplit
from lib.dataset.vcoco import VCocoSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel, Prediction
from lib.models.branches import Cache, AbstractModule, \
    PartStateBranch, FrozenPartStateBranch, \
    ZSGCBranch, FromPartStateLogitsBranch, AttBranch, PartStateInReprBranch, PartWeightedZSGCBranch
from lib.models.gcns import BipartiteGCN
from lib.models.graphs import get_vcoco_graphs


class AbstractTriBranchModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError

    def __init__(self, dataset: HoiDatasetSplit, part_only=False, **kwargs):
        super().__init__(dataset, **kwargs)
        assert not (cfg.no_part and part_only)
        self.dataset = dataset
        self.part_dataset = None  # type: Union[None, HicoDetHake]
        self.zs_enabled = (cfg.seenf >= 0)
        self.part_only = part_only
        self.predict_act = False

        self.repr_dims = [cfg.repr_dim0, cfg.repr_dim1]

        self.cache = Cache()
        self.cache.word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        if isinstance(self.dataset, VCocoSplit):
            oa_adj, as_cooccs, as_num_acts = get_vcoco_graphs(vcoco_split=self.dataset, source_ds=self.part_dataset, to_torch=True)
            self.cache.oa_adj = oa_adj
            self.cache.as_cooccs = as_cooccs
            self.cache.as_num_acts = as_num_acts

        self.branches = nn.ModuleDict()
        if not cfg.no_part:
            self.part_dataset = HicoDetHake()
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
                prediction.obj_scores = obj_scores
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
            prediction.obj_boxes = x.ex_data[2]
            prediction.ho_pairs = x.ex_data[3]
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
        assert isinstance(self.dataset, VCocoSplit)  # FIXME
        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.affordance_gcn = BipartiteGCN(adj_block=self.cache.oa_adj, input_dim=gc_emb_dim, gc_dims=gc_dims)

    def _forward(self, x: Minibatch, inference=True):
        if self.zs_enabled:
            self.cache['oa_gcn_obj_class_embs'], self.cache['oa_gcn_act_class_embs'] = self.affordance_gcn()


class DoubleGcnActModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return '2gcnact'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.predict_act = True

    def _init_act_branch(self):
        super()._init_act_branch()
        self.branches['act'] = ZSGCBranch(label_type='act', gcn_tag='ap', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                          wrapped_branch=self.branches['act'], use_dir_repr=False)

    def _init_gcn(self):
        super()._init_gcn()
        gc_latent_dim = cfg.gcldim
        gc_emb_dim = cfg.gcrdim
        gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
        self.pstate_gcn = BipartiteGCN(adj_block=self.cache.as_cooccs, input_dim=gc_emb_dim, gc_dims=gc_dims)

    def _forward(self, x: Minibatch, inference=True):
        super()._forward(x=x, inference=inference)
        if self.zs_enabled:
            self.cache['as_gcn_act_class_embs'], self.cache['as_gcn_pstate_class_embs'] = self.pstate_gcn()


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


class DualModel(ActModel):
    @classmethod
    def get_cline_name(cls):
        return 'dual'

    def __init__(self, dataset: HoiDatasetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _init_act_branch(self):
        pstate_branch = FromPartStateLogitsBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims)
        act_branch = PartStateInReprBranch(label_type='act', dataset=self.dataset, cache=self.cache, repr_dims=self.repr_dims,
                                           wrapped_branch=pstate_branch)
        self.branches['act'] = act_branch


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











############################################################################## BRANCH
class AttBranch(PartStateInReprBranch):
    def __init__(self, part_repr_dim, *args, **kwargs):
        raise ValueError  # obsolete
        self.part_repr_dim = part_repr_dim
        super().__init__(*args, **kwargs)

        part_to_state_norm = np.zeros((self.part_dataset.num_parts, self.part_dataset.num_states))
        for p, states in enumerate(self.part_dataset.states_per_part):
            part_to_state_norm[p, states] = 1 / states.size
        self.ps_adj = nn.Parameter(torch.from_numpy(part_to_state_norm).float(), requires_grad=False)

    def _init_repr(self):
        input_dim = self.dims.F_img + self.dims.F_ex + self.dims.S
        self._add_mlp(name='repr', input_dim=input_dim, output_dim=self.repr_dims[0])

        self._add_mlp(name='part_repr', input_dim=self.part_repr_dim, output_dim=self.repr_dims[0])
        self._add_mlp(name='part_att', input_dim=self.dims.F_ex, output_dim=self.dims.B)
        self._add_mlp(name='act_from_part_repr', input_dim=self.repr_dims[0], output_dim=self.repr_dims[0], num_classes=self.ld.num_classes)

        # self._add_mlp(name='obj_scores_att', input_dim=self.dims.F_obj, output_dim=1)

        def input_repr_f(x: Minibatch):
            img_feats = x.im_data[1]
            ex_feats = x.ex_data[1]
            prs_feats = x.person_data[1].squeeze(dim=1)
            obj_scores = x.obj_data[1].squeeze(dim=1)
            state_logits = self.cache['part_dir_logits']
            part_reprs = self.cache['part_reprs'].squeeze(dim=1)  # N x B x f0

            part_reprs = self.repr_mlps['part_repr'](part_reprs)  # N x B x f

            part_att_unbounded = self.repr_mlps['part_att'](ex_feats)  # N x B
            part_att = nn.functional.softmax(part_att_unbounded, dim=1)
            state_att = part_att @ self.ps_adj
            self.extra_infos['part_att'] = part_att
            self.extra_infos['state_att'] = state_att

            aggr_part_repr = (part_reprs * part_att.unsqueeze(dim=2)).sum(dim=1)
            act_logits_from_part = self.repr_mlps['act_from_part_repr'](aggr_part_repr) @ self.linear_predictors['act_from_part_repr']  # N x A
            self.extra_infos['act_from_part'] = act_logits_from_part

            state_logits = state_logits * state_att

            # obj_feats = x.obj_data[2].squeeze(dim=1)
            # obj_att = torch.sigmoid(self.repr_mlps['obj_scores_att'](obj_feats))
            # obj_scores = obj_scores * obj_att
            # self.extra_infos['obj_att'] = obj_att

            # vis_att = 1 - state_att
            # img_feats = img_feats * vis_att
            # ex_feats = ex_feats * vis_att

            return torch.cat([img_feats, ex_feats, state_logits], dim=1)

        return input_dim, input_repr_f

    def _get_losses(self, x: Minibatch, output, **kwargs):
        losses = super()._get_losses(x=x, output=output, **kwargs)

        # att_coeffs_u = self.extra_infos['state_att_u']
        # losses[f'state_att_loss'] = bce_loss(logits=att_coeffs_u, labels=(self.ld.label_f(x.labels) @ self.as_adj).clamp(max=1))

        # Maximise the norm 2 = promote sparsity
        loss_coeff = 0.5  # FIXME
        losses[f'pstate_att_sp_loss'] = loss_coeff * (1 - self.extra_infos['part_att'].norm(p=2, dim=1).mean())

        losses[f'act_from_part'] = bce_loss(logits=self.extra_infos['act_from_part'], labels=x.labels.act)
        return losses
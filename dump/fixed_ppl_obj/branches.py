from typing import List, NamedTuple

from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hico_hake import HicoHakeKPSplit
from lib.dataset.utils import interactions_to_mat
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.misc import bce_loss, LIS, MemoryEfficientSwish as Swish


class Cache:
    def __init__(self, word_embs: WordEmbeddings):
        self.word_embs = word_embs
        self.batch_cache = {}
        self.reset()

    def reset(self):
        self.batch_cache = {}

    def __getitem__(self, item):
        return self.batch_cache[item]

    def __setitem__(self, key, value):
        assert key not in self.batch_cache.keys()
        self.batch_cache[key] = value


class AbstractBranch(nn.Module):
    def __init__(self, dataset: HicoHakeKPSplit, cache: Cache, repr_dim=cfg.repr_dim, **kwargs):
        super().__init__()
        self.dataset = dataset  # type: HicoHakeKPSplit
        self.cache = cache
        self.repr_dim = repr_dim
        self.repr_mlps = nn.ModuleDict()
        self.linear_predictors = nn.ParameterDict()
        self.zs_enabled = (cfg.seenf >= 0)

    def _add_mlp(self, name, input_dim, output_dim, num_classes=None, hidden_dim=1024, dropout_p=cfg.dropout):
        assert name not in self.repr_mlps
        self.repr_mlps[name] = nn.Sequential(*([nn.Linear(input_dim, hidden_dim),
                                                (Swish() if cfg.swish else nn.ReLU(inplace=True))
                                                ] +
                                               ([nn.Dropout(p=dropout_p)] if dropout_p > 0 else []) +
                                               [nn.Linear(hidden_dim, output_dim)
                                                ]
                                               ))
        nn.init.xavier_normal_(self.repr_mlps[name][0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.repr_mlps[name][3 if dropout_p > 0 else 2].weight, gain=torch.nn.init.calculate_gain('linear'))
        if num_classes is not None:
            self._add_linear_layer(name=name, input_dim=output_dim, output_dim=num_classes)

    def _add_linear_layer(self, name, input_dim, output_dim):
        assert name not in self.linear_predictors
        self.linear_predictors[name] = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.linear_predictors[name], gain=1.0)

    def _get_dims(self, img_data, person_data, obj_data):
        class Dims(NamedTuple):
            N: int  # number of images
            P: int  # number of people
            M: int  # number of objects
            K_coco: int  # number of keypoints returned by the keypoint detector
            K_hake: int  # number of keypoints in HAKE
            B: int  # number of body part classes
            O: int  # number of object classes
            F_kp: int  # keypoint feature vector dimensionality
            F_obj: int  # object feature vector dimensionality

        B, O = self.dataset.full_dataset.num_parts, self.dataset.full_dataset.num_objects

        ppl_boxes, coco_kps, kp_boxes, kp_feats = person_data
        N, P, K_hake, F_kp = kp_feats.shape
        K_coco = coco_kps.shape[2]
        assert ppl_boxes.shape == (N, P, 4)
        assert coco_kps.shape == (N, P, K_coco, 3)
        assert kp_boxes.shape == (N, P, K_hake, 5)

        obj_boxes, obj_scores, obj_feats = obj_data
        M, F_obj = obj_feats.shape[1:]
        assert obj_boxes.shape == (N, M, 4)
        assert obj_scores.shape == (N, M, O)

        return Dims(N=N, P=P, M=M, K_coco=K_coco, K_hake=K_hake, B=B, O=O, F_kp=F_kp, F_obj=F_obj)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            img_data, person_data, obj_data, orig_labels, orig_part_labels, other = x
            output = self._forward(img_data, person_data, obj_data, orig_labels, orig_part_labels, other)
            if not isinstance(output, tuple):
                output = (output,)
            output = tuple((sum(o) / len(o)) if isinstance(o, list) else o for o in output)
            if isinstance(output, tuple) and len(output) == 1:
                output = output[0]
            if not inference:
                return self._get_losses(x, output)
            else:
                return self._predict(x, output)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        raise NotImplementedError()

    def _get_losses(self, x, output, **kwargs):
        raise NotImplementedError()

    def _predict(self, x, output, **kwargs):
        raise NotImplementedError()


class PartActionBranch(AbstractBranch):
    def __init__(self, enable_gcn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zs_enabled &= enable_gcn
        hicohake = self.dataset.full_dataset

        # Initialise part actions attributes according to GT.
        self.bg_part_actions = nn.Parameter(torch.from_numpy(np.array([app[-1] for app in hicohake.states_per_part])), requires_grad=False)
        self.fg_part_actions = nn.Parameter(torch.from_numpy(np.concatenate([app[:-1] for app in hicohake.states_per_part])), requires_grad=False)

        self.final_part_repr_dim = None
        for i, p_acts in enumerate(hicohake.states_per_part):
            part_name = hicohake.parts[i]
            self._add_mlp(name=f'img_part_{part_name}', input_dim=self.dataset.precomputed_visual_feat_dim, output_dim=self.repr_dim)
            logit_input_dim = self.repr_dim
            logit_input_dim += hicohake.num_objects  # OD's scores

            if not cfg.no_kp:
                self._add_mlp(name=f'kp_part_{part_name}', input_dim=self.dataset.kp_feats_dim, output_dim=self.repr_dim)
                logit_input_dim += self.repr_dim

            if cfg.use_sk:
                self._add_mlp(name=f'ppl_skeleton_for_{part_name}', input_dim=34, hidden_dim=100, output_dim=200)
                logit_input_dim += 200

            if self.final_part_repr_dim is None:
                self.final_part_repr_dim = logit_input_dim
            else:
                assert self.final_part_repr_dim == logit_input_dim
            self._add_linear_layer(name=f'img_part_{part_name}', input_dim=logit_input_dim, output_dim=p_acts.size - 1)
            self._add_linear_layer(name=f'img_part_{part_name}_null', input_dim=logit_input_dim, output_dim=1)

        if self.zs_enabled:
            gc_latent_dim = cfg.gcldim
            self._add_mlp(name='predictor_part', input_dim=gc_latent_dim, hidden_dim=(gc_latent_dim + self.final_part_repr_dim) // 2,
                          output_dim=self.final_part_repr_dim, dropout_p=cfg.gcdropout)

    def _get_losses(self, x, output, **kwargs):
        img_data, person_data, obj_data, orig_labels, orig_part_labels, other = x
        part_dir_logits, part_gcn_logits = output
        losses = {'fg_part_loss': bce_loss(part_dir_logits[:, self.fg_part_actions], orig_part_labels[:, self.fg_part_actions],
                                           pos_weights=cfg.cspc_p if cfg.cspc_p > 0 else None),
                  'bg_part_loss': bce_loss(part_dir_logits[:, self.bg_part_actions], orig_part_labels[:, self.bg_part_actions],
                                           pos_weights=cfg.cspbgc_p if cfg.cspbgc_p > 0 else None)
                  }
        if self.zs_enabled:
            losses['fg_part_loss'] += bce_loss(part_gcn_logits[:, self.fg_part_actions], orig_part_labels[:, self.fg_part_actions],
                                               pos_weights=cfg.cspc_p if cfg.cspc_p > 0 else None)
            losses['bg_part_loss'] += bce_loss(part_gcn_logits[:, self.bg_part_actions], orig_part_labels[:, self.bg_part_actions],
                                               pos_weights=cfg.cspbgc_p if cfg.cspbgc_p > 0 else None)
        return losses

    def _predict(self, x, output, **kwargs):
        img_data, person_data, obj_data, orig_labels, orig_part_labels, other = x
        im_ids, feats, orig_img_wh = img_data
        part_dir_logits, part_gcn_logits = output
        if part_dir_logits is None:
            part_action_scores = np.zeros((feats.shape[0], self.dataset.full_dataset.num_part_states))
        else:
            part_action_scores = torch.sigmoid(part_dir_logits).cpu().numpy()
        return part_action_scores

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        im_part_reprs = self._get_repr(img_data, person_data, obj_data)
        im_part_logits_list = []
        for part_idx, p in enumerate(self.dataset.full_dataset.parts):
            repr = im_part_reprs[:, part_idx, :]
            p_logits = repr @ self.linear_predictors[f'img_part_{p}']
            p_interactiveness_logits = repr @ self.linear_predictors[f'img_part_{p}_null']
            if cfg.part_bg_diff:
                p_logits = p_logits - p_interactiveness_logits
            im_part_logits_list.append(torch.cat([p_logits, p_interactiveness_logits], dim=1))
        im_part_dir_logits = torch.cat(im_part_logits_list, dim=1)

        if self.zs_enabled:
            im_part_gcn_logits_list = []
            apgcn_part_predictor = self.repr_mlps['predictor_part'](self.cache['ap_gcn_part_class_embs'])
            for part_idx, pa_inds in enumerate(self.dataset.full_dataset.states_per_part):
                im_part_gcn_logits_list.append(im_part_reprs[:, part_idx, :] @ apgcn_part_predictor[pa_inds, :].t())
            im_part_gcn_logits = torch.cat(im_part_gcn_logits_list, dim=1)
        else:
            im_part_gcn_logits = None

        # # Make NULL exclusive with other interactions (in a soft way).
        # for p, acts in enumerate(hh.actions_per_part):
        #     im_part_logits[:, acts[:-1]] -= im_part_logits[:, [acts[-1]]]

        self.cache['part_dir_logits'] = im_part_dir_logits
        self.cache['part_gcn_logits'] = im_part_gcn_logits

        return im_part_dir_logits, im_part_gcn_logits

    def _get_repr(self, img_data, person_data, obj_data):
        im_ids, feats, orig_img_wh = img_data
        ppl_boxes, coco_kps, kp_boxes, kp_feats = person_data
        obj_boxes, obj_scores, obj_feats = obj_data

        hh = self.dataset.full_dataset
        N, P, M, K_coco, K_hake, B, O, F_kp, F_obj = self._get_dims(img_data, person_data, obj_data)

        im_part_reprs_dir = []
        for p in hh.parts:
            im_part_reprs_dir.append(self.repr_mlps[f'img_part_{p}'](feats))
        im_part_reprs_dir = torch.stack(im_part_reprs_dir, dim=1)
        assert im_part_reprs_dir.shape[:-1] == (N, B)

        if not cfg.no_kp:
            # All of these are computed for each person and then aggregated to obtain one per image.
            im_part_reprs = [im_part_reprs_dir]

            # Part representation from keypoint bounding boxes visual features.
            ppl_part_reprs = []
            for p, kps in enumerate(hh.part_to_kp):
                # Lower representation's contribution based on score. Soft threshold is at around 0.1 (LIS with w=96 and k=10).
                kp_repr = self.repr_mlps[f'kp_part_{hh.parts[p]}'](kp_feats[:, :, kps, :])
                kp_weights = LIS(kp_boxes[:, :, kps, -1:], w=96, k=10)
                part_repr = (kp_repr * kp_weights).mean(dim=2)
                ppl_part_reprs.append(part_repr)
            ppl_part_reprs = torch.stack(ppl_part_reprs, dim=2)
            assert ppl_part_reprs.shape[:-1] == (N, P, B)
            im_part_reprs_from_kps = ppl_part_reprs.mean(dim=1)
            im_part_reprs.append(im_part_reprs_from_kps)

            # Object detection's scores
            im_obj_scores = obj_scores.max(dim=1)[0]
            im_part_reprs.append(im_obj_scores.unsqueeze(dim=1).expand(-1, B, -1))

            if cfg.use_sk:
                # Part representation from skeleton (simple concatenation of normalised keypoints).
                ppl_boxes_ext = ppl_boxes.unsqueeze(dim=2)  # N x P x 1 (bodyparts) x 4
                norm_coco_kps = (coco_kps[:, :, :, :2] - ppl_boxes_ext[:, :, :, :2]) / (ppl_boxes_ext[:, :, :, 2:4] - ppl_boxes_ext[:, :, :, 2])
                assert (0 <= norm_coco_kps).all() and (norm_coco_kps <= 1).all()
                skeletons = norm_coco_kps.view(N, P, -1)
                part_specific_skeleton_reprs = torch.stack([self.repr_mlps[f'ppl_skeleton_for_{p}'](skeletons) for p in hh.parts], dim=2)
                assert part_specific_skeleton_reprs.shape[:-1] == (N, P, B)
                im_part_specific_skeleton_reprs = part_specific_skeleton_reprs.mean(dim=1)
                im_part_reprs.append(im_part_specific_skeleton_reprs)

            im_part_reprs = torch.cat(im_part_reprs, dim=2)
        else:
            im_part_reprs = im_part_reprs_dir
        assert im_part_reprs.shape == (N, B, self.final_part_repr_dim)

        self.cache['im_part_reprs'] = im_part_reprs
        return im_part_reprs


class FrozenPartActionBranch(PartActionBranch):
    def __init__(self, enable_gcn=False, *args, **kwargs):
        super().__init__(enable_gcn=enable_gcn, *args, **kwargs)
        self.part_cache = {}
        self.starting_epoch = None

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        im_ids, feats, orig_img_wh = img_data
        epoch_idx = other[0]
        if self.starting_epoch is None:
            self.starting_epoch = epoch_idx
        if epoch_idx > self.starting_epoch and all([imid in self.part_cache for imid in im_ids]):
            # The second condition might not be in epochs after the first because of the drop_last option of the data loader.
            dir_logits, gcn_logits, part_reprs = zip(*[self.part_cache[imid] for imid in im_ids])
            im_part_dir_logits = torch.stack(dir_logits, dim=0)
            im_part_reprs = torch.stack(part_reprs, dim=0)

            if self.zs_enabled:
                assert not any([gcnl is None for gcnl in gcn_logits])
                im_part_gcn_logits = torch.stack(gcn_logits, dim=0)
            else:
                assert all([gcnl is None for gcnl in gcn_logits])
                im_part_gcn_logits = None

            self.cache['part_dir_logits'] = im_part_dir_logits
            self.cache['part_gcn_logits'] = im_part_gcn_logits
            self.cache['im_part_reprs'] = im_part_reprs
        else:
            im_part_dir_logits, im_part_gcn_logits = super()._forward(img_data, person_data, obj_data, labels, part_labels, other)
            im_part_reprs = self.cache['im_part_reprs']

            # Detach
            im_part_reprs.detach_()  # This HAS to be done in place because this is saved into the cache
            im_part_dir_logits.detach_()
            if im_part_gcn_logits is not None:
                im_part_gcn_logits.detach_()

            # Save for later
            for i, imid in enumerate(im_ids):
                assert imid not in self.part_cache or epoch_idx > self.starting_epoch
                dir_logits = im_part_dir_logits[i, :]
                gcn_logits = im_part_gcn_logits[i, :] if im_part_gcn_logits is not None else None
                part_reprs = im_part_reprs[i, :, :]
                self.part_cache[imid] = [dir_logits, gcn_logits, part_reprs]
        return im_part_dir_logits, im_part_gcn_logits


class ZSBranch(AbstractBranch):
    class LabelData(NamedTuple):
        tag: str
        num_classes: int
        active_classes: np.array
        trasf_mat: np.array
        all_classes_str: np.array

    def __init__(self, enable_gcn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zs_enabled &= enable_gcn
        self.ld = self._get_label_data()

        self._add_mlp(name='repr', input_dim=self.dataset.precomputed_visual_feat_dim, output_dim=self.repr_dim)
        self._add_linear_layer(name='dir', input_dim=self.repr_dim, output_dim=self.ld.num_classes)

        if self.zs_enabled:
            seen_inds = self.ld.active_classes
            unseen_inds = np.setdiff1d(np.arange(self.ld.num_classes), seen_inds)
            self.seen_inds = nn.Parameter(torch.tensor(seen_inds), requires_grad=False)
            self.unseen_inds = nn.Parameter(torch.tensor(unseen_inds), requires_grad=False)

            if cfg.seprep:
                self._add_mlp(name='zs_repr',
                              input_dim=self.dataset.precomputed_visual_feat_dim,
                              output_dim=self.repr_dim)
            self._init_predictor_mlps()

    def _get_label_data(self) -> LabelData:
        raise NotImplementedError

    def _init_predictor_mlps(self):
        gc_latent_dim = cfg.gcldim
        self._add_mlp(name=f'oa_gcn_predictor', input_dim=gc_latent_dim, hidden_dim=(gc_latent_dim + self.repr_dim) // 2,
                      output_dim=self.repr_dim, dropout_p=cfg.gcdropout)

    def _get_losses(self, x, output, **kwargs):
        img_data, person_data, obj_data, interaction_labels, part_labels, other = x
        dir_logits, zs_logits = output
        labels = (interaction_labels @ torch.from_numpy(self.ld.trasf_mat).to(interaction_labels)).clamp(min=0, max=1).detach()
        if self.zs_enabled:
            losses = {f'{self.ld.tag}_loss_seen': bce_loss(dir_logits[:, self.seen_inds], labels[:, self.seen_inds]) +
                                                  bce_loss(zs_logits[:, self.seen_inds], labels[:, self.seen_inds])}
        else:
            losses = {f'{self.ld.tag}_loss': bce_loss(dir_logits, labels)}
        return losses

    def _predict(self, x, output, **kwargs):
        dir_logits, zs_logits = output
        logits = dir_logits
        if self.zs_enabled:
            logits[:, self.unseen_inds] = zs_logits[:, self.unseen_inds]
        if cfg.merge_dir:
            logits[:, self.seen_inds] = (dir_logits[:, self.seen_inds] + zs_logits[:, self.seen_inds]) / 2
        return torch.sigmoid(logits).cpu().numpy()

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        im_ids, feats, orig_img_wh = img_data

        repr = self.repr_mlps['repr'](feats)
        dir_logits = repr @ self.linear_predictors['dir']
        if self.zs_enabled:
            if cfg.seprep:
                repr = self.repr_mlps['zs_repr'](feats)
            zs_predictor = self._get_zs_predictor()
            zs_logits = repr @ zs_predictor.t()
        else:
            zs_logits = None

        return dir_logits, zs_logits

    def _get_zs_predictor(self):
        gcn_class_embs = self.cache[f'oa_gcn_{self.ld.tag}_class_embs']
        zs_predictor = self.repr_mlps['oa_gcn_predictor'](gcn_class_embs)
        return zs_predictor


class GcnObjectBranch(ZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_label_data(self):
        return ZSBranch.LabelData(tag='obj',
                                  active_classes=self.dataset.active_objects,
                                  trasf_mat=self.dataset.full_dataset.interaction_to_object_mat,
                                  num_classes=self.dataset.full_dataset.num_objects,
                                  all_classes_str=self.dataset.full_dataset.objects)

    def _init_predictor_mlps(self):
        super()._init_predictor_mlps()
        word_embs = self.cache.word_embs
        obj_wembs = word_embs.get_embeddings(self.dataset.full_dataset.objects, retry='avg')
        self.word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self._add_mlp(name='wemb_predictor', input_dim=word_embs.dim, hidden_dim=600, output_dim=self.repr_dim)

    def _get_zs_predictor(self):
        zs_predictor = super()._get_zs_predictor() + self.repr_mlps['wemb_predictor'](self.word_embs)
        return zs_predictor


class PretrainedObjectBranch(AbstractBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.dataset.active_objects) < self.dataset.full_dataset.num_objects:
            raise ValueError('This branch cannot perform zero-shot object recognition.')

    def _get_losses(self, x, output, **kwargs):
        return {}

    def _predict(self, x, output, **kwargs):
        img_data, person_data, obj_data, orig_labels, orig_part_labels, other = x
        obj_im_inds, obj_boxes, obj_scores, obj_feats, kp_box_prox_to_obj_fmaps, obj_scores_per_kp_box = obj_data
        assert len(obj_im_inds) <= 1
        if obj_im_inds and obj_im_inds[0].size > 0:
            im_obj_scores = obj_scores[obj_im_inds[0]].max(dim=0)[0].cpu().numpy()
        else:
            im_obj_scores = np.ones(self.dataset.num_objects) / self.dataset.num_objects
        return np.atleast_2d(im_obj_scores)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        return None


class GcnHoiBranch(ZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cfg.awsu > 0:  # if using weak supervision for unseen actions
            wembs = self.cache.word_embs.get_embeddings(self.dataset.full_dataset.actions, retry='avg')
            self.act_wemb_sim = nn.Parameter(torch.from_numpy(wembs @ wembs.T).clamp(min=0), requires_grad=False)
            self.obj_act_feasibility = nn.Parameter(self.cache['oa_adj'], requires_grad=False)
        self.seen_fg_inds = nn.Parameter(torch.from_numpy(np.intersect1d(self.ld.active_classes,
                                                                         np.flatnonzero(self.dataset.full_dataset.interactions[:, 0] > 0)
                                                                         )),
                                         requires_grad=False)

    def _get_label_data(self):
        return ZSBranch.LabelData(tag='hoi',
                                  active_classes=self.dataset.active_interactions,
                                  trasf_mat=np.eye(self.dataset.full_dataset.num_interactions),
                                  num_classes=self.dataset.full_dataset.num_interactions,
                                  all_classes_str=self.dataset.full_dataset.interactions_str)

    def _get_losses(self, x, output, **kwargs):
        img_data, person_data, obj_data, interaction_labels, part_labels, other = x
        dir_logits, zs_logits = output
        labels = interaction_labels
        assert dir_logits.shape[1] == labels.shape[1]
        if self.zs_enabled:
            assert zs_logits.shape[1] == labels.shape[1]
            seen_inds = self.seen_inds
            if not cfg.train_null_act:
                seen_inds = self.seen_fg_inds
            losses = {f'{self.ld.tag}_loss_seen': bce_loss(dir_logits[:, seen_inds], labels[:, seen_inds]) +
                                                  bce_loss(zs_logits[:, seen_inds], labels[:, seen_inds])}

            if cfg.awsu > 0:
                unseen_class_labels = self.get_aff_ws_labels(interaction_labels)
                losses[f'{self.ld.tag}_loss_unseen'] = cfg.awsu * bce_loss(zs_logits[:, self.unseen_inds], unseen_class_labels)
        else:
            if not cfg.train_null_act:
                inds = self.seen_fg_inds
                assert len(inds) == self.dataset.full_dataset.num_interactions - self.dataset.full_dataset.num_objects
                dir_logits = dir_logits[:, inds]
                labels = labels[:, inds]
            losses = {f'{self.ld.tag}_loss': bce_loss(dir_logits, labels)}
        return losses

    def get_aff_ws_labels(self, interaction_labels):
        assert cfg.awsu > 0
        batch_size = interaction_labels.shape[0]
        extended_inter_mat = interactions_to_mat(interaction_labels, hico=self.dataset.full_dataset)  # N x I -> N x O x A

        similar_acts_per_obj = torch.bmm(extended_inter_mat, self.act_wemb_sim.unsqueeze(dim=0).expand(batch_size, -1, -1))
        similar_acts_per_obj = similar_acts_per_obj / extended_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
        feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)

        all_interactions = self.dataset.full_dataset.interactions
        ws_labels = feasible_similar_acts_per_obj[:, all_interactions[:, 1], all_interactions[:, 0]]
        unseen_class_labels = ws_labels[:, self.unseen_inds]
        return unseen_class_labels.detach()


class GcnActionBranch(ZSBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cfg.awsu > 0:  # if using weak supervision for unseen actions
            wembs = self.cache.word_embs.get_embeddings(self.ld.all_classes_str, retry='avg')
            self.wemb_sim = nn.Parameter(torch.from_numpy(wembs @ wembs.T), requires_grad=False)
            self.obj_act_feasibility = nn.Parameter(self.cache['oa_adj'], requires_grad=False)

    def _get_label_data(self):
        return ZSBranch.LabelData(tag='act',
                                  active_classes=self.dataset.active_actions,
                                  trasf_mat=self.dataset.full_dataset.interaction_to_action_mat,
                                  num_classes=self.dataset.full_dataset.num_actions,
                                  all_classes_str=self.dataset.full_dataset.actions)

    def _get_losses(self, x, output, **kwargs):
        img_data, person_data, obj_data, interaction_labels, part_labels, other = x
        dir_logits, zs_logits = output
        labels = (interaction_labels @ torch.from_numpy(self.ld.trasf_mat).to(interaction_labels)).clamp(min=0, max=1).detach()
        assert dir_logits.shape[1] == labels.shape[1]
        if self.zs_enabled:
            assert zs_logits.shape[1] == labels.shape[1]
            seen_inds = self.seen_inds
            if not cfg.train_null_act:
                seen_inds = seen_inds[1:]
            losses = {f'{self.ld.tag}_loss_seen': bce_loss(dir_logits[:, seen_inds], labels[:, seen_inds]) +
                                                  bce_loss(zs_logits[:, seen_inds], labels[:, seen_inds])}

            if cfg.awsu > 0:
                unseen_class_labels = self.get_aff_ws_labels(interaction_labels)
                losses[f'{self.ld.tag}_loss_unseen'] = cfg.awsu * bce_loss(zs_logits[:, self.unseen_inds], unseen_class_labels)
        else:
            if not cfg.train_null_act:  # for part actions null is always trained, because a part needs not be relevant in an image.
                dir_logits = dir_logits[:, 1:]
                labels = labels[:, 1:]
            losses = {f'{self.ld.tag}_loss': bce_loss(dir_logits, labels)}
        return losses

    def get_aff_ws_labels(self, interaction_labels):
        assert cfg.awsu > 0
        batch_size = interaction_labels.shape[0]
        extended_inter_mat = interactions_to_mat(interaction_labels, hico=self.dataset.full_dataset)  # N x I -> N x O x A
        wemb_sim = self.wemb_sim.clamp(min=0)

        similar_acts_per_obj = torch.bmm(extended_inter_mat, wemb_sim.unsqueeze(dim=0).expand(batch_size, -1, -1))
        similar_acts_per_obj = similar_acts_per_obj / extended_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
        feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)

        unseen_class_labels = feasible_similar_acts_per_obj.max(dim=1)[0][:, self.unseen_inds]
        return unseen_class_labels.detach()


class TriheadActionBranch(GcnActionBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hh = self.dataset.full_dataset
        self._add_mlp(name='part_logits_to_act',
                      input_dim=self.dataset.num_part_actions - self.dataset.num_parts,
                      output_dim=self.repr_dim,
                      num_classes=self.ld.num_classes)
        self.fg_part_actions = nn.Parameter(torch.from_numpy(np.array([app[:-1] for app in hh.states_per_part])), requires_grad=False)
        self.bg_part_actions = nn.Parameter(torch.from_numpy(np.array([app[-1] for app in hh.states_per_part])), requires_grad=False)
        if cfg.awsu_fp > 0:
            part_actions = hh.split_part_annotations[self.dataset._data_split]
            parts = np.stack([part_actions[:, inds[:-1]].any(axis=1) for inds in hh.states_per_part], axis=1)

            inters = hh.split_annotations[self.dataset._data_split]  # FIXME oracle
            actions = np.minimum(1, inters @ self.ld.trasf_mat)
            part_actions_cooccs = parts.T @ actions
            part_actions_cooccs /= part_actions_cooccs.sum(axis=0)
            self.part_actions_coocs = nn.Parameter(torch.from_numpy(part_actions_cooccs).float(), requires_grad=False)

    def _get_losses(self, x, output, **kwargs):
        img_data, person_data, obj_data, interaction_labels, part_labels, other = x
        dir_logits, zs_logits, from_part_logits = output
        labels = (interaction_labels @ torch.from_numpy(self.ld.trasf_mat).to(interaction_labels)).clamp(min=0, max=1).detach()
        assert dir_logits.shape[1] == labels.shape[1]
        if self.zs_enabled:
            assert zs_logits.shape[1] == labels.shape[1]
            seen_inds = self.seen_inds
            if not cfg.train_null_act:
                seen_inds = seen_inds[1:]
            losses = {f'{self.ld.tag}_loss_seen': bce_loss(dir_logits[:, seen_inds], labels[:, seen_inds]) +
                                                  bce_loss(zs_logits[:, seen_inds], labels[:, seen_inds]) +
                                                  bce_loss(from_part_logits[:, seen_inds], labels[:, seen_inds])
                      }

            if cfg.awsu > 0:
                unseen_class_labels = self.get_aff_ws_labels(interaction_labels)
                losses[f'{self.ld.tag}_loss_unseen'] = cfg.awsu * bce_loss(zs_logits[:, self.unseen_inds], unseen_class_labels)

            if cfg.awsu_fp > 0:
                unseen_class_labels = self.get_part_req_ws_labels(labels, self.cache['part_dir_logits'])
                losses[f'{self.ld.tag}_loss_unseen_from_part'] = cfg.awsu_fp * bce_loss(from_part_logits[:, self.unseen_inds], unseen_class_labels)
        else:
            if not cfg.train_null_act:  # for part actions null is always trained, because a part needs not be relevant in an image.
                dir_logits = dir_logits[:, 1:]
                labels = labels[:, 1:]
            losses = {f'{self.ld.tag}_loss': bce_loss(dir_logits, labels)}
        return losses

    def _predict(self, x, output, **kwargs):
        dir_logits, zs_logits, from_part_logits = output
        logits = (dir_logits + from_part_logits) / 2
        if self.zs_enabled:
            logits[:, self.unseen_inds] = (zs_logits + from_part_logits)[:, self.unseen_inds] / 2
        return torch.sigmoid(logits).cpu().numpy()

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        dir_logits, zs_logits = super()._forward(img_data, person_data, obj_data, labels, part_labels, other)
        part_logits = self.cache['part_dir_logits']
        from_part_logits = self.repr_mlps['part_logits_to_act'](part_logits) @ self.linear_predictors['part_logits_to_act']
        return dir_logits, zs_logits, from_part_logits

    def get_part_req_ws_labels(self, action_labels, part_act_logits):
        assert cfg.awsu_fp > 0
        part_interactiveness_logits = -part_act_logits[:, self.bg_part_actions]
        act_weak_labels_from_part = (part_interactiveness_logits @ self.part_actions_coocs).clamp(min=0, max=1)
        unseen_act_labels = act_weak_labels_from_part[:, self.unseen_inds]
        return unseen_act_labels.detach()


class ActionFromPartWrapperBranch(GcnActionBranch):
    def __init__(self, base_branch=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = base_branch
        self._add_mlp(name='part_logits_to_act', input_dim=self.dataset.num_part_actions, output_dim=self.repr_dim,
                      num_classes=self.ld.num_classes)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        part_logits = self.cache['part_dir_logits']
        act_logits = self.repr_mlps['part_logits_to_act'](part_logits) @ self.linear_predictors['part_logits_to_act']
        dir_logits_list = [act_logits]
        zs_logits_list = [act_logits]
        if self.branch is not None:
            im_act_dir_logits, im_act_zs_logits = self.branch._forward(img_data, person_data, obj_data, labels, part_labels, other)
            im_act_dir_logits = im_act_dir_logits if isinstance(im_act_dir_logits, list) else [im_act_dir_logits]
            im_act_zs_logits = im_act_zs_logits if isinstance(im_act_zs_logits, list) else [im_act_zs_logits]
            dir_logits_list.extend(im_act_dir_logits)
            zs_logits_list.extend(im_act_zs_logits)
        return dir_logits_list, zs_logits_list


class PartInteractivenessActionWrapperBranch(GcnActionBranch):
    def __init__(self, base_branch=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = base_branch
        self.bg_part_actions = nn.Parameter(torch.from_numpy(np.array([app[-1] for app in self.dataset.full_dataset.states_per_part])),
                                            requires_grad=False)
        self._add_linear_layer(name='part_interactiveness', input_dim=self.dataset.num_parts, output_dim=self.ld.num_classes)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        part_logits = self.cache['part_dir_logits']
        fg_part_logits = -part_logits[:, self.bg_part_actions]
        act_logits = fg_part_logits @ self.linear_predictors['part_interactiveness']
        dir_logits_list = [act_logits]
        zs_logits_list = [act_logits]
        if self.branch is not None:
            im_act_dir_logits, im_act_zs_logits = self.branch._forward(img_data, person_data, obj_data, labels, part_labels, other)
            im_act_dir_logits = im_act_dir_logits if isinstance(im_act_dir_logits, list) else [im_act_dir_logits]
            im_act_zs_logits = im_act_zs_logits if isinstance(im_act_zs_logits, list) else [im_act_zs_logits]
            dir_logits_list.extend(im_act_dir_logits)
            zs_logits_list.extend(im_act_zs_logits)
        return dir_logits_list, zs_logits_list


class PartToHoiGcnActionBranch(GcnActionBranch):
    def __init__(self, part_repr_dim, use_ap_gcn=False, *args, **kwargs):
        self.use_ap_gcn = use_ap_gcn
        super().__init__(*args, **kwargs)
        for part_name in self.dataset.full_dataset.parts:
            self._add_mlp(name=f'img_part_to_act_{part_name}',
                          input_dim=part_repr_dim + self.repr_dim,
                          hidden_dim=2048,
                          output_dim=self.repr_dim)

    def _init_predictor_mlps(self):
        super()._init_predictor_mlps()
        gc_latent_dim = cfg.gcldim
        if self.use_ap_gcn:
            self._add_mlp(name='ap_gcn_predictor', input_dim=gc_latent_dim, hidden_dim=(gc_latent_dim + self.repr_dim) // 2,
                          output_dim=self.repr_dim, dropout_p=cfg.gcdropout)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels, other):
        im_ids, feats, orig_img_wh = img_data

        im_part_reprs = self.cache['im_part_reprs']
        repr = self.repr_mlps['repr'](feats)
        x = torch.cat([repr.unsqueeze(dim=1).expand(-1, im_part_reprs.shape[1], -1), im_part_reprs], dim=2)
        repr += torch.stack([self.repr_mlps[f'img_part_to_act_{p_n}'](x[:, p_i, :]) for p_i, p_n in enumerate(self.dataset.full_dataset.parts)],
                            dim=1).mean(dim=1)

        im_act_dir_logits = repr @ self.linear_predictors['dir']
        if self.zs_enabled:
            predictor = self.repr_mlps['oa_gcn_predictor'](self.cache[f'oa_gcn_{self.ld.tag}_class_embs'])
            if self.use_ap_gcn:
                predictor += self.repr_mlps['ap_gcn_predictor'](self.cache[f'ap_gcn_{self.ld.tag}_class_embs'])
            if cfg.seprep:
                repr = self.repr_mlps['zs_repr'](feats)
            im_act_zs_logits = repr @ predictor.t()
        else:
            im_act_zs_logits = None

        return im_act_dir_logits, im_act_zs_logits

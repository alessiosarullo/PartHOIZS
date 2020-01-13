from typing import List

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico_hake import HicoHakeKPSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel
from lib.models.gcns import HicoGCN
from lib.models.misc import bce_loss, LIS


class BaseModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        self.dataset = dataset
        self.repr_dim = cfg.repr_dim
        img_feats_dim = self.dataset.precomputed_visual_feat_dim
        kp_feats_dim = self.dataset.pc_kp_feats_dim
        self.zs_enabled = (cfg.seenf >= 0)

        # Initialise part actions attributes according to GT.
        hicohake = self.dataset.full_dataset
        bg_part_actions = np.array([app[-1] for app in hicohake.actions_per_part])
        fg_part_actions = np.concatenate([app[:-1] for app in hicohake.actions_per_part])
        self.bg_part_actions = nn.Parameter(torch.from_numpy(bg_part_actions), requires_grad=False)
        self.fg_part_actions = nn.Parameter(torch.from_numpy(fg_part_actions), requires_grad=False)
        # objs = np.minimum(1, np.maximum(0, self.dataset.labels) @ hicohake.interaction_to_object_mat)
        # part_object_cooccs = self.dataset.part_labels.T @ objs

        # Adjacency matrices
        interactions = self.dataset.full_dataset.interactions  # FIXME only "oracle" is supported ATM, even for zero-shot
        oa_adj = np.zeros([self.dataset.full_dataset.num_objects, self.dataset.full_dataset.num_actions], dtype=np.float32)
        oa_adj[interactions[:, 1], interactions[:, 0]] = 1
        oa_adj[:, 0] = 0
        oa_adj = torch.from_numpy(oa_adj)

        # Word embeddings + similarity matrices
        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')
        self.word_embs = nn.ParameterDict({'obj': nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False),
                                           'act': nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
                                           })
        self.obj_emb_sim = nn.Parameter(self.word_embs['obj'] @ self.word_embs['obj'].t(), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.word_embs['act'] @ self.word_embs['act'].t(), requires_grad=False)
        self.act_obj_emb_sim = nn.Parameter(self.word_embs['act'] @ self.word_embs['obj'].t(), requires_grad=False)

        ################## Branches
        self.repr_mlps = nn.ModuleDict()
        self.linear_predictors = nn.ParameterDict()

        # Part branches
        self.final_part_repr_dim = None
        for i, p_acts in enumerate(hicohake.actions_per_part):
            part_name = hicohake.parts[i]
            self._add_mlp_branch(branch_name=f'img_part_{part_name}', input_dim=img_feats_dim, output_dim=self.repr_dim)
            logit_input_dim = self.repr_dim
            logit_input_dim += hicohake.num_objects  # OD's scores

            if not cfg.no_kp:
                self._add_mlp_branch(branch_name=f'kp_part_{part_name}', input_dim=kp_feats_dim, output_dim=self.repr_dim)
                logit_input_dim += self.repr_dim

            if cfg.use_sk:
                self._add_mlp_branch(branch_name=f'ppl_skeleton_for_{part_name}', input_dim=34, hidden_dim=100, output_dim=200)
                logit_input_dim += 200

            if cfg.spcfmdim > 0:
                dim = (cfg.spcfmdim ** 2) + self.dataset.num_objects
                self._add_mlp_branch(branch_name=f'kp_box_spconf_{part_name}', input_dim=dim, hidden_dim=dim, output_dim=1)

            if self.final_part_repr_dim is None:
                self.final_part_repr_dim = logit_input_dim
            else:
                assert self.final_part_repr_dim == logit_input_dim
            self._add_linear_layer(branch_name=f'img_part_{part_name}', input_dim=logit_input_dim, output_dim=p_acts.size - 1)
            self._add_linear_layer(branch_name=f'img_part_{part_name}_null', input_dim=logit_input_dim, output_dim=1)

        # Object and action branches
        if not cfg.part_only:
            branches_output_dims = [('obj', dataset.full_dataset.num_objects),
                                    ('act', dataset.full_dataset.num_actions),
                                    ]
            # self.img_branches = sorted([k for k, d in branches_output_dims])
            gc_latent_dim = cfg.gcldim
            for k, n_classes in branches_output_dims:
                self._add_mlp_branch(branch_name=k, input_dim=img_feats_dim, output_dim=self.repr_dim, num_classes=n_classes)
                self._add_mlp_branch(branch_name=f'wemb_{k}', input_dim=word_embs.dim, hidden_dim=600, output_dim=self.repr_dim)
                self._add_mlp_branch(branch_name=f'predictor_{k}', input_dim=gc_latent_dim, hidden_dim=(gc_latent_dim + self.repr_dim) // 2,
                                     output_dim=self.repr_dim, dropout_p=cfg.gcdropout)

            if self.zs_enabled:
                # GCN
                print('Zero-shot enabled.')
                gc_emb_dim = cfg.gcrdim
                gc_dims = ((gc_emb_dim + gc_latent_dim) // 2, gc_latent_dim)
                self.gcn = HicoGCN(dataset, oa_adj=oa_adj, input_dim=gc_emb_dim, gc_dims=gc_dims)

                self.seen_inds = {}
                self.unseen_inds = {}

                seen_obj_inds = dataset.active_objects
                unseen_obj_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_objects)) - set(seen_obj_inds.tolist())))
                self.seen_inds['obj'] = nn.Parameter(torch.tensor(seen_obj_inds), requires_grad=False)
                self.unseen_inds['obj'] = nn.Parameter(torch.tensor(unseen_obj_inds), requires_grad=False)

                seen_act_inds = dataset.active_actions
                unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
                self.seen_inds['act'] = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
                self.unseen_inds['act'] = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

                # if self.soft_labels_enabled:
                #     self.obj_act_feasibility = nn.Parameter(oa_adj, requires_grad=False)

    def _add_mlp_branch(self, branch_name, input_dim, output_dim, num_classes=None, hidden_dim=1024, dropout_p=cfg.dropout):
        k = branch_name
        assert k not in self.repr_mlps
        self.repr_mlps[k] = nn.Sequential(*([nn.Linear(input_dim, hidden_dim),
                                             nn.ReLU(inplace=True)] +
                                            ([nn.Dropout(p=dropout_p)] if dropout_p > 0 else []) +
                                            [nn.Linear(hidden_dim, output_dim)
                                             ]
                                            ))
        nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.repr_mlps[k][3 if dropout_p > 0 else 2].weight, gain=torch.nn.init.calculate_gain('linear'))
        if num_classes is not None:
            self._add_linear_layer(branch_name=branch_name, input_dim=output_dim, output_dim=num_classes)

    def _add_linear_layer(self, branch_name, input_dim, output_dim):
        k = branch_name
        assert k not in self.linear_predictors
        self.linear_predictors[k] = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.linear_predictors[k], gain=1.0)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            # Prepare the data
            img_data, person_data, obj_data, orig_labels, orig_part_labels, other = x
            feats, orig_img_wh = img_data
            if not inference:
                epoch, iter = other

            # Compute logits
            part_logits, obj_dir_logits, act_dir_logits, obj_zs_logits, act_zs_logits = \
                self._forward(img_data, person_data, obj_data, orig_labels, orig_part_labels)
            if part_logits is None:
                assert inference

            if not inference:
                losses = {'fg_part_loss': bce_loss(part_logits[:, self.fg_part_actions], orig_part_labels[:, self.fg_part_actions],
                                                   pos_weights=cfg.cspc if cfg.cspc > 0 else None),
                          'bg_part_loss': bce_loss(part_logits[:, self.bg_part_actions], orig_part_labels[:, self.bg_part_actions],
                                                   pos_weights=cfg.cspbgc if cfg.cspbgc > 0 else None)
                          }

                if not cfg.part_only:
                    obj_labels = (orig_labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(orig_labels)
                                  ).clamp(min=0, max=1).detach()
                    act_labels = (orig_labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(orig_labels)
                                  ).clamp(min=0, max=1).detach()

                    if self.zs_enabled:
                        seen_obj, unseen_obj = self.seen_inds['obj'], self.unseen_inds['obj']
                        seen_act, unseen_act = self.seen_inds['act'], self.unseen_inds['act']
                        if not cfg.train_null_act:
                            seen_act = seen_act[1:]

                        losses['obj_loss_seen'] = bce_loss(obj_dir_logits[:, seen_obj], obj_labels[:, seen_obj]) + \
                                                  bce_loss(obj_zs_logits[:, seen_obj], obj_labels[:, seen_obj])
                        losses['act_loss_seen'] = bce_loss(act_dir_logits[:, seen_act], act_labels[:, seen_act]) + \
                                                  bce_loss(act_zs_logits[:, seen_act], act_labels[:, seen_act])
                    else:
                        if not cfg.train_null_act:  # for part actions null is always trained, because a part needs not be relevant in an image.
                            act_dir_logits = act_dir_logits[:, 1:]
                            act_labels = act_labels[:, 1:]

                        losses['obj_loss'] = bce_loss(obj_dir_logits, obj_labels)
                        losses['act_loss'] = bce_loss(act_dir_logits, act_labels)
                return losses
            else:
                prediction = Prediction()

                # Part action logits
                if part_logits is None:
                    prediction.part_action_scores = np.zeros((feats.shape[0], self.dataset.full_dataset.num_part_actions))
                else:
                    prediction.part_action_scores = torch.sigmoid(part_logits).cpu().numpy()
                assert prediction.part_action_scores.shape[1] == self.dataset.full_dataset.num_part_actions

                if not cfg.part_only:
                    # HOI logits
                    # if self.zs_enabled and len(all_knowl_logits) > 0:
                    #     for k in all_logits.keys():
                    #         unseen = self.unseen_inds[k]
                    #         all_logits[k][:, unseen] = all_knowl_logits[k][:, unseen]
                    interactions = self.dataset.full_dataset.interactions
                    obj_scores = torch.sigmoid(obj_dir_logits).cpu().numpy()
                    act_scores = torch.sigmoid(act_dir_logits).cpu().numpy()
                    prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]]
                return prediction

    def _part_branch(self, feats, person_data, obj_data):
        # N = #images, P = #parts, F = image features dim, D = part representation dim

        im_inds, person_boxes, person_coco_kps, kp_boxes, kp_feats = person_data
        obj_im_inds, obj_boxes, obj_scores, obj_feats, kp_box_prox_to_obj_fmaps, obj_scores_per_kp_box = obj_data
        hh = self.dataset.full_dataset
        dim = self.final_part_repr_dim

        im_part_reprs = []
        for p in hh.parts:
            im_part_reprs.append(self.repr_mlps[f'img_part_{p}'](feats))
        im_part_reprs = torch.stack(im_part_reprs, dim=1)

        if not cfg.no_kp:
            if im_inds:  # All of these are computed for each person and then aggregated to obtain one per image.
                aggr_ppl_part_reprs = []

                # Part representation from keypoint bounding boxes visual features.
                part_kp_reprs_per_person = []
                for p, kps in enumerate(hh.part_to_kp):
                    # Lower representation's contribution based on score. Soft threshold is at around 0.1 (LIS with w=96 and k=10).
                    weights = LIS(kp_boxes[:, kps, -1:], w=96, k=10)
                    kp_repr = (self.repr_mlps[f'kp_part_{hh.parts[p]}'](kp_feats[:, kps, :]) * weights).mean(dim=1)
                    part_kp_reprs_per_person.append(kp_repr)
                part_kp_reprs_per_person = torch.stack(part_kp_reprs_per_person, dim=1)  # N x P x F

                if cfg.spcfmdim > 0:
                    kp_box_prox_to_obj_feats = kp_box_prox_to_obj_fmaps.view(kp_box_prox_to_obj_fmaps.shape[0], kp_box_prox_to_obj_fmaps.shape[1], -1)
                    kp_box_feats = torch.cat([kp_box_prox_to_obj_feats, obj_scores_per_kp_box], dim=2)

                    kp_weights_per_person = []
                    for p, kps in enumerate(hh.part_to_kp):
                        kp_repr = self.repr_mlps[f'kp_box_spconf_{hh.parts[p]}'](kp_box_feats[:, kps, :].mean(dim=1))
                        kp_weights_per_person.append(kp_repr)
                    kp_weights_per_person = torch.stack(kp_weights_per_person, dim=1)  # N x P x 1

                    # ppl_att_scores = torch.sigmoid(kp_weights_per_person)
                    ppl_att_scores = kp_weights_per_person
                    ppl_att_scores = torch.cat([nn.functional.softmax(ppl_att_scores[iids], dim=0) for iids in im_inds], dim=0)
                else:
                    ppl_att_scores = torch.cat([im_part_reprs.new_full((len(inds), self.dataset.num_parts, 1), fill_value=1 / len(inds)) for inds in
                                                im_inds],
                                               dim=0)

                part_kp_reprs_per_person *= ppl_att_scores
                part_kp_reprs = torch.stack([part_kp_reprs_per_person[iids].sum(dim=0) for iids in im_inds], dim=0)
                aggr_ppl_part_reprs.append(part_kp_reprs)

                # Object detection's scores
                im_obj_scores = torch.stack([obj_scores[iids].max(dim=0)[0] if iids.size > 0 else im_part_reprs.new_zeros(self.dataset.num_objects)
                                             for iids in obj_im_inds], dim=0)
                aggr_ppl_part_reprs.append(im_obj_scores.unsqueeze(dim=1).expand(-1, im_part_reprs.shape[1], -1))

                if cfg.use_sk:
                    # Part representation from skeleton (simple concatenation of normalised keypoints).
                    person_boxes_ext = person_boxes.view(person_boxes.shape[0], 1, -1)
                    norm_person_coco_kps = (person_coco_kps[:, :, :2] - person_boxes_ext[:, :, :2]) / \
                                           (person_boxes_ext[:, :, 2:4] - person_boxes_ext[:, :, :2])
                    assert (0 <= norm_person_coco_kps).all() and (norm_person_coco_kps <= 1).all()
                    person_skeleton = norm_person_coco_kps.view(norm_person_coco_kps.shape[0], -1)
                    skeleton_repr_per_person = torch.stack([self.repr_mlps[f'ppl_skeleton_for_{p}'](person_skeleton) for p in hh.parts], dim=1)
                    skeleton_reprs = torch.stack([skeleton_repr_per_person[iids].mean(dim=0) for iids in im_inds], dim=0)
                    aggr_ppl_part_reprs.append(skeleton_reprs)

                aggr_ppl_part_reprs = torch.cat(aggr_ppl_part_reprs, dim=2)
            else:
                aggr_ppl_part_reprs = im_part_reprs.new_zeros((im_part_reprs.shape[0], im_part_reprs.shape[1], dim - im_part_reprs.shape[2]))
            im_part_reprs = torch.cat([im_part_reprs, aggr_ppl_part_reprs], dim=2)
        assert im_part_reprs.shape[0] == feats.shape[0] and im_part_reprs.shape[1] == hh.num_parts and im_part_reprs.shape[2] == dim  # N x P x D

        im_part_logits_list = []
        for part_idx, p in enumerate(hh.parts):
            repr = im_part_reprs[:, part_idx, :]
            p_logits = repr @ self.linear_predictors[f'img_part_{p}']
            p_interactiveness_logits = repr @ self.linear_predictors[f'img_part_{p}_null']
            im_part_logits_list.append(torch.cat([p_logits, p_interactiveness_logits], dim=1))
        im_part_logits = torch.cat(im_part_logits_list, dim=1)

        # # Make NULL exclusive with other interactions (in a soft way).
        # for p, acts in enumerate(hh.actions_per_part):
        #     im_part_logits[:, acts[:-1]] -= im_part_logits[:, [acts[-1]]]

        return im_part_logits, im_part_reprs

    def _forward(self, img_data, person_data, obj_data, labels, part_labels):
        feats, orig_img_wh = img_data
        im_part_logits, im_part_reprs = self._part_branch(feats, person_data, obj_data)

        # Object and action branches
        im_zs_obj_logits = im_zs_act_logits = None
        if not cfg.part_only:
            obj_repr = self.repr_mlps['obj'](feats)
            act_repr = self.repr_mlps['act'](feats)
            im_dir_obj_logits = obj_repr @ self.linear_predictors['obj']
            im_dir_act_logits = act_repr @ self.linear_predictors['act']
            if self.zs_enabled:
                obj_class_embs, act_class_embs = self.gcn()  # P x E

                obj_predictor = self.repr_mlps['predictor_obj'](obj_class_embs) + self.repr_mlps['wemb_obj'](self.word_embs['obj'])
                im_zs_obj_logits = obj_repr @ obj_predictor.t()

                act_predictor = self.repr_mlps['predictor_act'](act_class_embs)
                im_zs_act_logits = act_repr @ act_predictor.t()
        else:
            im_dir_obj_logits = im_dir_act_logits = None

        return im_part_logits, im_dir_obj_logits, im_dir_act_logits, im_zs_obj_logits, im_zs_act_logits


class Part2HoiModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'part2hoi'

    def __init__(self, dataset: HicoHakeKPSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        hicohake = self.dataset.full_dataset
        for part_name in hicohake.parts:
            self._add_mlp_branch(branch_name=f'img_part_to_act_{part_name}',
                                 input_dim=self.final_part_repr_dim + self.repr_dim,
                                 hidden_dim=2048,
                                 output_dim=self.repr_dim)

    def _forward(self, img_data, person_data, obj_data, labels, part_labels):
        assert not cfg.part_only

        feats, orig_img_wh = img_data
        im_part_logits, im_part_reprs = self._part_branch(feats, person_data, obj_data)

        # Object and action branches
        obj_repr = self.repr_mlps['obj'](feats)
        im_dir_obj_logits = obj_repr @ self.linear_predictors['obj']

        act_repr = self.repr_mlps['act'](feats)
        x = torch.cat([act_repr.unsqueeze(dim=1).expand(-1, im_part_reprs.shape[1], -1), im_part_reprs], dim=2)
        act_repr += torch.stack([self.repr_mlps[f'img_part_to_act_{p_n}'](x[:, p_i, :]) for p_i, p_n in enumerate(self.dataset.full_dataset.parts)],
                                dim=1).mean(dim=1)
        im_dir_act_logits = act_repr @ self.linear_predictors['act']

        if self.zs_enabled:
            obj_class_embs, act_class_embs = self.gcn()  # P x E

            obj_predictor = self.repr_mlps['predictor_obj'](obj_class_embs) + self.repr_mlps['wemb_obj'](self.word_embs['obj'])
            im_zs_obj_logits = obj_repr @ obj_predictor.t()

            act_predictor = self.repr_mlps['predictor_act'](act_class_embs)
            im_zs_act_logits = act_repr @ act_predictor.t()
        else:
            im_zs_obj_logits = im_zs_act_logits = None

        return im_part_logits, im_dir_obj_logits, im_dir_act_logits, im_zs_obj_logits, im_zs_act_logits

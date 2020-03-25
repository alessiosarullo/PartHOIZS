import json
import os
from typing import Dict, List, NamedTuple, Union

import h5py
import numpy as np
import torch
import pickle

from config import cfg
from lib.dataset.hico import Hico, HicoSplit
from lib.dataset.utils import Splits, get_hico_to_coco_mapping, COCO_CLASSES, filter_on_score
from lib.timer import Timer
from lib.dataset.tin_utils import get_next_sp_with_pose


class Minibatch(NamedTuple):
    ex_data: List
    im_data: List
    person_data: List
    obj_data: List
    ex_labels: torch.Tensor
    pstate_labels: torch.Tensor
    other: List


class Dims(NamedTuple):
    N: Union[int, None]  # number of images in the batch (it is defined later)
    P: int  # number of people
    M: int  # number of objects
    K_coco: int  # number of keypoints returned by the keypoint detector
    K_hake: int  # number of keypoints in HAKE
    B: int  # number of body part classes
    S: int  # number of body part states
    O: int  # number of object classes
    A: int  # number of action classes
    C: int  # number of interaction classes
    F_img: int  # CNN feature vector dimensionality
    F_kp: int  # keypoint feature vector dimensionality
    F_obj: int  # object feature vector dimensionality
    D: int  # dimensionality of interaction pattern


# TODO remove
class PrecomputedFilesHandler:
    files = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def get_file(cls, file_name):
        return cls.files.setdefault(file_name, {}).setdefault('_handler', h5py.File(file_name, 'r'))

    @classmethod
    def get(cls, file_name, attribute_name, load_in_memory=None):
        if load_in_memory is None:
            load_in_memory = not cfg.no_load_memory
        file = cls.get_file(file_name)
        key = attribute_name
        if load_in_memory:
            key += '__loaded'
        if key not in cls.files[file_name].keys():
            attribute = file[attribute_name]
            if load_in_memory:
                attribute = attribute[:]
            cls.files[file_name][key] = attribute
        return cls.files[file_name][key]


class HicoHake(Hico):
    def __init__(self):
        """
        In the following, BP = body parts and PS = (body) part state.
        Attributes:
            - parts: List[str]. The 6 parts in HAKE's annotations.
            - part_inv_ind: Dict[str, int]. Parts inverse index (string -> index in `parts`).
            - bp_ps_pairs: List[List[str]]. This is essentially the content of file `Part_State_76.txt`.
            - states_per_part: List[np.array]. Associates an array of indices in `part_action_pairs` to each part (same order as `parts`).
            - keypoints: List[str]. Names of HAKE's 10 keypoints.
            - part_to_kp: List[np.array]. Associates to each part an array of keypoints (e.g., at the index corresponding to 'foot' there will be
                            indices for ['right foot', 'left foot']).
        """
        super().__init__()

        with open(os.path.join(cfg.data_root, 'HICO', 'HAKE', 'Part_State_76.txt'), 'r') as f:
            bp_ps_dict_str = {}
            bp_ps_pairs = []
            for i, l in enumerate(f.readlines()):
                if not l.strip():
                    break
                bp_ps_pair = [x.strip() for x in l.strip().split(':')]
                body_part, part_state = bp_ps_pair
                if part_state == 'no_interaction':
                    part_state = self.null_action
                bp_ps_dict_str.setdefault(body_part, []).append(len(bp_ps_pairs))
                bp_ps_pairs.append([body_part, part_state])
        self.parts = list(bp_ps_dict_str.keys())  # type: List[str]
        self.part_states = [f'{p} {a}' for p, a in bp_ps_pairs]  # type: List[str]
        self.part_inv_ind = {part: i for i, part in enumerate(self.parts)}  # type: Dict[str, int]
        self.bp_ps_pairs = bp_ps_pairs  # type: List[List[str]]
        self.states_per_part = [np.array(sorted(bp_ps_dict_str[p])) for p in self.parts]  # type: List[np.array]

        with open(os.path.join(cfg.data_root, 'HICO', 'HAKE', 'joints.txt'), 'r') as f:
            self.keypoints = [l.strip().lower().replace('_', ' ') for l in f.readlines()]  # type: List[str]
        self.part_to_kp = [np.array([j for j, kp in enumerate(self.keypoints) if p in kp]) for p in self.parts]  # type: List[np.array]
        assert np.all(np.sort(np.concatenate(self.part_to_kp)) == np.arange(len(self.keypoints)))
        kp_to_part = np.array([[1 if p in kp else 0 for i, p in enumerate(self.parts)] for kp in self.keypoints])
        assert kp_to_part.shape == (len(self.keypoints), len(self.parts)) and np.all(kp_to_part.sum(axis=1) == 1)
        self.kp_to_part = np.where(kp_to_part)[1].tolist()

        # Part annotations
        self.split_part_annotations = {}
        for split, fns in self.split_filenames.items():
            part_anns = json.load(open(os.path.join(cfg.data_root, 'HICO', 'HAKE', f'{split.value}.json'), 'r'))  # type: Dict
            part_ann_list = [part_anns[k] for k in fns]  # There are two extra files in this split
            part_ann_vecs = np.stack([np.concatenate([pa[f'{p}_list'] for p in self.parts]) for pa in part_ann_list], axis=0)
            self.split_part_annotations[split] = part_ann_vecs

    @property
    def num_parts(self):
        return len(self.parts)

    @property
    def num_keypoints(self):
        return len(self.keypoints)

    @property
    def num_part_states(self):
        return len(self.bp_ps_pairs)


class HicoHakeSplit(HicoSplit):
    def __init__(self, *args, **kwargs):
        super(HicoHakeSplit, self).__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: HicoHake
        self.img_pstate_labels = self.full_dataset.split_part_annotations[self.split]

    @classmethod
    def instantiate_full_dataset(cls) -> HicoHake:
        return HicoHake()

    def _collate(self, idx_list, device):
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
        if self.split != Splits.TEST:
            labels = torch.tensor(self.img_labels[idxs, :], dtype=torch.float32, device=device)
            part_labels = torch.tensor(self.img_pstate_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            labels = part_labels = None
        Timer.get('GetBatch').toc()
        return feats, labels, part_labels, []


class HicoHakeKPSplit(HicoHakeSplit):
    def __init__(self, no_feats=False, *args, **kwargs):
        super(HicoHakeKPSplit, self).__init__(*args, **kwargs)
        self.cache_tin = (cfg.ipsize <= 32) and cfg.cache_tin
        hico_to_coco_mapping = get_hico_to_coco_mapping(self.full_dataset.objects)

        #############################################################################################################################
        # Load precomputed data
        #############################################################################################################################
        #####################################################
        # Objects
        #####################################################
        self.objects_fn = cfg.precomputed_feats_format % ('hicoobjs', 'mask_rcnn_X_101_32x8d_FPN_3x',
                                                          f'{self.split.value}{"__nofeats" if no_feats else ""}')
        self.obj_boxes = PrecomputedFilesHandler.get(self.objects_fn, 'boxes', load_in_memory=True)
        self.obj_scores = PrecomputedFilesHandler.get(self.objects_fn, 'box_scores', load_in_memory=True)[:, hico_to_coco_mapping]
        if not no_feats:
            self.obj_feats = PrecomputedFilesHandler.get(self.objects_fn, 'box_feats')
            self.obj_feats_dim = self.obj_feats.shape[-1]
            assert self.obj_feats.dtype == np.float32
        assert self.obj_boxes.dtype == self.obj_scores.dtype == np.float32

        #####################################################
        # People
        #####################################################
        self.keypoints_fn = cfg.precomputed_feats_format % \
                            ('hicokps', 'keypoint_rcnn_R_101_FPN_3x', f'{self.split.value}{"__nofeats" if no_feats else ""}')
        self.person_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'boxes', load_in_memory=True)
        self.coco_kps = PrecomputedFilesHandler.get(self.keypoints_fn, 'keypoints', load_in_memory=True)
        self.person_scores = PrecomputedFilesHandler.get(self.keypoints_fn, 'scores', load_in_memory=True)
        self.hake_kp_boxes = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_boxes', load_in_memory=True)
        assert self.hake_kp_boxes.shape[1] == self.full_dataset.num_keypoints
        if not no_feats:
            self.hake_kp_feats = PrecomputedFilesHandler.get(self.keypoints_fn, 'kp_feats')
            self.person_feats = PrecomputedFilesHandler.get(self.keypoints_fn, 'person_feats')
            self.kp_net_dim = self.hake_kp_feats.shape[-1]
            assert self.kp_net_dim == self.person_feats.shape[-1]
            assert self.hake_kp_feats.dtype == np.float32
        assert np.all(np.min(self.coco_kps[:, :, :2], axis=1) >= self.person_boxes[:, :2])
        assert np.all(np.max(self.coco_kps[:, :, :2], axis=1) <= self.person_boxes[:, 2:4])
        assert self.person_boxes.dtype == self.coco_kps.dtype == self.person_scores.dtype == self.hake_kp_boxes.dtype == np.float32

        #############################################################################################################################
        # Cache variables to speed up loading
        #############################################################################################################################
        self.img_data_cache, imgs_with_kps = self._compute_img_data()  # type: List[Dict], List
        assert len(self.img_data_cache) == len(self.fnames)
        if cfg.tin and self.cache_tin:
            interactiveness_fname_ids, self.interactiveness_patterns = self._get_interactiveness_patterns()
            inv_ind = {fname_id: i for i, fname_id in enumerate(interactiveness_fname_ids)}
            self.interactiveness_idx_from_img_idx = [inv_ind.get(self.full_dataset.get_fname_id(self.fnames[idx]), None)
                                                     for idx in range(self.num_images)]

        # FIXME? Keep images without detected keypoints (i.e., people)
        if cfg.keep_unlabelled:
            self.non_empty_inds = np.array(imgs_with_kps)
        else:
            self.non_empty_inds = np.intersect1d(self.non_empty_inds, np.array(imgs_with_kps))

    def _compute_img_data(self):
        """
        Fields: 'fname_id': int, 'person_inds': array, 'obj_inds': array
        """

        # Filter out BG objects and persons (already have the BBs provided by the keypoint model).
        obj_classes = np.argmax(PrecomputedFilesHandler.get(self.objects_fn, 'box_scores'), axis=1)
        obj_to_keep = (obj_classes != COCO_CLASSES.index('__background__')) & (obj_classes != COCO_CLASSES.index('person'))
        fname_id_to_obj_inds = {}
        for i, fid in enumerate(PrecomputedFilesHandler.get(self.objects_fn, 'fname_ids')):
            if obj_to_keep[i]:
                fname_id_to_obj_inds.setdefault(fid, []).append(i)
        fname_id_to_obj_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_obj_inds.items()}

        fname_id_to_kp_inds = {}
        for i, fid in enumerate(PrecomputedFilesHandler.get(self.keypoints_fn, 'fname_ids')):
            fname_id_to_kp_inds.setdefault(fid, []).append(i)
        fname_id_to_kp_inds = {k: np.array(sorted(v)) for k, v in fname_id_to_kp_inds.items()}

        all_img_data = []
        imgs_with_kps = []
        for i, fname in enumerate(self.fnames):
            fname_id = self.full_dataset.get_fname_id(fname)
            im_data = {'fname_id': fname_id}

            # People
            im_person_inds = fname_id_to_kp_inds.get(fname_id, None)
            if im_person_inds is not None:
                im_person_inds = im_person_inds[np.any(self.hake_kp_boxes[im_person_inds, :], axis=(1, 2))]
                if im_person_inds.size > 0 and cfg.sbmar > 0:  # filter too small persons (with respect to the largest one)
                    pboxes = self.person_boxes[im_person_inds, :]
                    pboxes_areas = np.prod(pboxes[:, 2:4] - pboxes[:, :2], axis=1)
                    pboxes_area_ratios = pboxes_areas / np.amax(pboxes_areas)
                    im_person_inds = im_person_inds[pboxes_area_ratios >= cfg.sbmar]
                if im_person_inds.size > 0:
                    im_person_inds = np.atleast_1d(im_person_inds)
                    inds = filter_on_score(scores=self.person_scores[im_person_inds], min_score=cfg.min_ppl_score, maxn=cfg.max_ppl, keep_one=True)
                    assert inds.size > 0
                    im_data['person_inds'] = np.sort(im_person_inds[inds])
                    imgs_with_kps.append(i)

            # Objects
            obj_inds = fname_id_to_obj_inds.get(fname_id, None)
            if obj_inds is not None:
                im_obj_inds = np.atleast_1d(obj_inds)
                scores = np.max(self.obj_scores[im_obj_inds, :], axis=1)
                inds = filter_on_score(scores=scores, min_score=None, maxn=cfg.max_obj, keep_one=False)
                assert inds.size > 0
                im_data['obj_inds'] = np.sort(im_obj_inds[inds])

            all_img_data.append(im_data)
        return all_img_data, imgs_with_kps

    def _get_interactiveness_patterns(self):
        dims = self.dims
        P, M, K_hake, D = dims.P, dims.M, dims.K_hake, dims.D
        fname_cache_fname = os.path.join(cfg.cache_root, f'interactiveness_fnames_dim{D}_ppl{P}_obj{M}__{self.split.value}.pkl')
        data_cache_fname = os.path.join(cfg.cache_root, f'interactiveness_dim{D}_ppl{P}_obj{M}__{self.split.value}.h5')
        if not os.path.exists(fname_cache_fname):
            print('Caching interactiveness patterns.')
            img_data = [imd for imd in self.img_data_cache if 'person_inds' in imd.keys() and 'obj_inds' in imd.keys()]
            interactiveness_patterns = np.full((len(img_data), P, M, D, D, 3 + K_hake), fill_value=np.nan, dtype=np.float32)
            for i, imd in enumerate(img_data):
                person_inds = imd['person_inds']
                obj_inds = imd['obj_inds']
                patterns = get_next_sp_with_pose(human_boxes=self.person_boxes[person_inds],
                                                 human_poses=self.coco_kps[person_inds],
                                                 object_boxes=self.obj_boxes[obj_inds],
                                                 size=D,
                                                 part_boxes=self.hake_kp_boxes[person_inds, :, :4]
                                                 )
                interactiveness_patterns[i, :person_inds.size, :obj_inds.size] = patterns

            ip_file = h5py.File(data_cache_fname, 'w')
            ip_file.create_dataset('interactiveness_patterns', data=interactiveness_patterns)
            ip_file.close()
            del interactiveness_patterns

            interactiveness_fname_ids = np.array([imd['fname_id'] for imd in img_data])
            with open(fname_cache_fname, 'wb') as f:
                pickle.dump(interactiveness_fname_ids, f)

        print('Loading interactiveness patterns.')
        with open(fname_cache_fname, 'rb') as f:
            interactiveness_fname_ids = pickle.load(f)
        interactiveness_patterns = PrecomputedFilesHandler.get(data_cache_fname, 'interactiveness_patterns', load_in_memory=False)

        return interactiveness_fname_ids, interactiveness_patterns

    @property
    def dims(self) -> Dims:
        P, M = cfg.max_ppl, cfg.max_obj
        K_coco, K_hake = self.coco_kps.shape[1], self.full_dataset.num_keypoints
        B, S = self.full_dataset.num_parts, self.full_dataset.num_part_states
        O, A, C = self.full_dataset.num_objects, self.full_dataset.num_actions, self.full_dataset.num_interactions
        F_img, F_kp, F_obj = self.img_feat_dim, self.kp_net_dim, self.obj_feats_dim
        if cfg.tin:
            D = cfg.ipsize
        else:
            D = None
        return Dims(N=None, P=P, M=M, K_coco=K_coco, K_hake=K_hake, B=B, S=S, O=O, A=A, C=C, F_img=F_img, F_kp=F_kp, F_obj=F_obj, D=D)

    def _collate(self, idx_list, device):
        # This assumes the filtering has been done on initialisation (i.e., there are no more people/objects than the max number)
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)

        # Example data
        ex_ids = [f'ex{idx}' for idx in idxs]
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)

        # Image data
        im_ids = [f'{self.split.value}_{idx}' for idx in idxs]
        im_feats = feats
        orig_img_wh = torch.tensor(self.img_dims[idxs, :], dtype=torch.float32, device=device)

        dims = self.dims
        P, M, K_coco, K_hake, B, O, F_kp, F_obj, D = dims.P, dims.M, dims.K_coco, dims.K_hake, dims.B, dims.O, dims.F_kp, dims.F_obj, dims.D
        N = idxs.size  # #imgs

        ppl_boxes = np.zeros((N, P, 4), dtype=np.float32)
        ppl_feats = np.zeros((N, P, F_kp), dtype=np.float32)
        coco_kps = np.zeros((N, P, K_coco, 3), dtype=np.float32)
        kp_boxes = np.zeros((N, P, K_hake, 5), dtype=np.float32)
        kp_feats = np.zeros((N, P, K_hake, F_kp), dtype=np.float32)
        obj_boxes = np.zeros((N, M, 4), dtype=np.float32)
        obj_scores = np.zeros((N, M, O), dtype=np.float32)
        obj_feats = np.zeros((N, M, F_obj), dtype=np.float32)

        person_inds = np.full((N, P), fill_value=np.nan)
        obj_inds = np.full((N, M), fill_value=np.nan)
        for i, idx in enumerate(idxs):
            im_data = self.img_data_cache[idx]
            try:
                i_person_inds = im_data['person_inds']
                P_i = i_person_inds.size
                assert P_i <= P
                person_inds[i, :P_i] = i_person_inds
            except KeyError:
                pass

            try:
                i_obj_inds = im_data['obj_inds']
                M_i = i_obj_inds.size
                assert M_i <= M
                obj_inds[i, :M_i] = i_obj_inds
            except KeyError:
                pass

        valid_person_inds_mask = ~np.isnan(person_inds)
        valid_person_mask = np.zeros(self.person_boxes.shape[0], dtype=bool)  # using mask instead of inds in case H5 file are not loaded in memory.
        valid_person_mask[person_inds[valid_person_inds_mask].astype(np.int)] = True
        ppl_boxes[valid_person_inds_mask] = self.person_boxes[valid_person_mask]
        ppl_feats[valid_person_inds_mask] = self.person_feats[valid_person_mask, ...]
        coco_kps[valid_person_inds_mask] = self.coco_kps[valid_person_mask]
        kp_boxes[valid_person_inds_mask] = self.hake_kp_boxes[valid_person_mask]
        kp_feats[valid_person_inds_mask] = self.hake_kp_feats[valid_person_mask, ...]

        valid_obj_inds_mask = ~np.isnan(obj_inds)
        valid_obj_mask = np.zeros(self.obj_boxes.shape[0], dtype=bool)
        valid_obj_mask[obj_inds[valid_obj_inds_mask].astype(np.int)] = True
        obj_boxes[valid_obj_inds_mask] = self.obj_boxes[valid_obj_mask]
        obj_scores[valid_obj_inds_mask] = self.obj_scores[valid_obj_mask]
        obj_feats[valid_obj_inds_mask] = self.obj_feats[valid_obj_mask, ...]

        t_ppl_boxes = torch.from_numpy(ppl_boxes).to(device=device)
        t_ppl_feats = torch.from_numpy(ppl_feats).to(device=device)
        t_coco_kps = torch.from_numpy(coco_kps).to(device=device)
        t_kp_boxes = torch.from_numpy(kp_boxes).to(device=device)
        t_kp_feats = torch.from_numpy(kp_feats).to(device=device)
        t_obj_boxes = torch.from_numpy(obj_boxes).to(device=device)
        t_obj_scores = torch.from_numpy(obj_scores).to(device=device)
        t_obj_feats = torch.from_numpy(obj_feats).to(device=device)

        if self.split != Splits.TEST:
            img_labels = torch.tensor(self.img_labels[idxs, :], dtype=torch.float32, device=device)
            pstate_labels = torch.tensor(self.img_pstate_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            img_labels = pstate_labels = None

        mb = Minibatch(ex_data=[ex_ids, feats],
                       im_data=[im_ids, im_feats, orig_img_wh],
                       person_data=[t_ppl_boxes, t_ppl_feats, t_coco_kps, t_kp_boxes, t_kp_feats],
                       obj_data=[t_obj_boxes, t_obj_scores, t_obj_feats],
                       ex_labels=img_labels,
                       pstate_labels=pstate_labels,
                       other=[])
        if cfg.tin:
            if self.cache_tin:
                ip_inds = [self.interactiveness_idx_from_img_idx[idx] for idx in idx_list]
                ip_inds = np.array([idx if idx is not None else np.nan for idx in ip_inds])
                valid_ip_inds_mask = ~np.isnan(ip_inds)
                valid_ip_inds = ip_inds[valid_ip_inds_mask].astype(np.int)
                sort_inds = np.argsort(valid_ip_inds)
                sorted_valid_ip_inds = valid_ip_inds[sort_inds]
                unsort_inds = np.argsort(sort_inds)
                assert np.all(valid_ip_inds == sorted_valid_ip_inds[unsort_inds])
                interactiveness_patterns = np.full((N, *self.interactiveness_patterns.shape[1:]), fill_value=np.nan, dtype=np.float32)
                interactiveness_patterns[valid_ip_inds_mask] = self.interactiveness_patterns[sorted_valid_ip_inds][unsort_inds]
                interactiveness_patterns[np.isnan(interactiveness_patterns)] = 0
            else:
                interactiveness_patterns = np.zeros((N, P, M, D, D, 3 + K_hake), dtype=np.float32)
                for i in range(N):
                    valid_person_inds_mask_i = valid_person_inds_mask[i]
                    valid_obj_inds_mask_i = valid_obj_inds_mask[i]
                    pattern = get_next_sp_with_pose(human_boxes=ppl_boxes[i, valid_person_inds_mask_i],
                                                    human_poses=coco_kps[i, valid_person_inds_mask_i],
                                                    object_boxes=obj_boxes[i, valid_obj_inds_mask_i],
                                                    size=D,
                                                    part_boxes=kp_boxes[i, valid_person_inds_mask_i, :, :4]
                                                    )
                    r, c = np.where(valid_person_inds_mask_i[:, None] & valid_obj_inds_mask_i[None, :])
                    interactiveness_patterns[i, r, c] = pattern.reshape((-1, *pattern.shape[2:]))
            t_interactiveness_patterns = torch.from_numpy(interactiveness_patterns).to(device=device)
            mb.person_data.append(t_interactiveness_patterns)
        Timer.get('GetBatch').toc(discard=5)
        return mb


if __name__ == '__main__':
    HicoHake()

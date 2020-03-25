from enum import Enum
from typing import NamedTuple

import numpy as np
import torch

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', '__background__'
]


class Splits(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class HoiData(NamedTuple):
    im_idx: int
    fname_id: int
    p_idx: int
    o_idx: int
    hoi_classes: np.ndarray


class HoiTripletsData(NamedTuple):
    boxes: np.ndarray  # each is (image_idx, x1, y1, x2, y2, class)
    ho_pairs: np.ndarray  # each is (image_idx, hum_idx, obj_idx)
    labels: np.ndarray  # one-hot encoded


def get_hico_to_coco_mapping(hico_objects, split_objects=None):
    if split_objects is None:
        split_objects = hico_objects
    coco_obj_to_idx = {('hair dryer' if c == 'hair drier' else c).replace(' ', '_'): i for i, c in enumerate(COCO_CLASSES)}
    assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(hico_objects)
    mapping = np.array([coco_obj_to_idx[obj] for obj in split_objects], dtype=np.int)
    return mapping


def interactions_to_mat(hois, hico, np2np=False):
    # Default is Torch to Torch
    if not np2np:
        hois_np = hois.detach().cpu().numpy()
    else:
        hois_np = hois
    all_hois = np.stack(np.where(hois_np > 0), axis=1)
    all_interactions = np.concatenate([all_hois[:, :1], hico.interactions[all_hois[:, 1], :]], axis=1)
    inter_mat = np.zeros((hois.shape[0], hico.num_objects, hico.num_actions))
    inter_mat[all_interactions[:, 0], all_interactions[:, 2], all_interactions[:, 1]] = 1
    if np2np:
        return inter_mat
    else:
        return torch.from_numpy(inter_mat).to(hois)


def get_hoi_adjacency_matrix(dataset, isolate_null=True):
    interactions = dataset.full_dataset.interactions
    inter_obj_adj = np.zeros((dataset.full_dataset.num_interactions, dataset.full_dataset.num_objects))
    inter_obj_adj[np.arange(interactions.shape[0]), interactions[:, 0]] = 1

    inter_act_adj = np.zeros((dataset.full_dataset.num_interactions, dataset.full_dataset.num_actions))
    inter_act_adj[np.arange(interactions.shape[0]), interactions[:, 1]] = 1

    adj = inter_obj_adj @ inter_obj_adj.T + inter_act_adj @ inter_act_adj.T
    adj = torch.from_numpy(adj).clamp(max=1).float()

    if isolate_null:
        null_hois = np.flatnonzero(np.any(inter_act_adj[:, 1:], axis=1))
        adj[null_hois, :] = 0
        adj[:, null_hois] = 0
        return adj
    else:
        return adj


def get_noun_verb_adj_mat(dataset, isolate_null=True):
    noun_verb_links = torch.from_numpy((dataset.full_dataset.oa_pair_to_interaction >= 0).astype(np.float32))
    if isolate_null:
        noun_verb_links[:, 0] = 0
    return noun_verb_links


def filter_on_score(scores, min_score=None, maxn=0, keep_one=True):
    inds = np.argsort(scores)[::-1]
    im_person_scores = scores[inds]
    if min_score is not None:
        to_keep = im_person_scores > min_score
    else:
        to_keep = np.ones_like(inds, dtype=bool)
    if keep_one:
        to_keep[0] = True
    if maxn > 0:
        to_keep[maxn:] = False
    inds = inds[to_keep]
    if min_score is not None:
        assert np.all(im_person_scores[to_keep][1:] > min_score)
        assert im_person_scores[to_keep][0] > min_score or keep_one
        assert maxn == 0 or (np.all(im_person_scores[len(inds):maxn] <= min_score))
    return inds

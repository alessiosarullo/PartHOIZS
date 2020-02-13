# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Adapted from lib/ult/ult.py in https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network.
"""

import cv2
import numpy as np


def pattern_bbox_transform(human_boxes, object_boxes, size, part_boxes=None):
    def _crop_to_human(part_boxes, human_box):
        part_boxes[..., :2] = np.maximum(part_boxes[..., :2], human_box[..., None, :2])
        part_boxes[..., 2:] = np.minimum(part_boxes[..., 2:], human_box[..., None, 2:])
        idxs = np.any(part_boxes[..., 2:] - part_boxes[..., :2] <= 0, axis=-1)
        if np.any(idxs):
            idxs_h = np.any(idxs, axis=2)
            part_boxes[idxs, :2] = np.minimum(part_boxes[idxs, :2], human_box[idxs_h, None, 2:] - 1)
            part_boxes[idxs, 2:] = np.maximum(part_boxes[idxs, 2:], human_box[idxs_h, None, :2] + 1)
        assert part_boxes_orig is None or np.all(part_boxes[..., 2:] - part_boxes[..., :2] > 0)
        return part_boxes

    def _compute_union_bb(human_box, object_box):
        return np.concatenate([np.minimum(human_box[..., :2], object_box[..., :2]),
                               np.maximum(human_box[..., 2:], object_box[..., 2:])
                               ], axis=-1)  # P x M x 4

    def _shift_and_resize(boxes, union_boxes, scale):
        # boxes: P x M x n x 4
        # union_boxes: P x M x 4
        # scale: P x M
        scale = scale[..., None, None]
        boxes[..., 0::2] -= union_boxes[..., None, [0]]
        boxes[..., 1::2] -= union_boxes[..., None, [1]]
        boxes[..., :2] = np.minimum(size - 2, boxes[..., :2] * scale)
        boxes[..., 2:] = np.maximum(1, np.minimum(size - 1, boxes[..., 2:] * scale))

    part_boxes_orig = part_boxes
    if part_boxes_orig is None:
        part_boxes = np.zeros((*human_boxes.shape[:-1], 0, human_boxes.shape[-1]))

    human_boxes = human_boxes.copy()
    object_boxes = object_boxes.copy()
    part_boxes = part_boxes.copy()

    if human_boxes.ndim == 1:
        assert object_boxes.ndim == part_boxes.ndim - 1 == 1
        human_boxes = human_boxes[None, None, :]  # P x 1 x 4
        object_boxes = object_boxes[None, None, :]  # 1 x M x 4
        part_boxes = part_boxes[None, None, :]  # P x 1 x K x 4
    else:
        assert human_boxes.ndim == object_boxes.ndim == part_boxes.ndim - 1 == 2
        human_boxes = human_boxes[:, None, :]  # P x 1 x 4
        object_boxes = object_boxes[None, :, :]  # 1 x M x 4
        part_boxes = part_boxes[:, None, :, :]  # P x 1 x K x 4
    P = human_boxes.shape[0]
    M = object_boxes.shape[1]
    assert part_boxes.shape[0] == P or part_boxes.shape[0] == 0

    part_boxes = _crop_to_human(part_boxes, human_boxes)
    interaction_pattern = _compute_union_bb(human_boxes, object_boxes)  # P x M x 4

    heights = interaction_pattern[..., 3] - interaction_pattern[..., 1] + 1  # P x M
    widths = interaction_pattern[..., 2] - interaction_pattern[..., 0] + 1  # P x M
    k = (widths > heights).astype(np.int)

    pair_boxes = np.concatenate([np.tile(human_boxes[:, :, None, :], reps=(1, M, 1, 1)),  # P x M x 1 x 4
                                 np.tile(object_boxes[:, :, None, :], reps=(P, 1, 1, 1)),  # P x M x 1 x 4
                                 np.tile(part_boxes, reps=(1, M, 1, 1)),  # P x 1 x K x 4
                                 ], axis=2)  # P x M x (1 + 1 + K) x 4
    _shift_and_resize(pair_boxes, interaction_pattern, scale=size / np.maximum(heights, widths))

    human_boxes = pair_boxes[:, :, 0, :]  # P x M x 4
    object_boxes = pair_boxes[:, :, 1, :]  # P x M x 4
    part_boxes = pair_boxes[:, :, 2:, :]  # P x M x K x 4
    interaction_pattern = _compute_union_bb(human_boxes, object_boxes)

    all_rows, all_cols = np.mgrid[:P, :M]
    mask = (human_boxes[all_rows, all_cols, 3 - k] > object_boxes[all_rows, all_cols, 3 - k])  # P x M
    rows, cols = np.where(mask)
    human_boxes[rows, cols, 3 - k[mask]] = size - 1
    rows, cols = np.where(~mask)
    object_boxes[rows, cols, 3 - k[~mask]] = size - 1

    shifts = size / 2 - (interaction_pattern[all_rows, all_cols, 2 + k] + 1) / 2  # P x M
    deltas = np.stack([shifts * (1 - k),
                       shifts * k,
                       shifts * (1 - k),
                       shifts * k], axis=-1)  # P x M x 4
    human_boxes = human_boxes + deltas
    object_boxes = object_boxes + deltas
    part_boxes = part_boxes + deltas[:, :, None, :]
    part_boxes = _crop_to_human(part_boxes, human_boxes)

    return np.round(human_boxes), np.round(object_boxes), np.round(part_boxes) if part_boxes_orig is not None else None


def draw_relation(joints, size):
    # joints: P x M x K x 2

    joint_relation = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10], [11, 17], [12, 17], [11, 13],
                      [12, 14], [13, 15], [14, 16]]
    color = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    P, M = joints.shape[:2]
    skeleton = np.zeros((P, M, size, size, 1), dtype="float32")

    for p in range(P):
        for m in range(M):
            for i, (j1, j2) in enumerate(joint_relation):
                cv2.line(skeleton[p, m], tuple(joints[p, m, j1, :]), tuple(joints[p, m, j2, :]), color[i])

    return skeleton


def get_skeleton(unscaled_human_boxes, human_poses, scaled_human_boxes, size):
    # unscaled_human_boxes: P x 4
    # human_poses: P x K x 3
    # scaled_human_boxes: P x M x 4
    P, M = scaled_human_boxes.shape[:2]
    K = human_poses.shape[1]

    unscaled_human_boxes = unscaled_human_boxes[:, None, None, :]  # P x 1 x 1 x 4
    human_poses = human_poses[:, None, :, :]  # P x 1 x K x 4
    scaled_human_boxes = scaled_human_boxes[..., None, :]  # P x M x 1 x 4

    dims = unscaled_human_boxes[..., 2:] - unscaled_human_boxes[..., :2] + 1  # P x 1 x 2
    pattern_dims = scaled_human_boxes[..., 2:] - scaled_human_boxes[..., :2] + 1  # P x M x 2
    ratios = (human_poses[..., :2] - unscaled_human_boxes[..., :2]) / dims  # P x 1 x K x 2

    joints = np.zeros((P, M, K + 1, 2), dtype='int32')
    joints[..., :K, :] = np.minimum(size - 1, np.round(ratios * pattern_dims + scaled_human_boxes[..., :2]).astype(np.int))
    joints[..., K, :] = (joints[..., 5, :] + joints[..., 6, :]) / 2

    return draw_relation(joints, size=size)


def get_patterns(boxes, size, fill_boxes):
    range_v = np.arange(size).reshape((1, -1))
    mask_col = (boxes[:, 0, None] <= range_v) & (range_v <= boxes[:, 2, None])
    mask_row = (boxes[:, 1, None] <= range_v) & (range_v <= boxes[:, 3, None])
    mask = mask_col[:, None, :] & mask_row[:, :, None]
    if ~fill_boxes:
        mask_col = (boxes[:, 0, None] + 1 <= range_v) & (range_v <= boxes[:, 2, None] - 1)
        mask_row = (boxes[:, 1, None] + 1 <= range_v) & (range_v <= boxes[:, 3, None] - 1)
        mask = mask & ~(mask_col[:, None, :] & mask_row[:, :, None])
    patterns = mask.astype(np.int)
    return patterns


def get_next_sp_with_pose(human_boxes, object_boxes, human_poses, *, size=64, fill_boxes=True, part_boxes=None):
    """
    Return channels: ([part boxes], human box, object box, skeleton)
    :param human_boxes:
    :param object_boxes:
    :param human_poses:
    :param num_joints:
    :param size:
    :param fill_boxes:
    :param part_boxes:
    :return:
    """
    assert human_boxes.ndim == object_boxes.ndim == human_poses.ndim - 1 == 2 and (part_boxes is None or part_boxes.ndim == 3)
    P, K_coco = human_poses.shape[:2]
    M = object_boxes.shape[0]
    assert human_boxes.shape[0] == P and (part_boxes is None or part_boxes.shape[0] == P)
    assert human_boxes.shape[-1] == object_boxes.shape[-1] == 4 and (part_boxes is None or part_boxes.shape[-1] == 4)

    humans, objs, parts = pattern_bbox_transform(human_boxes, object_boxes, size=size, part_boxes=part_boxes)  # P x M (x K) x 4

    boxes = np.stack([humans, objs], axis=2)
    if part_boxes is not None:
        boxes = np.concatenate([parts, boxes], axis=2)

    Kplus2 = boxes.shape[2]
    boxes = boxes.astype(np.int).reshape((P * M * Kplus2, 4))
    patterns = get_patterns(boxes, size, fill_boxes=fill_boxes)
    patterns = patterns.reshape((P, M, Kplus2, size, size)).transpose([0, 1, 3, 4, 2])

    skeleton = get_skeleton(human_boxes, human_poses, humans, size=size)

    patterns = np.concatenate((patterns, skeleton), axis=-1)

    return patterns

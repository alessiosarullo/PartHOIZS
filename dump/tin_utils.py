# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Adapted from lib/ult/ult.py in https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network.
"""

import cv2
import numpy as np
from PIL import ImageDraw


def old_bbox_trans(human_box, object_box, size, part_boxes=None):
    def _shift_and_resize(boxes):
        boxes = np.atleast_2d(boxes)
        boxes[:, 0] -= interaction_pattern[0]
        boxes[:, 1] -= interaction_pattern[1]
        boxes[:, 2] -= interaction_pattern[0]
        boxes[:, 3] -= interaction_pattern[1]
        boxes[:, 0] = 0 + size * boxes[:, 0] / bigger_dim
        boxes[:, 1] = 0 + size * boxes[:, 1] / bigger_dim
        boxes[:, 2 + k] = (size * smaller_dim / bigger_dim - 1) - size * (smaller_dim - 1 - boxes[:, 2 + k]) / bigger_dim
        boxes[:, 3 - k] = (size - 1) - size * (bigger_dim - 1 - boxes[:, 3 - k]) / bigger_dim

    if part_boxes is None:
        part_boxes = np.zeros((0, 4))

    human_box = human_box.copy()
    object_box = object_box.copy()
    part_boxes = part_boxes.copy()

    # Crop part boxes to human
    part_boxes[:, :2] = np.maximum(human_box[:2][None, :], part_boxes[:, :2])
    part_boxes[:, 2:] = np.minimum(human_box[2:][None, :], part_boxes[:, 2:])
    assert np.all(part_boxes[:, 2:] - part_boxes[:, :2] > 0)

    interaction_pattern = [min(human_box[0], object_box[0]),
                           min(human_box[1], object_box[1]),
                           max(human_box[2], object_box[2]),
                           max(human_box[3], object_box[3])]

    height = interaction_pattern[3] - interaction_pattern[1] + 1
    width = interaction_pattern[2] - interaction_pattern[0] + 1
    if height > width:
        k = 0
        bigger_dim = height
        smaller_dim = width
    else:
        k = 1
        bigger_dim = width
        smaller_dim = height

    _shift_and_resize(human_box)
    _shift_and_resize(object_box)
    _shift_and_resize(part_boxes)

    interaction_pattern = [min(human_box[0], object_box[0]),
                           min(human_box[1], object_box[1]),
                           max(human_box[2], object_box[2]),
                           max(human_box[3], object_box[3])]

    if human_box[3 - k] > object_box[3 - k]:
        human_box[3 - k] = size - 1
    else:
        object_box[3 - k] = size - 1

    shift = size / 2 - (interaction_pattern[2 + k] + 1) / 2
    delta = np.array([shift * (1 - k), shift * k] * 2)
    human_box = human_box + delta
    object_box = object_box + delta
    part_boxes = part_boxes + delta[None, :]

    return np.round(human_box), np.round(object_box), np.round(part_boxes)


def old_draw_relation(joints, size):
    joint_relation = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10], [11, 17], [12, 17], [11, 13],
                      [12, 14], [13, 15], [14, 16]]
    color = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))

    # cv2.rectangle(skeleton, (int(human_pattern[0]), int(human_pattern[1])), (int(human_pattern[2]), int(human_pattern[3])), (255))
    # cv2.imshow("Joints", skeleton)
    # cv2.waitKey(0)
    # print(skeleton[:,:,0])

    return skeleton


def old_get_skeleton(human_boxes, human_poses, human_pattern, num_joints, size):
    width = human_boxes[2] - human_boxes[0] + 1
    height = human_boxes[3] - human_boxes[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_poses[i, :]
        x_ratio = (joint_x - human_boxes[0]) / float(width)
        y_ratio = (joint_y - human_boxes[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    return old_draw_relation(joints, size=size)


def old_get_patterns(boxes, size, fill_boxes):
    pattern = np.zeros((size, size, boxes.shape[0]), dtype='float32')
    for k, box in enumerate(boxes):
        box = box.astype(np.int)
        if fill_boxes:
            pattern[box[1]:box[3] + 1, box[0]:box[2] + 1, 0] = 1
        else:
            pattern[box[1]:box[3] + 1, box[0], k] = 1
            pattern[box[1]:box[3] + 1, box[2], k] = 1
            pattern[box[1], box[0]:box[2] + 1, k] = 1
            pattern[box[3], box[0]:box[2] + 1, k] = 1
    return pattern


def old_get_next_sp_with_pose(human_box, object_box, human_pose, *, num_joints=17, size=64, fill_boxes=True, part_boxes=None):
    """
    Return channels: ([part boxes], human box, object box, skeleton)
    :param human_box:
    :param object_box:
    :param human_pose:
    :param num_joints:
    :param size:
    :param fill_boxes:
    :param part_boxes:
    :return:
    """
    assert human_box.shape[-1] == object_box.shape[-1] == 4 and (part_boxes is None or part_boxes.shape[-1] == 4)
    human, obj, parts = old_bbox_trans(human_box, object_box, size=size, part_boxes=part_boxes)

    boxes = np.stack([human, obj], axis=0)
    if part_boxes is not None:
        boxes = np.concatenate([parts, boxes], axis=0)

    pattern = old_get_patterns(boxes, size, fill_boxes=fill_boxes)

    if human_pose is not None:
        skeleton = old_get_skeleton(human_box, human_pose, human, num_joints=num_joints, size=size)
    else:
        skeleton = np.zeros((size, size, 1), dtype='float32')
        skeleton[int(human[1]):int(human[3]) + 1, int(human[0]):int(human[2]) + 1, 0] = 0.05

    pattern = np.concatenate((pattern, skeleton), axis=2)

    return pattern

import cv2
import numpy as np
import torch


def compute_ious(boxes_a, boxes_b):
    if isinstance(boxes_a, np.ndarray):
        assert isinstance(boxes_b, np.ndarray)
        max_xy = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        min_xy = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
        intersection_dims = np.maximum(0, max_xy - min_xy + 1.0)  # A x B x 2, where last dim is [width, height]
        intersections_areas = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

        areas_a = ((boxes_a[:, 2] - boxes_a[:, 0] + 1.0) *
                   (boxes_a[:, 3] - boxes_a[:, 1] + 1.0))[:, None]  # Ax1
        areas_b = ((boxes_b[:, 2] - boxes_b[:, 0] + 1.0) *
                   (boxes_b[:, 3] - boxes_b[:, 1] + 1.0))[None, :]  # 1xB
        union_areas = areas_a + areas_b - intersections_areas
        return intersections_areas / union_areas
    else:
        A = boxes_a.size(0)
        B = boxes_b.size(0)
        max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           boxes_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((boxes_a[:, 2] - boxes_a[:, 0] + 1.0) *
                  (boxes_a[:, 3] - boxes_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((boxes_b[:, 2] - boxes_b[:, 0] + 1.0) *
                  (boxes_b[:, 3] - boxes_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


def get_union_boxes(boxes, union_inds):
    assert union_inds.shape[1] == 2
    union_rois = np.concatenate([
        np.minimum(boxes[:, :2][union_inds[:, 0]], boxes[:, :2][union_inds[:, 1]]),
        np.maximum(boxes[:, 2:][union_inds[:, 0]], boxes[:, 2:][union_inds[:, 1]]),
    ], axis=1)
    return union_rois


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def rescale_masks_to_img(masks, ref_boxes, im_h, im_w):
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    mask_resolution = masks.shape[1]
    assert masks.shape[2] == mask_resolution
    scale = (mask_resolution + 2.0) / mask_resolution
    ref_boxes = expand_boxes(ref_boxes, scale).astype(np.int32)
    padded_mask = np.zeros((mask_resolution + 2, mask_resolution + 2), dtype=np.float32)

    im_masks = []
    for i, ref_box in enumerate(ref_boxes):
        padded_mask[1:-1, 1:-1] = masks[i, :, :]

        # Resize mask to box scale
        w = (ref_box[2] - ref_box[0] + 1)
        h = (ref_box[3] - ref_box[1] + 1)
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        mask = np.array(cv2.resize(padded_mask, (w, h)).round(), dtype=np.uint8)

        # Find image mask
        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[2] + 1, im_w)
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[3] + 1, im_h)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

        im_masks.append(im_mask)

    im_masks = np.stack(im_masks, axis=0)
    assert im_masks.shape[0] == ref_boxes.shape[0]
    assert im_masks.shape[1] == im_h and im_masks.shape[2] == im_w
    return im_masks

import argparse
import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, average_precision_score

from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hico_hake import HicoHakeKPSplit, HicoHake
from lib.dataset.utils import Splits


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=sorted(FUNCS.keys()))
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--max_obj', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', default=False, action='store_true')
    namespace = parser.parse_known_args()
    sys.argv = sys.argv[:1] + namespace[1]
    cfg.parse_args(fail_if_missing=False, reset=True)
    return namespace[0]


def widths(bbs):
    return bbs[..., 2] - bbs[..., 0]


def heights(bbs):
    return bbs[..., 3] - bbs[..., 1]


def compute_center_dists(kp_boxes, kp_1hot_mask, all_obj_box_centers, im_relevant_obj_per_kp, img_wh, hh):
    kp_boxes[..., [0, 2]] /= img_wh[0]
    kp_boxes[..., [1, 3]] /= img_wh[1]
    assert np.all((0 <= kp_boxes) & (kp_boxes <= 1))

    kp_box_centers = np.stack((kp_boxes[..., 2] + kp_boxes[..., 0], kp_boxes[..., 3] + kp_boxes[..., 1]), axis=-1) / 2
    obj_box_centers = all_obj_box_centers[im_relevant_obj_per_kp]
    obj_box_centers[..., 0] /= img_wh[0]
    obj_box_centers[..., 1] /= img_wh[1]
    assert np.all((0 <= kp_box_centers) & (kp_box_centers <= 1))
    assert np.all((0 <= obj_box_centers) & (obj_box_centers < 1))
    center_dist_to_obj_per_kp_box = np.where(im_relevant_obj_per_kp >= 0,
                                             np.sqrt(np.sum((kp_box_centers - obj_box_centers) ** 2, axis=-1)),
                                             np.inf)
    assert np.all(((0 <= center_dist_to_obj_per_kp_box) & (center_dist_to_obj_per_kp_box < np.sqrt(2)))
                  | np.isinf(center_dist_to_obj_per_kp_box))
    # Keypoints -> parts
    center_dist_to_obj_per_part = []
    for part_idx, kps in enumerate(hh.part_to_kp):
        center_dist_to_obj_per_part.append(center_dist_to_obj_per_kp_box[np.any(kp_1hot_mask[..., kps], axis=-1)].reshape(-1))
    return center_dist_to_obj_per_part


def compute_edge_dists(kp_boxes, kp_1hot_mask, pc_obj_boxes, im_relevant_obj_per_kp, img_wh, hh):
    kp_boxes[..., [0, 2]] /= img_wh[0]
    kp_boxes[..., [1, 3]] /= img_wh[1]
    assert np.all((0 <= kp_boxes) & (kp_boxes <= 1))

    im_obj_boxes_per_kp = pc_obj_boxes[im_relevant_obj_per_kp]
    im_obj_boxes_per_kp[im_relevant_obj_per_kp < 0] = np.nan
    im_obj_boxes_per_kp[..., [0, 2]] /= img_wh[0]
    im_obj_boxes_per_kp[..., [1, 3]] /= img_wh[1]
    assert np.all(((0 <= im_obj_boxes_per_kp) & (im_obj_boxes_per_kp <= 1)) | np.isnan(im_obj_boxes_per_kp))

    union_boxes = np.stack([np.minimum(kp_boxes[..., 0], im_obj_boxes_per_kp[..., 0]),
                            np.minimum(kp_boxes[..., 1], im_obj_boxes_per_kp[..., 1]),
                            np.maximum(kp_boxes[..., 2], im_obj_boxes_per_kp[..., 2]),
                            np.maximum(kp_boxes[..., 3], im_obj_boxes_per_kp[..., 3])],
                           axis=-1)
    inner_rect_width = np.maximum(0, widths(union_boxes) - widths(kp_boxes) - widths(im_obj_boxes_per_kp))
    inner_rect_height = np.maximum(0, heights(union_boxes) - heights(kp_boxes) - heights(im_obj_boxes_per_kp))
    edge_dist_to_obj_per_kp_box = np.sqrt(inner_rect_width ** 2 + inner_rect_height ** 2)
    edge_dist_to_obj_per_kp_box[im_relevant_obj_per_kp < 0] = np.inf
    assert np.all(((0 <= edge_dist_to_obj_per_kp_box) & (edge_dist_to_obj_per_kp_box < np.sqrt(2))) | np.isinf(edge_dist_to_obj_per_kp_box))
    # Keypoints -> parts
    edge_dist_to_obj_per_part = []
    for part_idx, kps in enumerate(hh.part_to_kp):
        edge_dist_to_obj_per_part.append(edge_dist_to_obj_per_kp_box[np.any(kp_1hot_mask[..., kps], axis=-1)].reshape(-1))
    return edge_dist_to_obj_per_part


def compute_obj_boxes_ious(im_kp_boxes, kp_1hot_mask, im_obj_boxes, hh):
    obj_ious_per_part = np.zeros((hh.num_parts, im_obj_boxes.shape[0]))
    for part_idx, kps in enumerate(hh.part_to_kp):
        kp_boxes_for_part = im_kp_boxes[np.any(kp_1hot_mask[..., kps], axis=-1), :]
        assert kp_boxes_for_part.ndim == 2 and kp_boxes_for_part.shape[1] == 4
        if kp_boxes_for_part.shape[0] == 0:
            continue

        ious = compute_ious(im_obj_boxes, kp_boxes_for_part)
        # intersecting = ious.any(axis=1)
        # kp_match_per_obj = np.argmax(ious, axis=1)
        match_ious = np.max(ious, axis=1)
        obj_ious_per_part[part_idx, :] = match_ious
    return obj_ious_per_part


def compute_kp_boxes_ious(im_kp_boxes, kp_1hot_mask, im_obj_boxes, hh):
    kp_ious_per_part = np.zeros(hh.num_parts)
    for part_idx, kps in enumerate(hh.part_to_kp):
        kp_boxes_for_part = im_kp_boxes[np.any(kp_1hot_mask[..., kps], axis=-1), :]
        assert kp_boxes_for_part.ndim == 2 and kp_boxes_for_part.shape[1] == 4
        if kp_boxes_for_part.shape[0] == 0:
            continue
        ious = compute_ious(kp_boxes_for_part, im_obj_boxes)
        kp_ious_per_part[part_idx] = np.max(ious)
    return kp_ious_per_part


def analyse_interactiveness(args, hh: HicoHake, split):
    save_dir = os.path.join('analysis', 'output', 'part_obj_dist_interactiveness_hist', 'gt', split.value)
    os.makedirs(save_dir, exist_ok=True)

    cache_fn = os.path.join('cache', 'analysis.pkl')

    compute = True
    load_if_possible = False
    if load_if_possible:
        try:
            with open(cache_fn, 'rb') as f:
                stats_per_interactiveness_per_part = pickle.load(f)
                compute = False
        except FileNotFoundError:
            pass

    if compute:
        hhkps = HicoHakeKPSplit(split, full_dataset=hh)

        all_obj_boxes = hhkps.pc_obj_boxes
        all_obj_box_centers = np.stack((all_obj_boxes[:, 2] + all_obj_boxes[:, 0], all_obj_boxes[:, 3] + all_obj_boxes[:, 1]), axis=1) / 2

        n = len(hhkps) if args.num_imgs <= 0 else args.num_imgs
        stats_per_interactiveness_per_part = {part_idx: {} for part_idx in range(hh.num_parts)}
        for idx in range(n):
            if idx % 1000 == 0:
                print(f'Image {idx + 1}/{n}.')

            im_data = hhkps.pc_img_data[idx]
            try:
                person_inds = im_data['person_inds']
                obj_inds = im_data['obj_inds']
            except KeyError:
                # print(f'No person/object data for image {idx}.')
                continue

            labels = hhkps.part_labels[idx, :]
            per_part_interactiveness_labels = [labels[acts[-1]] == 0 for acts in hh.actions_per_part]

            img_wh = hhkps.img_dims[idx]

            im_person_boxes = hhkps.pc_person_boxes[person_inds]
            im_person_kps = hhkps.pc_coco_kps[person_inds]
            im_kp_boxes = hhkps.pc_hake_kp_boxes[person_inds]
            im_kp_feats = hhkps.pc_hake_kp_feats[person_inds]
            im_obj_boxes = hhkps.pc_obj_boxes[obj_inds]
            im_obj_scores = hhkps.pc_obj_scores[obj_inds]
            im_obj_feats = hhkps.pc_obj_feats[obj_inds]
            kp_box_prox_to_obj_fmaps = hhkps.kp_boxes_obj_proximity_fmaps[person_inds]
            im_relevant_obj_per_kp = hhkps.most_relevant_obj_per_kp_box[person_inds]

            im_kp_scores = im_kp_boxes[:, :, 4]
            im_kp_boxes = im_kp_boxes[:, :, :4]

            num_keypoints = len(hh.keypoints)
            kp_1hot_mask = np.zeros((im_kp_boxes.shape[0], im_kp_boxes.shape[1], num_keypoints), dtype=bool)
            kp_1hot_mask[:, np.arange(num_keypoints), np.arange(num_keypoints)] = 1

            im_kp_boxes = im_kp_boxes.reshape(-1, 4)
            kp_1hot_mask = kp_1hot_mask.reshape(-1, num_keypoints)
            im_relevant_obj_per_kp = im_relevant_obj_per_kp.reshape(-1)

            keep = im_kp_scores > 0.1
            if not keep.any():
                continue

            keep = keep.reshape(-1)
            im_kp_boxes = im_kp_boxes[keep]
            kp_1hot_mask = kp_1hot_mask[keep]
            im_relevant_obj_per_kp = im_relevant_obj_per_kp[keep]

            center_dist_to_obj_per_part = compute_center_dists(im_kp_boxes.copy(), kp_1hot_mask, all_obj_box_centers, im_relevant_obj_per_kp,
                                                               img_wh, hh)
            edge_dist_to_obj_per_part = compute_edge_dists(im_kp_boxes.copy(), kp_1hot_mask, hhkps.pc_obj_boxes, im_relevant_obj_per_kp, img_wh, hh)
            obj_ious_per_part = compute_obj_boxes_ious(im_kp_boxes, kp_1hot_mask, im_obj_boxes, hh)
            kp_ious_per_part = compute_kp_boxes_ious(im_kp_boxes, kp_1hot_mask, im_obj_boxes, hh)

            # Build data structure
            for part_idx, kps in enumerate(hh.part_to_kp):
                stats_per_interactiveness_per_part[part_idx].setdefault('interactiveness', []).append(per_part_interactiveness_labels[part_idx])
                stats_per_interactiveness_per_part[part_idx].setdefault('center_dist', []).append(center_dist_to_obj_per_part[part_idx])
                stats_per_interactiveness_per_part[part_idx].setdefault('edge_dist', []).append(edge_dist_to_obj_per_part[part_idx])
                stats_per_interactiveness_per_part[part_idx].setdefault('obj_iou', []).append(obj_ious_per_part[part_idx, :])
                stats_per_interactiveness_per_part[part_idx].setdefault('kp_iou', []).append(np.atleast_1d(kp_ious_per_part[part_idx]))

        with open(cache_fn, 'wb') as f:
            pickle.dump(stats_per_interactiveness_per_part, f)

    def _hist(ax, values, bins, rwidth=0.8):
        u, y = np.unique(values, return_counts=True)
        assert u[0] > 0
        hist = np.zeros(len(bins))
        hist[u - 1] = y
        ax.bar(bins, hist, width=(bins[1] - bins[0]) * rwidth, align='edge')

    baselines = []
    labels = hh.split_part_annotations[split]
    for part_idx in range(hh.num_parts):
        actions = hh.actions_per_part[part_idx]
        baselines.append(np.sum(labels[:, actions[:-1]].any(axis=1)) / labels.shape[0])

    for part_idx, stats_per_interactiveness in stats_per_interactiveness_per_part.items():
        orig_interactiveness = np.array(stats_per_interactiveness['interactiveness'])
        num_stats = len(stats_per_interactiveness) - 1
        del stats_per_interactiveness['interactiveness']

        fig, axs = plt.subplots(num_stats, 2, tight_layout=True)
        for i, (k, v) in enumerate(stats_per_interactiveness.items()):
            max_value = np.sqrt(2) if 'dist' in k else 1

            interactiveness = np.concatenate([np.full_like(x, fill_value=orig_interactiveness[i]) for i, x in enumerate(v)]).astype(np.bool)
            stats = np.concatenate(v)

            inds = ~np.isinf(stats)
            stats = stats[inds]
            interactiveness = interactiveness[inds]

            inds = np.argsort(stats)
            stats = stats[inds]
            interactiveness = interactiveness[inds]

            bins = np.linspace(0, max_value, 11)
            binned_stats = np.digitize(stats, bins=bins)
            assert np.all(binned_stats == np.sort(binned_stats))

            # This should be equivalent to ax.hist(values, bins=bins, rwidth=rwidth)
            _hist(ax=axs[i, 0], values=binned_stats[interactiveness], bins=bins, rwidth=0.8)
            axs[i, 0].set_title('"P"(x | interactiveness=1)')

            u, idxs = np.unique(binned_stats, return_index=True)
            binned_interactiveness = np.add.reduceat(interactiveness, idxs)
            binned_uninteractiveness = np.add.reduceat(~interactiveness, idxs)
            binned_interactiveness_perc = binned_interactiveness / (binned_interactiveness + binned_uninteractiveness)
            binned_interactiveness_perc[np.isnan(binned_interactiveness_perc)] = 0
            assert u[0] > 0
            y = np.zeros(len(bins))
            y[u - 1] = binned_interactiveness_perc
            axs[i, 1].bar(bins, y, width=(bins[1] - bins[0]) * 0.8, align='edge')
            axs[i, 1].axhline(y=baselines[part_idx])
            axs[i, 1].set_title('% Interactive')

        plt.savefig(os.path.join(save_dir, f'{hh.parts[part_idx]}'))


def analyse_obj_coverage(args, hh: HicoHake, split):
    max_obj_per_img = args.max_obj

    objs_per_img = np.minimum(1, hh.split_annotations[split] @ hh.interaction_to_object_mat)

    hhkps = HicoHakeKPSplit(split, full_dataset=hh)

    n = len(hhkps) if args.num_imgs <= 0 else args.num_imgs
    pred_obj_classes_per_img = np.zeros_like(objs_per_img, dtype=bool)
    pred_obj_scores_per_img = np.zeros_like(objs_per_img)
    for idx in range(n):
        if idx % 1000 == 0:
            print(f'Image {idx + 1}/{n}.')

        im_data = hhkps.pc_img_data[idx]
        try:
            person_inds = im_data['person_inds']
            obj_inds = im_data['obj_inds']
        except KeyError:
            # print(f'No person/object data for image {idx}.')
            continue

        im_obj_scores = hhkps.pc_obj_scores[obj_inds]
        if max_obj_per_img > 0:
            max_scores = im_obj_scores.max(axis=1)
            inds = np.argsort(max_scores)[::-1]
            im_obj_scores = im_obj_scores[inds[:max_obj_per_img], :]
        im_obj_classes = np.unique(np.argmax(im_obj_scores, axis=1))
        pred_obj_classes_per_img[idx, im_obj_classes] = 1

        pred_obj_scores_per_img[idx, :] = im_obj_scores.max(axis=0)

    hits = (objs_per_img.astype(bool) & pred_obj_classes_per_img)
    recall_per_obj = hits.sum(axis=0) / objs_per_img.sum(axis=0)
    ap_per_obj = average_precision_score(objs_per_img.astype(np.int), pred_obj_scores_per_img, average=None)
    # bins = np.linspace(0, 1, 11)

    fig, axs = plt.subplots(1, 2, tight_layout=True, squeeze=False)
    axs[0, 0].bar(np.arange(hh.num_objects), recall_per_obj, width=0.8, align='center')
    axs[0, 0].axhline(recall_per_obj[np.setdiff1d(np.arange(hh.num_objects), [49])].mean())
    axs[0, 0].set_title('Object recall')

    axs[0, 1].bar(np.arange(hh.num_objects), ap_per_obj, width=0.8, align='center')
    axs[0, 1].axhline(ap_per_obj[np.setdiff1d(np.arange(hh.num_objects), [49])].mean())
    axs[0, 1].set_title('Object mAP')

    save_dir = os.path.join('analysis', 'output', 'generic', 'gt', split.value)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'obj_coverage_max{max_obj_per_img}'))


def analyse_hico_hake_kps(args):
    if args.debug:
        try:  # PyCharm debugging
            print('Starting remote debugging (resume from debug server)')
            import pydevd_pycharm
            pydevd_pycharm.settrace('130.88.195.105', port=16008, stdoutToServer=True, stderrToServer=True)
            print('Remote debugging activated.')
        except:
            print('Remote debugging failed.')
            raise

    split = Splits[args.split.upper()]
    hh = HicoHake()
    FUNCS[args.func](args, hh, split)


FUNCS = {'i': analyse_interactiveness,
         'o': analyse_obj_coverage}

if __name__ == '__main__':
    analyse_hico_hake_kps(get_args())

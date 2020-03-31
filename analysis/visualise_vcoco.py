import argparse
import os
import sys
import time

import matplotlib

try:
    matplotlib.use('Qt5Agg')
    sys.argv[1:] = ['vis',
                    '--gt', '--pbb', '--obb',
                    '--num_imgs', '-1',
                    '--vis',
                    '--max_ppl', '0', '--max_obj', '0']
except ImportError:
    pass

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

from analysis.visualise_utils import Visualizer
from analysis.utils import analysis_hub
from lib.dataset.vcoco import VCoco, VCocoKPSplit
from lib.dataset.utils import Splits
from lib.dataset.tin_utils import get_next_sp_with_pose
from config import cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--kp_thr', type=float, default=0.05)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--part', default=False, action='store_true')
    parser.add_argument('--kp', default=False, action='store_true')
    parser.add_argument('--obb', default=False, action='store_true')
    parser.add_argument('--pbb', default=False, action='store_true')
    parser.add_argument('--gt', default=False, action='store_true')
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--tin', default=False, action='store_true')
    parser.add_argument('--rnd', default=False, action='store_true')

    namespace = parser.parse_known_args()
    args = namespace[0]
    sys.argv = sys.argv[:1] + namespace[1]
    cfg.parse_args(fail_if_missing=False, reset=True)
    return args


def vis_vcoco():
    args = get_args()
    split = Splits[args.split.upper()]

    folder = []
    if args.part:
        folder += ['partbbs']
    if args.kp:
        folder += ['kps']
    if args.obb:
        folder += ['obbs']
    if args.pbb:
        folder += ['pbbs']
    if not folder:
        folder = ['img']
    save_dir = os.path.join('analysis', 'output', 'vis', 'gt', '_'.join(folder), split.value)
    os.makedirs(save_dir, exist_ok=True)

    ds = VCoco()
    dssplit = VCocoKPSplit(split=split, full_dataset=ds, no_feats=True)

    objects_str = ds.objects
    actions_str = ds.actions
    det_data = ds._split_det_data[split]
    im_idx_to_ho_pairs_inds = {}
    for i, (image_idx, hum_idx, obj_idx) in enumerate(det_data.ho_pairs):
        im_idx_to_ho_pairs_inds.setdefault(image_idx, []).append(i)
    im_idx_to_ho_pairs_inds = {k: np.array(v) for k, v in im_idx_to_ho_pairs_inds.items()}

    n = dssplit.num_images if args.num_imgs <= 0 else args.num_imgs
    img_inds = list(range(n))
    if args.rnd:
        seed = np.random.randint(1_000_000_000)
        print('Seed:', seed)
        np.random.seed(seed)
        np.random.shuffle(img_inds)

    all_t = 0
    for idx in img_inds:
        # if idx != 27272:  # FIXME delete
        #     continue
        # rotated images:
        #     'train2015': [18679, 19135, 27301, 28302, 32020],
        #     'test2015': [3183, 7684, 8435, 8817],
        fname = ds.split_filenames[split][idx]

        print(f'Image {idx + 1:6d}/{n}, file {fname}.')

        try:
            img = Image.open(ds.get_img_path(split, fname))
        except:
            print(f'Error on image {idx}: {fname}.')
            continue

        visualizer = Visualizer(img, kp_thr=args.kp_thr)

        # Print annotations
        all_img_ho_anns_inds = im_idx_to_ho_pairs_inds[idx]
        img_hoi_pairs = ds._split_det_data[split].ho_pairs[all_img_ho_anns_inds]
        img_act_labels = ds._split_det_data[split].labels[all_img_ho_anns_inds]
        img_box_inds = np.unique(img_hoi_pairs[:, 1:])
        img_box_inds = img_box_inds[~np.isnan(img_box_inds)].astype(np.int)
        img_boxes_ext = ds._split_det_data[split].boxes[img_box_inds]
        if args.gt:
            img_boxes = img_boxes_ext[:, 1:5]
            img_box_classes = img_boxes_ext[:, 5].astype(np.int)
            gt_box_labels = [ds.objects[c] for c in img_box_classes]
            visualizer.overlay_instances(labels=gt_box_labels, boxes=img_boxes, alpha=0.7, line_style='--', color='red')

        gt_str = []
        for hp, l in zip(img_hoi_pairs, img_act_labels):
            obj_idx = hp[-1]
            if np.isnan(obj_idx):
                continue
            act_ind = np.flatnonzero(l).item()  # labels are one-hot encoded
            obj_ind = int(ds._split_det_data[split].boxes[int(obj_idx), -1])
            gt_str.append(f'{actions_str[act_ind]} {objects_str[obj_ind]}')
        print(f'\t\t{", ".join(gt_str)}')

        im_data = dssplit._feat_provider.img_data_cache[idx]
        try:
            person_inds = im_data['person_inds']
        except KeyError:
            print(f'No people in image {idx}.')
            continue

        person_boxes = dssplit._feat_provider.person_boxes[person_inds]
        person_kps = dssplit._feat_provider.coco_kps[person_inds]
        kp_boxes = dssplit._feat_provider.hake_kp_boxes[person_inds]

        if args.tin:
            try:
                obj_inds = im_data['obj_inds']
                obj_boxes = dssplit._feat_provider.obj_boxes[obj_inds]

                start = time.perf_counter()
                patterns = get_next_sp_with_pose(human_boxes=person_boxes, human_poses=person_kps, object_boxes=obj_boxes,
                                                 fill_boxes=False,
                                                 part_boxes=kp_boxes[:, :, :4]
                                                 )
                all_t += (time.perf_counter() - start)
                patterns = np.pad(patterns, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)], mode='constant', constant_values=1)

                print(f'{all_t * 1000 / (idx + 1):.3f}ms')

                pose_grid = patterns[:, :, :, :, -1]
                pose_grid = np.stack([pose_grid] * 3, axis=-1)  # greyscale -> RBG (still grey)
                pose_grid[..., 2] = patterns[:, :, :, :, -2]  # blue channel = object
                pose_grid[..., 1] = np.sum(patterns[:, :, :, :, :-3], axis=-1)  # green channel = parts
                pose_grid = pose_grid.transpose([0, 2, 1, 3, 4])
                pose_grid = pose_grid.reshape((patterns.shape[0] * patterns.shape[2], patterns.shape[1] * patterns.shape[3], 3))

                scores = dssplit._feat_provider.person_scores[person_inds]
                visualizer.overlay_instances(boxes=person_boxes, labels=[f'{s:.2f}'.lstrip('0') for s in scores], alpha=0.7)
                visualizer.overlay_instances(boxes=obj_boxes, alpha=0.7)

                plt.figure(figsize=(13, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(visualizer.output.get_image())
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(pose_grid)
                plt.axis('off')
                plt.show()
            except KeyError:
                print(f'Image {idx} is empty.')
            continue  # this is correct here (TIN => nothing else)

        if args.pbb:
            scores = dssplit._feat_provider.person_scores[person_inds]
            visualizer.overlay_instances(boxes=person_boxes, labels=[f'{s:.2f}'.lstrip('0') for s in scores], alpha=0.7)

        if args.obb:
            try:
                obj_inds = im_data['obj_inds']
                obj_boxes = dssplit._feat_provider.obj_boxes[obj_inds]
                obj_classes = np.argmax(dssplit._feat_provider.obj_scores[obj_inds], axis=1)
                obj_scores = dssplit._feat_provider.obj_scores[obj_inds]
                labels = [dssplit.full_dataset.objects[c] for c in obj_classes]
                labels = [f'{l} {obj_scores[i, obj_classes[i]]:.1f}' for i, l in enumerate(labels)]  # add scores
                visualizer.overlay_instances(labels=labels, boxes=obj_boxes, alpha=0.7)
            except KeyError:
                pass

        # Draw part bounding boxes
        if args.part:
            kp_boxes = kp_boxes.reshape(-1, 5)
            visualizer.overlay_instances(boxes=kp_boxes[:, :4], alpha=0.7)

        # Draw keypoints
        if args.kp:
            for kps in person_kps:
                visualizer.draw_keypoints(kps, print_names=not args.part)

        # Save output
        if args.vis:
            plt.imshow(visualizer.output.get_image())
            plt.show()
        else:
            visualizer.output.save(os.path.join(save_dir, fname))


if __name__ == '__main__':
    analysis_hub({'vis': vis_vcoco})

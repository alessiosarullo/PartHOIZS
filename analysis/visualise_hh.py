import argparse
import os
import sys
import time

import matplotlib

try:
    matplotlib.use('Qt5Agg')
    sys.argv[1:] = ['hhkps', '--tin', '--max_ppl', '3', '--max_obj', '3',
                    '--num_imgs', '-1',
                    # '--rnd',
                    '--filter'
                    ]
    # sys.argv[1:] = ['hhkps', '--vis',
    #                 # '--no_kp',
    #                 '--no_bb',
    #                 '--num_imgs', '0'
    #                 ]
except ImportError:
    pass

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

from analysis.visualise_utils import Visualizer
from analysis.utils import analysis_hub
from lib.dataset.hico_hake import HicoHakeKPSplit, HicoHake

from lib.dataset.tin_utils import get_next_sp_with_pose
import lib.utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--seenf', type=int, default=4)
    parser.add_argument('--kp_thr', type=float, default=0.05)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--no_bb', default=False, action='store_true')
    parser.add_argument('--no_kp', default=False, action='store_true')
    parser.add_argument('--obb', default=False, action='store_true')
    parser.add_argument('--pbb', default=False, action='store_true')
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--tin', default=False, action='store_true')
    parser.add_argument('--max_ppl', type=int, default=0)
    parser.add_argument('--max_obj', type=int, default=0)
    parser.add_argument('--rnd', default=False, action='store_true')
    parser.add_argument('--filter', default=False, action='store_true')
    return parser.parse_args()


def vis_hico_hake_kps():
    args = get_args()
    split = args.split

    folder = []
    if not args.no_bb:
        folder += ['bbs']
    if not args.no_kp:
        folder += ['kps']
    if args.obb:
        folder += ['obbs']
    if args.pbb:
        folder += ['pbbs']
    if not folder:
        folder = ['img']
    save_dir = os.path.join('analysis', 'output', 'vis', 'gt', '_'.join(folder), split)
    os.makedirs(save_dir, exist_ok=True)

    ds = HicoHake()
    dssplit = HicoHakeKPSplit(split=split, full_dataset=ds, no_feats=True)

    if args.seenf >= 0:
        seen_objs, unseen_objs, seen_acts, unseen_acts, seen_interactions, unseen_interactions = \
            lib.utils.get_zs_classes(hico_hake=ds, fname=f'zero-shot_inds/seen_inds_{args.seenf}.pkl.push')
        objects_str = [f'{o}{"*" if i in unseen_objs else ""}' for i, o in enumerate(ds.objects)]
        actions_str = [f'{a}{"*" if i in unseen_acts else ""}' for i, a in enumerate(ds.actions)]
        interactions_str = [f'{actions_str[a]} {objects_str[o]}' for a, o in ds.interactions]
    else:
        objects_str = ds.objects
        actions_str = ds.objects

    n = len(dssplit) if args.num_imgs <= 0 else args.num_imgs
    img_inds = list(range(n))
    if args.rnd:
        seed = np.random.randint(1_000_000_000)
        print('Seed:', seed)
        np.random.seed(seed)
        np.random.shuffle(img_inds)

    if args.filter:
        queries_str = [
            # ['cook', 'pizza'],
            # ['eat', 'sandwich'],
            # ['eat', 'apple'],
            # ['stab', 'person'],
            # ['hug', 'cat'],
            ['block', '*'],
        ]
        queries = set()
        for a, o in queries_str:
            if o == '*':
                o = np.arange(ds.num_objects)
            elif isinstance(o, (list, tuple)):
                o = np.array([ds.object_index[x] for x in o])
            else:
                o = ds.object_index[o]
            queries.update({x for x in np.atleast_1d(ds.oa_pair_to_interaction[o, ds.action_index[a]])})
        queries = np.array(sorted(queries - {-1}))
        if np.any(queries < 0):
            raise ValueError('Unknown interaction(s).')

        inds = np.flatnonzero(np.any(dssplit.labels[:, queries], axis=1))  # FIXME this assumes labels are for images
        print(f'{len(inds)} images retrieved.')
        img_inds = sorted(set(img_inds) & set(inds.tolist()))

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

        # Print annotations
        img_anns = ds.split_labels[dssplit.split][idx, :]
        gt_str = []
        for i, s in enumerate(img_anns):
            act_ind = ds.interactions[i, 0]
            obj_ind = ds.interactions[i, 1]
            if s > 0:
                gt_str.append(f'{actions_str[act_ind]} {objects_str[obj_ind]}')
        print(f'\t\t{", ".join(gt_str)}')

        # Visualise
        visualizer = Visualizer(img, kp_thr=args.kp_thr)

        im_data = dssplit._feat_provider.img_data_cache[idx]

        try:
            person_inds = im_data['person_inds']
        except KeyError:
            print(f'No people in image {idx}.')
            continue

        if args.max_ppl > 0:
            person_scores = dssplit._feat_provider.person_scores[person_inds]
            inds = np.argsort(person_scores)[::-1]
            person_inds = person_inds[inds[:args.max_ppl]]

        person_boxes = dssplit._feat_provider.person_boxes[person_inds]
        person_kps = dssplit._feat_provider.coco_kps[person_inds]
        kp_boxes = dssplit._feat_provider.hake_kp_boxes[person_inds]

        if args.tin:
            try:
                obj_inds = im_data['obj_inds']
                if args.max_obj > 0:
                    obj_scores = np.max(dssplit._feat_provider.obj_scores[obj_inds, :], axis=1)
                    inds = np.argsort(obj_scores)[::-1]
                    obj_inds = obj_inds[inds[:args.max_obj]]
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
                labels = [dssplit.full_dataset.objects[c] for c in obj_classes]
                # labels = [f'{hhkps.full_dataset.objects[c]} {im_data["obj_scores"][i, c]:.1f}' for i, c in enumerate(obj_classes)]
                visualizer.overlay_instances(labels=labels, boxes=obj_boxes, alpha=0.7)
            except KeyError:
                pass

        # Draw part bounding boxes
        if not args.no_bb:
            kp_boxes = kp_boxes.reshape(-1, 5)
            assert kp_boxes.shape[0] == person_boxes.shape[0] * len(ds.keypoints)
            labels = ds.keypoints * person_boxes.shape[0]
            labels = [f'{l} {kp_boxes[i, -1]:.1f}' for i, l in enumerate(labels)]
            visualizer.overlay_instances(labels=labels, boxes=kp_boxes[:, :4], alpha=0.7)

        # Draw keypoints
        if not args.no_kp:
            for kps in person_kps:
                visualizer.draw_keypoints(kps, print_names=args.no_bb)

        # Save output
        if args.vis:
            plt.imshow(visualizer.output.get_image())
            plt.show()
        else:
            visualizer.output.save(os.path.join(save_dir, fname))


if __name__ == '__main__':
    analysis_hub({'hhkps': vis_hico_hake_kps,
                  })

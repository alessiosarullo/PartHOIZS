import argparse
import os

import numpy as np
from PIL import Image

from analysis.visualise_utils import Visualizer
from lib.dataset.hico_hake import HicoHakeKPSplit, HicoHake
from lib.dataset.utils import Splits


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imgs', type=int, default=50)
    parser.add_argument('--kp_thr', type=float, default=0.05)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--no_bb', default=False, action='store_true')
    parser.add_argument('--no_kp', default=False, action='store_true')
    parser.add_argument('--obb', default=False, action='store_true')
    parser.add_argument('--pbb', default=False, action='store_true')
    return parser.parse_args()


def vis_hico_hake_kps(args):
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
    save_dir = os.path.join('analysis', 'output', 'vis', 'gt', '_'.join(folder), split.value)
    os.makedirs(save_dir, exist_ok=True)

    hh = HicoHake()
    hhkps = HicoHakeKPSplit(split, full_dataset=hh)

    n = len(hhkps) if args.num_imgs <= 0 else args.num_imgs
    for idx in range(n):
        fname = hh.split_filenames[split][idx]

        path = os.path.join(hh.get_img_dir(split), fname)
        print(f'Image {idx + 1:6d}/{n}, file {fname}.')

        try:
            img = Image.open(path)
        except:
            print(f'Error on image {idx}: {fname}.')
            continue

        visualizer = Visualizer(img, kp_thr=args.kp_thr)
        vis_output = visualizer.output

        im_data = hhkps.pc_img_data[idx]
        if im_data:
            person_inds = im_data['person_inds']
            person_boxes = hhkps.pc_person_boxes[person_inds]
            person_kps = hhkps.pc_coco_kps[person_inds]
            kp_boxes = hhkps.pc_hake_kp_boxes[person_inds].reshape(-1, 5)
            assert kp_boxes.shape[0] == person_boxes.shape[0] * len(hh.keypoints)

            if args.pbb:
                scores = hhkps.pc_person_scores[person_inds]
                vis_output = visualizer.overlay_instances(boxes=person_boxes, labels=[f'{s:.2f}'.lstrip('0') for s in scores], alpha=0.7)

            if args.obb:
                try:
                    obj_inds = im_data['obj_inds']
                    obj_boxes = hhkps.pc_obj_boxes[obj_inds]
                    obj_classes = np.argmax(hhkps.pc_obj_scores[obj_inds], axis=1)
                    labels = [hhkps.full_dataset.objects[c] for c in obj_classes]
                    # labels = [f'{hhkps.full_dataset.objects[c]} {im_data["obj_scores"][i, c]:.1f}' for i, c in enumerate(obj_classes)]
                    vis_output = visualizer.overlay_instances(labels=labels, boxes=obj_boxes, alpha=0.7)
                except KeyError:
                    pass

            # Draw part bounding boxes
            if not args.no_bb:
                labels = hh.keypoints * person_boxes.shape[0]
                labels = [f'{l} {kp_boxes[i, -1]:.1f}' for i, l in enumerate(labels)]
                vis_output = visualizer.overlay_instances(labels=labels, boxes=kp_boxes[:, :4], alpha=0.7)

            # Draw keypoints
            if not args.no_kp:
                for kps in person_kps:
                    vis_output = visualizer.draw_keypoints(kps, print_names=args.no_bb)

        # Save output
        vis_output.save(os.path.join(save_dir, fname))


if __name__ == '__main__':
    vis_hico_hake_kps(get_args())

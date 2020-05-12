import argparse
import os
from typing import Type

import h5py
import numpy as np

from lib.dataset.hoi_dataset_split import HoiDatasetSplit
from lib.dataset.det_gt_assignment import HumObjPairsModule

DATASETS = ['hico', 'vcoco', 'cocoa']


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, choices=DATASETS)
    parser.add_argument('--od_file', default=None)  # without the split and .h5 extension
    parser.add_argument('--kp_file', default=None)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--gt_iou_thr', default=0.5, type=float)
    parser.add_argument('--hoi_bg_ratio', default=10, type=float)  # oversampling
    return parser


def get_all_pairs(is_human: np.ndarray):
    num_boxes = is_human.shape[0]
    if num_boxes == 0:
        ho_infos = np.empty([0, 2], dtype=np.int)
    else:
        block_img_mat = np.ones(num_boxes)  # type: np.ndarray
        possible_rels_mat = block_img_mat - np.eye(num_boxes)  # no self relations
        possible_rels_mat[~is_human, :] = 0  # only from human
        hum_inds, obj_inds = np.where(possible_rels_mat)
        ho_infos = np.stack([hum_inds, obj_inds], axis=1).astype(np.int, copy=False)  # box indices are over the original boxes
    return ho_infos


def main():
    args = get_parser().parse_args()
    od_file_basename = args.od_file if args.od_file is not None else f'cache/precomputed_{args.ds}objs__mask_rcnn_X_101_32x8d_FPN_3x'
    kp_file_basename = args.kp_file if args.kp_file is not None else f'cache/precomputed_{args.ds}kps__keypoint_rcnn_R_101_FPN_3x'

    if args.debug:
        try:  # PyCharm debugging
            print('Starting remote debugging (resume from debug server)')
            import pydevd_pycharm
            pydevd_pycharm.settrace('130.88.195.105', port=16005, stdoutToServer=True, stderrToServer=True)
            print('Remote debugging activated.')
        except:
            print('Remote debugging failed.')
            raise

    if args.ds == 'hico':
        from lib.dataset.hicodet_hake import HicoDetHakeSplit as DsSplit
    elif args.ds == 'vcoco':
        from lib.dataset.vcoco import VCocoSplit as DsSplit
    elif args.ds == 'cocoa':
        from lib.dataset.cocoa import CocoaSplit as DsSplit
    else:
        raise ValueError(f'Dataset should be one of {DATASETS}.')
    DsSplit = DsSplit  # type: Type[HoiDatasetSplit]
    full_dataset = DsSplit.instantiate_full_dataset()

    pair_mod = HumObjPairsModule(dataset=full_dataset, gt_iou_thr=args.gt_iou_thr, hoi_bg_ratio=args.hoi_bg_ratio, null_as_bg=True)
    for split in ['train', 'test']:
        hds = DsSplit(split=split, full_dataset=full_dataset)  # type: HoiDatasetSplit

        od_file = h5py.File(f'{od_file_basename}_{split}.h5', 'r')
        kp_file = h5py.File(f'{kp_file_basename}_{split}.h5', 'r')

        hum_boxes = kp_file['boxes'][:]
        # hum_scores = kp_file['scores'][:]
        hum_fname_ids = kp_file['fname_ids'][:]
        assert hum_fname_ids.shape[0] == hum_boxes.shape[0]
        hum_fname_ids_to_inds = {}
        for i, fname_id in enumerate(hum_fname_ids):
            hum_fname_ids_to_inds.setdefault(fname_id, []).append(i)
        hum_fname_ids_to_inds = {k: np.array(v) for k, v in hum_fname_ids_to_inds.items()}

        obj_boxes = od_file['boxes'][:]
        obj_scores = od_file['box_scores'][:]
        obj_fname_ids = od_file['fname_ids'][:]
        assert obj_fname_ids.shape[0] == obj_boxes.shape[0]
        obj_fname_ids_to_inds = {}
        for i, fname_id in enumerate(obj_fname_ids):
            obj_fname_ids_to_inds.setdefault(fname_id, []).append(i)
        obj_fname_ids_to_inds = {k: np.array(v) for k, v in obj_fname_ids_to_inds.items()}

        all_ho_infos, all_labels, all_obj_labels, all_pstate_labels = [], [], [], []
        skipped_counter = 0
        for im_i, im_data in enumerate(hds.all_gt_img_data):
            fname_id = int(os.path.splitext(im_data.filename)[0].split('_')[-1])

            hum_inds_i = hum_fname_ids_to_inds.get(fname_id, None)
            obj_inds_i = obj_fname_ids_to_inds.get(fname_id, None)

            # # Filter out BG/person objects
            # if obj_inds_i is not None:
            #     obj_classes_i = np.argmax(obj_scores[obj_inds_i, :], axis=1)
            #     keep = (obj_classes_i >= 2)  # 0 and 1 are COCO's background and person class, respectively
            #     obj_inds_i = obj_inds_i[keep]
            #     if obj_inds_i.size == 0:
            #         obj_inds_i = None

            if hum_inds_i is not None and obj_inds_i is not None and im_data.boxes is not None:
                hum_boxes_i = hum_boxes[hum_inds_i, :]
                obj_boxes_i = obj_boxes[obj_inds_i, :]
                assert hum_boxes_i.shape[0] > 0 and obj_boxes_i.shape[0] > 0
                boxes_i = np.concatenate([hum_boxes_i, obj_boxes_i], axis=0)
                is_human_i = np.concatenate([np.ones(hum_boxes_i.shape[0], dtype=bool),
                                             np.zeros(obj_boxes_i.shape[0], dtype=bool)])
                boxes_inds_i = np.concatenate([hum_inds_i, obj_inds_i])

                if split == 'test':
                    ho_pairs = get_all_pairs(is_human_i)
                    if ho_pairs.shape[0] == 0:
                        ho_pairs = None
                    labels = obj_labels = pstate_labels = None
                else:
                    ho_pairs, labels, obj_labels, pstate_labels = pair_mod.compute_pairs(gt_img_data=im_data,
                                                                                         boxes=boxes_i,
                                                                                         is_human=is_human_i)

                if ho_pairs is not None:
                    assert not np.any(np.isnan(ho_pairs))
                    num_humans = hum_boxes_i.shape[0]
                    assert np.all(0 <= ho_pairs[:, 0]) and np.all(ho_pairs[:, 0] < num_humans)
                    ho_pairs = ho_pairs.astype(np.int, copy=False)

                    # Map from local indices to absolute ones
                    is_obj_human = is_human_i[ho_pairs[:, 1]]
                    ho_pairs[:, 0] = boxes_inds_i[ho_pairs[:, 0]]
                    ho_pairs[:, 1] = boxes_inds_i[ho_pairs[:, 1]]

                    ho_infos = np.concatenate([np.full((ho_pairs.shape[0], 1), fill_value=fname_id, dtype=np.int),
                                               ho_pairs,
                                               is_obj_human[:, None].astype(np.int)
                                               ], axis=1)
                    all_ho_infos.append(ho_infos)
                    if labels is not None:
                        all_labels.append(labels)
                    if obj_labels is not None:
                        all_obj_labels.append(obj_labels)
                    if pstate_labels is not None:
                        all_pstate_labels.append(pstate_labels)
                else:
                    assert labels is None and obj_labels is None and pstate_labels is None
                    print(f'No HOIs for image {im_data.filename}.')
                    skipped_counter += 1
            else:
                print(f'No predicted boxes for image {im_data.filename}.')
                skipped_counter += 1

            if im_i % 1000 == 0 or im_i == hds.num_images - 1:
                print(f'Image {im_i:6d}/{hds.num_images}')
        print(f'There were {skipped_counter} invalid images.')

        all_ho_infos = np.concatenate([x for x in all_ho_infos], axis=0)
        if split == 'test':
            assert not all_labels and not all_obj_labels and not all_pstate_labels

        hoi_assignment_fn = os.path.join('cache', f'precomputed_{args.ds}__hoi_pairs_{split}.h5')
        hoi_pairs_file = h5py.File(hoi_assignment_fn, 'w')
        hoi_pairs_file.create_dataset('ho_infos', data=all_ho_infos.astype(np.int))  # [fname_id, hum_idx, obj_idx, is_obj_human]
        if all_labels:
            assert all([x is not None for x in all_labels])
            all_labels = np.concatenate([x for x in all_labels], axis=0)
            hoi_pairs_file.create_dataset('labels', data=all_labels)
            assert all_labels.shape[0] == all_ho_infos.shape[0], (all_labels.shape[0], all_ho_infos.shape[0])

            assert all([x is not None for x in all_obj_labels])
            all_obj_labels = np.concatenate([x for x in all_obj_labels], axis=0)
            hoi_pairs_file.create_dataset('obj_labels', data=all_obj_labels)
            assert all_obj_labels.shape[0] == all_ho_infos.shape[0], (all_obj_labels.shape[0], all_ho_infos.shape[0])

            if full_dataset.labels_are_actions:  # labels are actions
                negatives = (all_labels[:, 0] > 0)
                positives = np.any(all_labels[:, 1:] > 0, axis=1)
            else:
                null_interactions = (hds.full_dataset.interactions[:, 0] == 0)
                negatives = np.any(all_labels[:, null_interactions] > 0, axis=1)
                positives = np.any(all_labels[:, ~null_interactions] > 0, axis=1)
            assert np.all(negatives ^ positives)
            print(f'Positives: {np.sum(positives)}')
            print(f'Negatives: {np.sum(negatives)}')
            print(f'Negatives per positive: {np.sum(negatives) / np.sum(positives)}')

        if all_pstate_labels:
            assert all([x is not None for x in all_pstate_labels])
            all_pstate_labels = np.concatenate([x for x in all_pstate_labels], axis=0)
            hoi_pairs_file.create_dataset('pstate_labels', data=all_pstate_labels)
            assert all_pstate_labels.shape[0] == all_ho_infos.shape[0], (all_pstate_labels.shape[0], all_ho_infos.shape[0])
        hoi_pairs_file.close()
        print(f'{split.capitalize()} feat file closed.')
        print('#' * 100, '\n\n')


if __name__ == '__main__':
    main()

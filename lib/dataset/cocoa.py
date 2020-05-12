import json
import os
from typing import Dict, List

import numpy as np

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset, GTImgData
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, HoiInstancesFeatProvider
from lib.dataset.utils import Dims


class CocoaSplit(HoiDatasetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: Cocoa

    @classmethod
    def instantiate_full_dataset(cls):
        return Cocoa()

    def _init_feat_provider(self, **kwargs):
        return HoiInstancesFeatProvider(ds=self, ds_name='cocoa', obj_mapping=np.arange(self.full_dataset.num_objects), **kwargs)

    @property
    def dims(self) -> Dims:
        dims = super().dims
        dims = dims._replace(P=1, M=1)  # each example is an interaction, so 1 person and 1 object
        if self._feat_provider is not None:
            B = self._feat_provider.hake_kp_boxes.shape[1]
            F_img = self._feat_provider.pc_img_feats.shape[1]
            F_kp = self._feat_provider.kp_net_dim
            F_obj = self._feat_provider.obj_feats_dim
            F_ex = self._feat_provider.ex_feat_dim
            dims = dims._replace(B=B, F_img=F_img, F_ex=F_ex, F_kp=F_kp, F_obj=F_obj)
        return dims


class Cocoa(HoiDataset):
    def __init__(self):
        driver = CocoaDriver()  # type: CocoaDriver

        object_classes = driver.objects

        # Extend actions, creating one for every role
        null_action = driver.null_interaction
        action_classes = driver.actions

        interactions = set()
        for img_anns in driver.hoi_annotations.values():
            for im_ann in img_anns:
                obj_id = im_ann['object_id']
                if obj_id < 0:
                    continue
                for a in im_ann['actions']:
                    interactions.add((a, driver.object_annotations[obj_id]['obj']))
        for j in range(len(object_classes)):
            interactions.add((0, j))  # make sure it is possible NOT to interact with any object
        interactions = np.unique(np.array(sorted(interactions)), axis=0)

        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action, interactions=interactions)

        self._split_gt_data = {'train': self.compute_img_data(img_ids=np.loadtxt(os.path.join(driver.data_dir, 'peyre', 'trainval.ids'), dtype=int),
                                                              driver=driver),
                               'test': self.compute_img_data(img_ids=np.loadtxt(os.path.join(driver.data_dir, 'peyre', 'test.ids'), dtype=int),
                                                             driver=driver),
                               }  # type: Dict[str, List[GTImgData]]
        self._img_dir = os.path.join(driver.coco_dir, 'images')

    def get_img_data(self, split):
        return self._split_gt_data[split]

    def get_img_path(self, split, fname):
        split = fname.split('_')[-2]
        assert split in ['train2014', 'val2014', 'test2014'], split
        return os.path.join(self._img_dir, split, fname)

    @property
    def labels_are_actions(self) -> bool:
        return True

    def compute_img_data(self, img_ids, driver) -> List[GTImgData]:
        driver = driver  # type: CocoaDriver
        hoi_annotations = {k: driver.hoi_annotations[k] for k in img_ids}

        split_data = []
        for img_id, img_anns in hoi_annotations.items():
            im_boxes = []
            im_box_classes = []
            im_ho_pairs = []
            im_actions = []
            box_ids_to_idx = {}
            for ann in img_anns:
                hum_id = ann['subject_id']
                hum_box = driver.object_annotations[hum_id]['bbox']
                hum_class = driver.object_annotations[hum_id]['obj']
                assert hum_class == self.human_class
                if hum_id not in box_ids_to_idx:
                    box_ids_to_idx[hum_id] = len(im_boxes)
                    im_boxes.append(hum_box)
                    im_box_classes.append(hum_class)
                hum_idx_in_img = box_ids_to_idx[hum_id]

                obj_id = ann['object_id']
                if obj_id > 0:
                    obj_box = driver.object_annotations[obj_id]['bbox']
                    obj_class = driver.object_annotations[obj_id]['obj']
                    if obj_id not in box_ids_to_idx:
                        box_ids_to_idx[obj_id] = len(im_boxes)
                        im_boxes.append(obj_box)
                        im_box_classes.append(obj_class)
                    obj_idx_in_img = box_ids_to_idx[obj_id]
                else:
                    obj_idx_in_img = np.nan

                for a in ann['actions']:
                    im_ho_pairs.append(np.array([hum_idx_in_img, obj_idx_in_img]))
                    im_actions.append(a)

            if not im_ho_pairs:
                continue

            im_boxes = np.stack(im_boxes, axis=0)
            im_box_classes = np.array(im_box_classes)
            im_ho_pairs = np.stack(im_ho_pairs, axis=0)
            im_actions = np.array(im_actions)

            split_data.append(GTImgData(filename=driver.image_infos[img_id]['file_name'],
                                        img_size=np.array([driver.image_infos[img_id]['width'], driver.image_infos[img_id]['height']]),
                                        boxes=im_boxes, box_classes=im_box_classes, ho_pairs=im_ho_pairs, labels=im_actions))
        return split_data


class CocoaDriver:
    def __init__(self):
        # TODO move these
        self.data_dir = os.path.join(cfg.data_root, 'COCO-A')
        self.coco_dir = os.path.join(self.data_dir, 'coco')
        self.null_interaction = '__no_interaction__'

        # Annotations
        annotations_per_image, image_infos, actions, objects, object_annotations = self.load_annotations()
        self.image_infos = image_infos
        self.object_annotations, self.objects = object_annotations, objects
        self.hoi_annotations, self.actions = annotations_per_image, actions

    def load_annotations(self):
        # This is adapted from the official COCO-A API. #TODO link

        # COCO annotations # TODO merge with VCOCO and move
        with open(os.path.join(self.coco_dir, 'annotations', 'instances_train2014.json'), 'r') as f:
            bbox_data = json.load(f)
        object_annotations = bbox_data['annotations']
        for oann in object_annotations:
            x, y, w, h = oann['bbox']
            oann['bbox'] = np.array([x, y, x + w, y + h])  # [x, y, w, h] -> [x1, y1, x2, y2]
        image_infos = bbox_data['images']
        object_class_data = bbox_data['categories']
        obj_ids_to_idx = {x['id']: i for i, x in enumerate(object_class_data)}
        objects = [x['name'] for x in object_class_data]

        # Filter out 'license', 'coco_url', 'date_captured', 'flickr_url'.
        image_infos = [{k: iinfo[k] for k in ['id', 'file_name', 'height', 'width']} for iinfo in image_infos]
        image_infos = {x['id']: {k: v for k, v in x.items() if k != 'id'} for x in image_infos}

        # Filter out 'segmentation', 'area', 'iscrowd', 'flickr_url'.
        object_annotations = [{k: ann[k] for k in ['image_id', 'bbox', 'category_id', 'id']} for ann in object_annotations]
        object_annotations = {x['id']: {**{k: v for k, v in x.items() if k not in ['id', 'category_id']},
                                        **{'obj': obj_ids_to_idx[x['category_id']]}}
                              for x in object_annotations}

        # Visual Verbnet. Each visual action is a dictionary with the following properties:
        #  - id:            unique id within VVN
        #  - name:          name of the visual action
        #  - category:      visual category as defined in the paper
        #  - definition:    [empty]
        #                   an english language description of the visual action
        #  - verbnet_class: [empty]
        #                   corresponding verbnet (http://verbs.colorado.edu/verb-index/index.php) entry id for each visual action
        # Each visual adverb is a dictionary with the following properties:
        #  - id:            unique id within VVN
        #  - name:          name of the visual action
        #  - category:      visual category as defined in the paper
        #  - definition:    [empty]
        #                   an english language description of the visual action
        with open(os.path.join(self.data_dir, 'visual_verbnet_beta2015.json')) as f:
            vvn = json.load(f)
        visual_actions = vvn['visual_actions']  # list of 145 visual actions contained in VVN
        visual_adverbs = vvn['visual_adverbs']  # list of 17 visual adverbs contained in VVN
        actions = [self.null_interaction]
        acts_ids_to_idx = {}
        for x in visual_actions:
            act_id = x['id']
            assert act_id not in acts_ids_to_idx
            acts_ids_to_idx[act_id] = len(actions)
            actions.append(x['name'])
        assert len(actions) == len(visual_actions) + 1

        # COCO-A annotations. Each annotation in cocoa is a dictionary with the following properties:
        #  - id:             unique id within coco-a
        #  - image_id:       unique id of the image from the MS COCO dataset
        #  - object_id:      unique id of the object from the MS COCO dataset
        #  - subject_id:     unique id of the subject from the MS COCO dataset
        #  - visual_actions: list of visual action ids performed by the subject (with the object if present)
        #  - visual_adverbs: list of visual adverb ids describing the subject (and object interaction if present)
        with open(os.path.join(self.data_dir, 'cocoa_beta2015.json')) as f:
            cocoa = json.load(f)
        raw_annotations = [ann for num_annotators_in_agreement in [1, 2, 3] for ann in cocoa['annotations'][str(num_annotators_in_agreement)]]

        # Processing. The output is a dictionary where keys are image IDs and values are a list of dictionaries with the following entries:
        #  - subject_id:    unique ID of the subject from the MS COCO dataset
        #  - object_id:     unique ID of the object from the MS COCO dataset, or -1 if the actions require no object
        #  - actions:       list of action IDs (i.e., indices over `actions`) performed by the subject (with the object if present)
        annotations_per_image = {}
        for ann in raw_annotations:
            new_ann = {'subject_id': ann['subject_id'],
                       'object_id': ann['object_id'],
                       'actions': [acts_ids_to_idx[a] for a in ann['visual_actions']],
                       }
            annotations_per_image.setdefault(ann['image_id'], []).append(new_ann)

        # Filter images without annotations
        imgs_with_anns = set(annotations_per_image.keys())
        assert not (imgs_with_anns - set(image_infos.keys()))
        image_infos = {k: v for k, v in image_infos.items() if k in imgs_with_anns}

        return annotations_per_image, image_infos, actions, objects, object_annotations


if __name__ == '__main__':
    # d = CocoaDriver()
    c = Cocoa()
    print('Done.')

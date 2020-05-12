import json
import os
from typing import Dict, List

import numpy as np

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset, GTImgData
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, HoiInstancesFeatProvider
from lib.dataset.utils import Dims


class VCocoSplit(HoiDatasetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: VCoco

    @classmethod
    def instantiate_full_dataset(cls):
        return VCoco()

    def _init_feat_provider(self, **kwargs):
        return HoiInstancesFeatProvider(ds=self, ds_name='vcoco', obj_mapping=np.arange(self.full_dataset.num_objects), **kwargs)

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


class VCoco(HoiDataset):
    def __init__(self):
        driver = VCocoDriver()  # type: VCocoDriver

        object_classes = driver.objects

        # Extend actions, creating one for every role
        null_action = driver.null_interaction
        action_classes = [null_action] + \
                         [f'{x["action_name"]}_{r}' for x in driver.interaction_class_data[1:] for r in (x['role_name'][1:] or ['agent'])]
        action_index = {x: i for i, x in enumerate(action_classes)}

        interactions = set()
        for s in ['train', 'test']:
            for img_anns in driver.hoi_annotations_per_split[s].values():
                for im_ann in img_anns:
                    a = im_ann['action']
                    act_name = driver.actions[a]  # the original actions have to be used here!
                    for r, obj_id in enumerate(im_ann['role_object_id']):
                        if obj_id > 0:
                            interactions.add((action_index[f'{act_name}_{driver.interaction_class_data[a]["role_name"][1 + r]}'],
                                              driver.object_annotations[obj_id]['obj']))
        for j in range(len(object_classes)):
            interactions.add((0, j))  # make sure it is possible NOT to interact with any object
        interactions = np.unique(np.array(sorted(interactions)), axis=0)

        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action, interactions=interactions)

        self._split_gt_data = {'train': self.compute_img_data(split='train', driver=driver),
                               'test': self.compute_img_data(split='test', driver=driver),
                               }  # type: Dict[str, List[GTImgData]]
        self._img_dir = driver.img_dir

    def get_img_data(self, split):
        return self._split_gt_data[split]

    def get_img_path(self, split, fname):
        split = fname.split('_')[-2]
        assert split in ['train2014', 'val2014', 'test2014'], split
        return os.path.join(self._img_dir, split, fname)

    @property
    def labels_are_actions(self) -> bool:
        return True

    def compute_img_data(self, split, driver) -> List[GTImgData]:
        driver = driver  # type: VCocoDriver
        hoi_annotations = driver.hoi_annotations_per_split[split]
        image_ids = list(hoi_annotations.keys())
        assert sorted(image_ids) == image_ids

        split_data = []
        for img_id, img_anns in hoi_annotations.items():
            im_boxes = []
            im_box_classes = []
            im_ho_pairs = []
            im_actions = []
            box_ids_to_idx = {}
            for ann in img_anns:
                hum_id = ann['ann_id']
                hum_box = driver.object_annotations[hum_id]['bbox']
                hum_class = driver.object_annotations[hum_id]['obj']
                assert hum_class == self.human_class
                if hum_id not in box_ids_to_idx:
                    box_ids_to_idx[hum_id] = len(im_boxes)
                    im_boxes.append(hum_box)
                    im_box_classes.append(hum_class)
                hum_idx_in_img = box_ids_to_idx[hum_id]

                orig_act_id = ann['action']

                role_names = (driver.interaction_class_data[orig_act_id]['role_name'][1:] or ['agent'])  # only consider 'agent' if no objects
                role_ids = ann['role_object_id'] or [0]  # object IDs, or 0 for actions with no objects
                assert len(role_names) == len(role_ids)

                # if two roles are specified but one of the objects is 0, filter it out
                if len(role_ids) > 1:
                    assert len(role_ids) == 2
                    new_role_ids, new_role_names = [], []
                    for rid, rname in zip(role_ids, role_names):
                        if rid > 0:
                            new_role_ids.append(rid)
                            new_role_names.append(rname)
                    if not new_role_ids:  # both are 0
                        assert not new_role_names
                        new_role_ids, new_role_names = [0], ['obj']
                    role_ids, role_names = new_role_ids, new_role_names
                    assert (role_ids and role_names) and (len(role_ids) == len(role_names))

                for role_name, obj_id in zip(role_names, role_ids):
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

                    actual_act_id = self.action_index[f'{driver.actions[orig_act_id]}_{role_name}']
                    im_ho_pairs.append(np.array([hum_idx_in_img, obj_idx_in_img]))
                    im_actions.append(actual_act_id)

            im_boxes = np.stack(im_boxes, axis=0)
            im_box_classes = np.array(im_box_classes)
            im_ho_pairs = np.stack(im_ho_pairs, axis=0)
            im_actions = np.array(im_actions)

            split_data.append(GTImgData(filename=driver.image_infos[img_id]['file_name'],
                                        img_size=np.array([driver.image_infos[img_id]['width'], driver.image_infos[img_id]['height']]),
                                        boxes=im_boxes, box_classes=im_box_classes, ho_pairs=im_ho_pairs, labels=im_actions))
        return split_data


class VCocoDriver:
    def __init__(self):
        self.data_dir = os.path.join(cfg.data_root, 'V-COCO')
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.null_interaction = '__no_interaction__'

        # Annotations
        self.image_infos, object_data, hoi_data, extra_data = self.load_annotations()
        self.object_annotations, self.objects = object_data
        self.hoi_annotations_per_split, self.actions = hoi_data
        self.object_class_data, self.interaction_class_data, self.all_hoi_data_per_split = extra_data

    def load_annotations(self):
        with open(os.path.join(self.data_dir, 'instances_vcoco_all_2014.json'), 'r') as f:
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

        all_hoi_data = {}
        with open(os.path.join(self.data_dir, 'vcoco', 'vcoco_train.json'), 'r') as f:
            all_hoi_data['train'] = json.load(f)
        with open(os.path.join(self.data_dir, 'vcoco', 'vcoco_val.json'), 'r') as f:
            all_hoi_data['val'] = json.load(f)
        with open(os.path.join(self.data_dir, 'vcoco', 'vcoco_test.json'), 'r') as f:
            all_hoi_data['test'] = json.load(f)

        interaction_class_data = []
        num_categories = len(all_hoi_data['train'])
        assert num_categories == len(all_hoi_data['val']) == len(all_hoi_data['test'])
        for i in range(num_categories):
            class_data = {}
            for k in ['action_name', 'role_name', 'include']:
                assert all_hoi_data['train'][i][k] == all_hoi_data['val'][i][k] == all_hoi_data['test'][i][k]
                class_data[k] = all_hoi_data['train'][i][k]
            interaction_class_data.append(class_data)
        interaction_class_data = [{'action_name': self.null_interaction, 'role_name': ['agent'], 'include': [[]]}] + interaction_class_data
        actions = [x['action_name'] for x in interaction_class_data]

        hoi_annotations = {'train': {}, 'test': {}}  # collapse val in train
        for split in ['train', 'val', 'test']:
            split_data = {}
            for i in range(num_categories):
                num_roles = len(interaction_class_data[i + 1]['role_name'])

                im_ids = all_hoi_data[split][i]['image_id']
                ann_ids = all_hoi_data[split][i]['ann_id']
                role_object_ids = all_hoi_data[split][i]['role_object_id']
                labels = all_hoi_data[split][i]['label']

                num_imgs = len(im_ids)
                assert num_imgs == len(ann_ids) == len(labels) == len(role_object_ids) // num_roles
                for j in range(num_imgs):
                    im_id_j = im_ids[j]
                    ann_id_j = ann_ids[j]
                    roles_j = [role_object_ids[k * num_imgs + j] for k in range(num_roles)]
                    label_j = labels[j]
                    assert ann_id_j == roles_j[0]  # the first role is always a person
                    roles_j = roles_j[1:]
                    if label_j == 1:
                        split_data.setdefault(im_id_j, []).append({'ann_id': ann_id_j,
                                                                   'role_object_id': roles_j,
                                                                   # 'label': label_j,
                                                                   'action': i + 1,
                                                                   })
            hsplit = 'test' if split == 'test' else 'train'
            assert not set(hoi_annotations[hsplit].keys()) & set(split_data.keys())
            hoi_annotations[hsplit].update(split_data)
        hoi_annotations = {k: {k1: v[k1] for k1 in sorted(v.keys())} for k, v in hoi_annotations.items()}

        return image_infos, [object_annotations, objects], [hoi_annotations, actions], [object_class_data, interaction_class_data, all_hoi_data]


if __name__ == '__main__':
    h = VCoco()
    print('Done.')

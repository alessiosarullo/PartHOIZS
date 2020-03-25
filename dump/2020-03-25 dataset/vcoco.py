import json
import os
from typing import Dict
import numpy as np

from config import cfg

from lib.dataset.hoi_dataset import HoiDataset
from lib.dataset.hoi_dataset_split import HoiDatasetSplit
from lib.dataset.utils import Splits, HoiTripletsData


class VCocoSplit(HoiDatasetSplit):
    def __init__(self, split, full_dataset, object_inds=None, action_inds=None):
        super().__init__(split, full_dataset, object_inds, action_inds)
        self.full_dataset = self.full_dataset  # type: VCoco
        self.img_dims = self.full_dataset.split_img_dims[self.split]
        self.fnames = self.full_dataset.split_filenames[self.split]

    def _get_precomputed_feats_fn(self, split):
        raise ValueError
        return cfg.precomputed_feats_format % ('hico', 'resnet152', split.value)

    @classmethod
    def instantiate_full_dataset(cls) -> HoiDataset:
        return VCoco()


class VCoco(HoiDataset):
    def __init__(self, add_detection_annotations=None):
        assert add_detection_annotations is None or add_detection_annotations is True

        driver = VCocoDriver()  # type: VCocoDriver
        self._split_det_data = {Splits.TRAIN: self.compute_annotations(split=Splits.TRAIN, driver=driver),
                                Splits.TEST: self.compute_annotations(split=Splits.TEST, driver=driver),
                                }  # type: Dict[Splits: HoiTripletsData]
        self._split_filenames = {Splits.TRAIN: [driver.image_infos[fid]['file_name'] for fid in driver.hoi_annotations_per_split['train'].keys()],
                                 Splits.TEST: [driver.image_infos[fid]['file_name'] for fid in driver.hoi_annotations_per_split['test'].keys()],
                                 }
        self.split_img_dims = {Splits.TRAIN: [(driver.image_infos[fid]['width'], driver.image_infos[fid]['height'])
                                              for fid in driver.hoi_annotations_per_split['train'].keys()],
                               Splits.TEST: [(driver.image_infos[fid]['width'], driver.image_infos[fid]['height'])
                                              for fid in driver.hoi_annotations_per_split['test'].keys()]}
        self._img_dir = driver.img_dir

        object_classes = driver.objects
        action_classes = driver.actions
        null_action = driver.null_interaction
        interactions = []
        for k, v in self._split_det_data.items():
            hoi_inds, acts = np.where(v.labels)
            obj_inds = v.ho_pairs[hoi_inds, 2]
            actual_interactions_mask = (obj_inds >= 0)
            acts = acts[actual_interactions_mask]
            obj_inds = obj_inds[actual_interactions_mask]
            assert np.all(obj_inds >= 0)
            objs = v.boxes[obj_inds]
            interactions.append(np.stack([acts, objs], axis=1))
        interactions = np.unique(np.concatenate(interactions, axis=0))

        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action, interactions=interactions)

    @property
    def split_filenames(self):
        return self._split_filenames

    @property
    def split_img_labels(self):
        raise ValueError('No image labels for V-COCO.')

    @property
    def split_hoi_triplets_data(self):
        return self._split_det_data

    def get_img_path(self, split, fname):
        split = fname.split('_')[2]
        assert split in ['train2014', 'val2014', 'test2014']
        return os.path.join(self._img_dir, split, fname)

    @staticmethod
    def compute_annotations(split, driver) -> HoiTripletsData:
        hoi_annotations = driver.hoi_annotations_per_split[split.value]
        image_ids = list(hoi_annotations.keys())
        assert sorted(image_ids) == image_ids
        img_id_to_idx = {imid: i for i, imid in enumerate(image_ids)}
        image_ids = set(image_ids)

        obj_id_to_idx = {}
        all_obj_annotations = driver.object_annotations
        boxes = np.zeros((len(all_obj_annotations), 6))  # each is (image_idx, x1, y1, x2, y2, class)
        for i, (obj_id, obj_ann) in enumerate(all_obj_annotations.items()):
            if obj_ann['image_id'] in image_ids:  # belongs to this split
                assert obj_id not in obj_id_to_idx
                obj_id_to_idx[obj_id] = i
                boxes[i] = [img_id_to_idx[obj_ann['image_id']],
                            *obj_ann['bbox'],
                            obj_ann['obj']
                            ]

        ho_pairs = np.zeros((len(hoi_annotations), 3))  # each is (image_idx, hum_idx, obj_idx)
        labels = np.zeros((len(hoi_annotations), len(driver.actions)))  # FIXME actions, not interactions
        for i, (imid, ann) in enumerate(hoi_annotations.items()):
            act_id = ann['action']
            num_roles = len(driver.interaction_class_data[act_id]['role_name'])
            if num_roles == 1:
                obj_idx = np.nan
            elif num_roles == 2:
                obj_idx = obj_id_to_idx[ann['role_object_id'][0]]
            elif num_roles == 3:
                # TODO? Handle  {'action_name': 'cut', 'role_name': ['agent', 'instr', 'obj']}, making it cut_with? What about similar actions?
                role_id = driver.interaction_class_data[act_id]['role_name'].index('obj') - 1
                obj_idx = obj_id_to_idx[ann['role_object_id'][role_id]]
            else:
                raise ValueError(f'Too many roles: {num_roles}')
            ho_pairs[i] = [img_id_to_idx[imid],
                           obj_id_to_idx[ann['ann_id']],
                           obj_idx
                           ]
            labels[i, act_id] = 1
        return HoiTripletsData(boxes=boxes, ho_pairs=ho_pairs, labels=labels)


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

        action_class_data = []
        num_categories = len(all_hoi_data['train'])
        assert num_categories == len(all_hoi_data['val']) == len(all_hoi_data['test'])
        for i in range(num_categories):
            class_data = {}
            for k in ['action_name', 'role_name', 'include']:
                assert all_hoi_data['train'][i][k] == all_hoi_data['val'][i][k] == all_hoi_data['test'][i][k]
                class_data[k] = all_hoi_data['train'][i][k]
            action_class_data.append(class_data)
        actions = [x['action_name'] for x in action_class_data]

        hoi_annotations = {'train': {}, 'test': {}}  # collapse val in train
        for split in ['train', 'val', 'test']:
            split_data = {}
            for i in range(num_categories):
                num_roles = len(action_class_data[i]['role_name'])

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
                                                                   'action': i,
                                                                   })
            hsplit = 'test' if split == 'test' else 'train'
            assert not set(hoi_annotations[hsplit].keys()) & set(split_data.keys())
            hoi_annotations[hsplit].update(split_data)
        hoi_annotations = {k: {k1: v[k1] for k1 in sorted(v.keys())} for k, v in hoi_annotations.items()}

        return image_infos, [object_annotations, objects], [hoi_annotations, actions], [object_class_data, action_class_data, all_hoi_data]


if __name__ == '__main__':
    h = VCocoDriver()
    print('Done.')

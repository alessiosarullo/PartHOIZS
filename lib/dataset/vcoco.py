import json
import os
from typing import Dict

import numpy as np
import torch

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, HoiInstancesFeatProvider
from lib.dataset.utils import HoiTripletsData, Dims
from lib.timer import Timer


class VCocoSplit(HoiDatasetSplit):
    def __init__(self, split, full_dataset, object_inds=None, action_inds=None):
        super().__init__(split, full_dataset, object_inds, action_inds, labels_are_actions=True)
        self.full_dataset = self.full_dataset  # type: VCoco

    @classmethod
    def instantiate_full_dataset(cls):
        return VCoco()

    def _collate(self, idx_list, device):
        raise NotImplementedError


class VCocoKPSplit(VCocoSplit):
    def __init__(self, split, full_dataset, object_inds=None, action_inds=None, no_feats=False):
        super().__init__(split, full_dataset, object_inds, action_inds)
        self._feat_provider = HoiInstancesFeatProvider(ds=self, ds_name='vcoco', no_feats=no_feats)
        self.non_empty_inds = None  # not used, and it would be determined by the precomputed HOI assignment anyway.

    @property
    def dims(self) -> Dims:
        K_hake = self._feat_provider.hake_kp_boxes.shape[1]
        F_img = self._feat_provider.pc_img_feats.shape[1]
        F_kp = self._feat_provider.kp_net_dim
        F_obj = self._feat_provider.obj_feats_dim

        # each example is an interaction, so 1 person and 1 object
        return super().dims._replace(P=1, M=1, K_hake=K_hake, F_img=F_img, F_kp=F_kp, F_obj=F_obj)

    def hold_out(self, ratio):
        if cfg.no_filter_bg_only:
            print('!!!!!!!!!! Filtering background-only images.')
        num_examples = len(self._feat_provider.ho_infos)
        example_ids = np.arange(num_examples)
        num_examples_to_keep = num_examples - int(num_examples * ratio)
        keep_inds = np.random.choice(example_ids, size=num_examples_to_keep, replace=False)
        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(example_ids, keep_inds)

    def _collate(self, idx_list, device):
        Timer.get('GetBatch').tic()
        mb = self._feat_provider.collate(idx_list, device)
        Timer.get('GetBatch').toc(discard=5)
        return mb


class VCoco(HoiDataset):
    def __init__(self):
        driver = VCocoDriver()  # type: VCocoDriver

        object_classes = driver.objects
        action_classes = driver.actions
        null_action = driver.null_interaction
        interactions = driver.interactions
        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action, interactions=interactions)

        self._split_det_data = {'train': self.compute_annotations(split='train', driver=driver),
                                'test': self.compute_annotations(split='test', driver=driver),
                                }  # type: Dict[str: HoiTripletsData]
        self._split_filenames = {'train': [driver.image_infos[fid]['file_name'] for fid in driver.hoi_annotations_per_split['train'].keys()],
                                 'test': [driver.image_infos[fid]['file_name'] for fid in driver.hoi_annotations_per_split['test'].keys()],
                                 }
        self._split_img_dims = {'train': [(driver.image_infos[fid]['width'], driver.image_infos[fid]['height'])
                                               for fid in driver.hoi_annotations_per_split['train'].keys()],
                                'test': [(driver.image_infos[fid]['width'], driver.image_infos[fid]['height'])
                                              for fid in driver.hoi_annotations_per_split['test'].keys()]}
        self._img_dir = driver.img_dir

    @property
    def split_filenames(self):
        return self._split_filenames

    @property
    def split_img_dims(self):
        return self._split_img_dims

    @property
    def split_labels(self):
        return {'train': self._split_det_data['train'].labels,
                'test': self._split_det_data['test'].labels
                }

    def get_img_path(self, split, fname):
        split = fname.split('_')[-2]
        assert split in ['train2014', 'val2014', 'test2014'], split
        return os.path.join(self._img_dir, split, fname)

    @staticmethod
    def compute_annotations(split, driver) -> HoiTripletsData:
        hoi_annotations = driver.hoi_annotations_per_split[split.value]
        image_ids = list(hoi_annotations.keys())
        assert sorted(image_ids) == image_ids
        img_id_to_idx = {imid: i for i, imid in enumerate(image_ids)}
        image_ids_set = set(image_ids)

        obj_id_to_idx = {}
        all_obj_annotations = driver.object_annotations
        boxes = []  # each is (image_idx, x1, y1, x2, y2, class)
        for obj_id, obj_ann in all_obj_annotations.items():
            if obj_ann['image_id'] in image_ids_set:  # belongs to this split
                assert obj_id not in obj_id_to_idx
                obj_id_to_idx[obj_id] = len(boxes)
                boxes.append(np.array([img_id_to_idx[obj_ann['image_id']],
                                       *obj_ann['bbox'],
                                       obj_ann['obj']
                                       ]))
        boxes = np.stack(boxes, axis=0)

        ho_pairs = []  # each is (image_idx, hum_idx, obj_idx)
        labels = []  # actions, not interactions
        fnames = []
        for i, (imid, im_annotations) in enumerate(hoi_annotations.items()):
            for ann in im_annotations:
                act_id = ann['action']
                num_roles = len(driver.interaction_class_data[act_id]['role_name'])
                if num_roles == 1:
                    obj_id = -1
                elif num_roles == 2:
                    obj_id = ann['role_object_id'][0]
                elif num_roles == 3:
                    # TODO? Handle  {'action_name': 'cut', 'role_name': ['agent', 'instr', 'obj']}, making it cut_with? What about similar actions?
                    role_id = driver.interaction_class_data[act_id]['role_name'].index('obj') - 1
                    obj_id = ann['role_object_id'][role_id]
                else:
                    raise ValueError(f'Too many roles: {num_roles}.')
                obj_idx = obj_id_to_idx.get(obj_id, np.nan)
                ho_pairs.append(np.array([img_id_to_idx[imid],
                                          obj_id_to_idx[ann['ann_id']],
                                          obj_idx
                                          ]))
                labels.append(act_id)
                fnames.append(driver.image_infos[imid]['file_name'])
        ho_pairs = np.stack(ho_pairs, axis=0)
        # TODO? These pairs are not unique
        labels = np.array(labels)
        onehot_labels = np.zeros((labels.size, len(driver.actions)))
        onehot_labels[np.arange(labels.size), labels] = 1
        return HoiTripletsData(boxes=boxes, ho_pairs=ho_pairs, labels=onehot_labels, fnames=fnames)


class VCocoDriver:
    def __init__(self):
        self.data_dir = os.path.join(cfg.data_root, 'V-COCO')
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.null_interaction = '__no_interaction__'

        # Annotations
        self.image_infos, object_data, hoi_data, extra_data = self.load_annotations()
        self.object_annotations, self.objects = object_data
        self.hoi_annotations_per_split, self.actions, self.interactions = hoi_data
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

        interactions = sorted({(im_ann['action'], object_annotations[obj_id]['obj'])
                               for s in ['train', 'test']
                               for img_anns in hoi_annotations[s].values()
                               for im_ann in img_anns
                               for obj_id in (im_ann['role_object_id'][-1:] if actions[im_ann['action']] != 'eat'
                                              else im_ann['role_object_id'][:1])
                               if obj_id > 0})
        interactions = np.unique(np.array(interactions), axis=0)

        return image_infos, [object_annotations, objects], [hoi_annotations, actions, interactions], \
               [object_class_data, interaction_class_data, all_hoi_data]


if __name__ == '__main__':
    h = VCoco()
    print('Done.')

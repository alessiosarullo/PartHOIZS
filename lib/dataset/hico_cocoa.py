from typing import Dict, List

import numpy as np

from lib.dataset.cocoa import Cocoa
from lib.dataset.hicodet_hake import HicoDetHake
from lib.dataset.hoi_dataset import HoiDataset, GTImgData
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, HoiInstancesFeatProvider
from lib.dataset.utils import Dims, get_obj_mapping


class HicoCocoaSplit(HoiDatasetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: HicoCocoa

    @classmethod
    def instantiate_full_dataset(cls, **kwargs):
        return HicoCocoa()

    def _init_feat_provider(self, **kwargs):
        if self.split == 'train':
            return HoiInstancesFeatProvider(ds=self, ds_name='hico', obj_mapping=np.arange(self.full_dataset.num_objects),
                                            label_mapping=self.full_dataset.hico_interaction_mapping,
                                            **kwargs)
        else:
            assert self.split == 'test'
            return HoiInstancesFeatProvider(ds=self, ds_name='cocoaall', obj_mapping=np.arange(self.full_dataset.num_objects),
                                            label_mapping=self.full_dataset.cocoa_interaction_mapping,
                                            **kwargs)

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


class HicoCocoa(HoiDataset):
    def __init__(self):

        hico = HicoDetHake()
        cocoa = Cocoa(all_as_test=True, fix_solos=False)

        hico_objects = hico.objects
        cocoa_objects = [o.replace(' ', '_') for o in cocoa.objects]
        assert hico_objects == sorted(cocoa_objects)
        object_classes = cocoa_objects  # object classes are in the same order as COCO
        hico_to_coco_obj_mapping = get_obj_mapping(hico_objects)

        hico_actions = hico.actions
        cocoa_actions = [a.replace(' ', '_') for a in cocoa.actions]
        action_classes = sorted(set(hico_actions) | set(cocoa_actions))

        hico_interactions_str_pair = [(hico_actions[a], hico_objects[o]) for a, o in hico.interactions]
        cocoa_interactions_str_pair = [(cocoa_actions[a], cocoa_objects[o]) for a, o in cocoa.interactions]
        interactions_classes = sorted(set(hico_interactions_str_pair + cocoa_interactions_str_pair))

        null_action = hico_actions[0]
        assert null_action == cocoa_actions[0]

        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action,
                         interactions_classes=interactions_classes)
        interaction_index = {c: i for i, c in enumerate(interactions_classes)}
        hico_mapping = np.array([interaction_index[c] for c in hico_interactions_str_pair])  # hico_i -> new_i (len = #interactions in HICO)
        cocoa_mapping = np.array([interaction_index[c] for c in cocoa_interactions_str_pair])  # same

        self._split_gt_data = {'train': [x._replace(labels=hico_mapping[x.labels], box_classes=hico_to_coco_obj_mapping[x.box_classes])
                                         for x in hico.get_img_data('train')]
                               }  # type: Dict[str, List[GTImgData]]
        cocoa_data = []
        for x in cocoa.get_img_data('test'):
            valid_ho_pairs = ~np.isnan(x.ho_pairs).any(axis=1)
            if ~valid_ho_pairs.any():
                ho_pairs = hoi_labels = None
            else:
                ho_pairs = x.ho_pairs[valid_ho_pairs]
                assert ~np.isnan(ho_pairs).any() and np.all(ho_pairs >= 0)
                ho_pairs = ho_pairs.astype(np.int)

                act_labels = x.labels[valid_ho_pairs]
                obj_label_per_pair = x.box_classes[ho_pairs[:, 1]]
                cocoa_hoi_labels = cocoa.oa_to_interaction[obj_label_per_pair, act_labels]
                assert np.all(cocoa_hoi_labels >= 0)
                hoi_labels = cocoa_mapping[cocoa_hoi_labels]

            assert x.ps_labels is None
            img_data = x._replace(ho_pairs=ho_pairs, labels=hoi_labels)
            cocoa_data.append(img_data)
        self._split_gt_data['test'] = cocoa_data

        self.hico = hico
        self.cocoa = cocoa
        self.hico_interaction_mapping = hico_mapping
        self.cocoa_interaction_mapping = cocoa_mapping

    def get_img_data(self, split) -> List[GTImgData]:
        return self._split_gt_data[split]

    def get_img_path(self, split, fname):
        if split == 'train':
            return self.hico.get_img_path(split, fname)
        else:
            assert split == 'test'
            return self.cocoa.get_img_path(split, fname)

    @property
    def labels_are_actions(self) -> bool:
        return False


if __name__ == '__main__':
    h = HicoCocoa()
    print('Done.')

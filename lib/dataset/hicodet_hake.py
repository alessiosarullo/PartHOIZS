import json
import os
import pickle
from typing import Dict, List

import numpy as np
from scipy.io import loadmat

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset, GTImgData
from lib.dataset.hoi_dataset_split import HoiDatasetSplit, HoiInstancesFeatProvider
from lib.dataset.utils import Dims, get_hico_to_coco_mapping


class HicoDetHakeSplit(HoiDatasetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_dataset = self.full_dataset  # type: HicoDetHake

    @classmethod
    def instantiate_full_dataset(cls):
        return HicoDetHake()

    def _init_feat_provider(self, **kwargs):
        return HoiInstancesFeatProvider(ds=self, ds_name='hico', obj_mapping=get_hico_to_coco_mapping(self.full_dataset.objects), **kwargs)

    @property
    def dims(self) -> Dims:
        dims = super().dims
        dims = dims._replace(B=self.full_dataset.num_parts, S=self.full_dataset.num_states)
        if self._feat_provider is not None:
            F_img = self._feat_provider.pc_img_feats.shape[1]
            F_kp = self._feat_provider.kp_net_dim
            F_obj = self._feat_provider.obj_feats_dim
            dims = dims._replace(F_img=F_img, F_kp=F_kp, F_obj=F_obj)
        return dims


class HicoDetHake(HoiDataset):
    def __init__(self):
        """
        In the following, BP = body parts and PS = (body) part state.
        Attributes:
            - parts: List[str]. Names of HAKE's 10 parts.
            - states: List[str]. All body part states. Note that states are duplicated for right/left body parts:
                            both "right_hand hold" and "left_hand hold" are included.
            - states_per_part: List[np.array]. Associates an array of indices in `states` to each part (same order as `parts`).
        """
        driver = HicoDetDriver()  # type: HicoDetDriver

        object_classes = sorted(set([inter['obj'] for inter in driver.interaction_list]))
        action_classes = list(driver.action_dict.keys())
        null_action = driver.null_interaction
        interactions_classes = [[inter['act'], inter['obj']] for inter in driver.interaction_list]

        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action,
                         interactions_classes=interactions_classes)

        with open(os.path.join(cfg.data_root, 'HICO-DET', 'HAKE', 'joints.txt'), 'r') as f:
            self.parts = [l.strip().lower() for l in f.readlines()]  # type: List[str]

        with open(os.path.join(cfg.data_root, 'HICO-DET', 'HAKE', 'Part_State_76.txt'), 'r') as f:
            part_state_dict_str = {}
            part_state_pairs = []
            for i, l in enumerate(f.readlines()):
                if not l.strip():
                    break
                part_state_pairs = [x.strip().lower() for x in l.strip().split(':')]
                body_part, part_state = part_state_pairs
                if part_state == 'no_interaction':
                    part_state = self.null_action
                part_state = part_state.replace(' ', '_')
                part_state_dict_str.setdefault(body_part, []).append(part_state)
                part_state_pairs.append([body_part, part_state])
        self._part_state_pairs = part_state_pairs  # type: List[List[str]]  # This is essentially the content of file `Part_State_76.txt`.
        self.states = []  # type: List[str]
        self.states_per_part = []  # type: List[np.array]
        for p in self.parts:
            part_states = part_state_dict_str[p.split("_")[-1]]
            self.states_per_part.append(len(self.states) + np.arange(len(part_states)))
            for s in part_states:
                self.states.append(f'{p} {s}')
        assert np.all(np.concatenate(self.states_per_part) == np.arange(self.num_states))

        img_data_per_split = {s: self.compute_img_data(split=s, driver=driver) for s in ['train', 'test']}  # type: Dict[str, List[GTImgData]]
        hake_img_data = {s: self.compute_hake_img_data(split=s) for s in ['train']}  # type: Dict[str, List[GTImgData]]

        # Sanity check + add pstate info to HICO-DET annotations (which are more exhaustive than HICO-DET_HAKE's ones)
        self._split_img_dir = driver.split_img_dir
        new_img_data_per_split = {'test': img_data_per_split['test']}  # type: Dict[str, List[GTImgData]]
        for split in ['train']:
            new_img_data_per_split[split] = []
            for data, hdata in zip(img_data_per_split[split], hake_img_data[split]):
                new_data = data

                assert data.filename == hdata.filename
                if hdata.img_size is None:  # image should be empty
                    assert all([d is None for d in data[2:]])
                    assert all([d is None for d in hdata[2:]])
                else:
                    assert np.all(data.img_size == hdata.img_size)

                    # New pstate labels
                    new_ps_labels = np.zeros((data.labels.shape[0], self.num_states))

                    # HAKE annotations are missing some null interactions, so they are excluded from the comparison.
                    d_fg_interactions = np.flatnonzero(self.interactions[np.where(data.labels)[1], 0] != 0)
                    h_fg_interactions = np.flatnonzero(self.interactions[np.where(hdata.labels)[1], 0] != 0)
                    assert d_fg_interactions.size == h_fg_interactions.size

                    if d_fg_interactions.size > 0:
                        d_box_pairs = np.concatenate([data.boxes[data.ho_pairs[d_fg_interactions, 0], :],
                                                      data.boxes[data.ho_pairs[d_fg_interactions, 1], :]
                                                      ], axis=1)
                        d_idxs = np.lexsort(d_box_pairs.T[::-1, :])
                        assert d_idxs.size == d_fg_interactions.size  # make sure it just sorted and didn't remove anything

                        h_box_pairs = np.concatenate([hdata.boxes[hdata.ho_pairs[h_fg_interactions, 0], :],
                                                      hdata.boxes[hdata.ho_pairs[h_fg_interactions, 1], :]
                                                      ], axis=1)
                        h_idxs = np.lexsort(h_box_pairs.T[::-1, :])
                        assert h_idxs.size == h_fg_interactions.size  # make sure it just sorted and didn't remove anything

                        h_to_d_hopairs_transf = h_idxs[np.argsort(d_idxs)]

                        assert np.all(d_box_pairs == h_box_pairs[h_to_d_hopairs_transf, :])
                        assert np.all(data.labels[d_fg_interactions, :] == hdata.labels[h_fg_interactions, :][h_to_d_hopairs_transf])

                        # Copy pstate labels from HAKE data for foreground interactions
                        new_ps_labels[d_fg_interactions, :] = hdata.ps_labels[h_fg_interactions, :][h_to_d_hopairs_transf]

                    new_data = new_data._replace(ps_labels=new_ps_labels)

                new_img_data_per_split[split].append(new_data)
        self._img_data_per_split = new_img_data_per_split

    @property
    def num_parts(self):
        return len(self.parts)

    @property
    def num_states(self):
        return len(self.states)

    def get_img_data(self, split) -> List[GTImgData]:
        return self._img_data_per_split[split]

    def get_img_path(self, split, fname):
        return os.path.join(self._split_img_dir[split], fname)

    def compute_img_data(self, split, driver) -> List[GTImgData]:
        annotations = driver.split_annotations[split]

        split_data = []
        for img_ann in annotations:
            im_boxes = []
            im_box_classes = []
            im_ho_pairs = []
            im_interactions = []
            curr_num_boxes = 0
            for inter in img_ann['interactions']:
                inter_id = inter['id']
                if not inter['invis']:
                    # Human
                    im_hum_boxes = inter['hum_bbox']
                    num_hum_boxes = im_hum_boxes.shape[0]
                    im_boxes.append(im_hum_boxes)
                    im_box_classes.append(np.full(num_hum_boxes, fill_value=self.human_class))

                    # Object
                    im_obj_boxes = inter['obj_bbox']
                    num_obj_boxes = im_obj_boxes.shape[0]
                    im_boxes.append(im_obj_boxes)
                    im_box_classes.append(np.full(num_obj_boxes, fill_value=self.interactions[inter_id, 1]))

                    # Interaction
                    new_inters = inter['conn']
                    new_inters += curr_num_boxes
                    new_inters[:, 1] += num_hum_boxes
                    num_inters = new_inters.shape[0]
                    im_ho_pairs.append(new_inters)
                    curr_num_boxes += num_hum_boxes + num_obj_boxes

                    im_labels_onehot = np.zeros((num_inters, self.num_interactions))
                    im_labels_onehot[np.arange(num_inters), inter_id] = 1
                    im_interactions.append(im_labels_onehot)

            gt_img_data = GTImgData(filename=img_ann['file'], img_size=img_ann['orig_img_size'][:2])
            if im_boxes:
                im_boxes = np.concatenate(im_boxes, axis=0)
                im_box_classes = np.concatenate(im_box_classes)
                im_ho_pairs = np.concatenate(im_ho_pairs, axis=0)
                im_interactions_onehot = np.concatenate(im_interactions, axis=0)

                im_boxes, u_idxs, inv_idxs = np.unique(im_boxes, return_index=True, return_inverse=True, axis=0)
                im_box_classes = im_box_classes[u_idxs]
                im_ho_pairs = np.stack([inv_idxs[im_ho_pairs[:, 0]], inv_idxs[im_ho_pairs[:, 1]]], axis=1)

                gt_img_data = gt_img_data._replace(boxes=im_boxes, box_classes=im_box_classes, ho_pairs=im_ho_pairs, labels=im_interactions_onehot)

            split_data.append(gt_img_data)

        return split_data

    def compute_hake_img_data(self, split) -> List[GTImgData]:
        all_img_anns = json.load(open(
            os.path.join(cfg.data_root, 'HICO-DET', 'HAKE', f'hico-det-{"training" if split == "train" else "test"}-set-instance-level.json'), 'r'))

        split_data = []
        for fname, img_ann in all_img_anns.items():
            assert img_ann['dataset'] == 'hico-det'
            assert img_ann['path_prefix'] == f'hico_20160224_det/images/{split}2015'

            if img_ann['labels']:
                img_sizes = {(x['width'], x['height']) for x in img_ann['labels']}
                assert len(img_sizes) == 1
                img_size = list(img_sizes)[0]

                im_boxes = []
                im_box_classes = []
                im_ho_pairs = []
                im_hoi_labels = []
                im_state_labels = []
                for hoi_ann in img_ann['labels']:
                    curr_num_boxes = len(im_boxes)
                    hoi_id = hoi_ann['hoi_id'] - 1

                    im_boxes.append(hoi_ann['human_bbox'])
                    im_box_classes.append(self.human_class)

                    im_boxes.append(hoi_ann['object_bbox'])
                    im_box_classes.append(self.interactions[hoi_id, 1])

                    im_ho_pairs.append([curr_num_boxes, curr_num_boxes + 1])

                    hoi_label_onehot = np.zeros(self.num_interactions)
                    hoi_label_onehot[hoi_id] = 1
                    im_hoi_labels.append(hoi_label_onehot)

                    hoi_state_labels_onehot = np.zeros(self.num_states)
                    for sl in hoi_ann.get('action_labels', []):
                        part, state = sl['human_part'], sl['partstate']
                        hoi_state_labels_onehot[self.states_per_part[part][state]] = 1
                    im_state_labels.append(hoi_state_labels_onehot)

                im_boxes = np.stack(im_boxes, axis=0) - 1  # for some reason there is a discrepancy of one pixel with HICO-DET annotations
                im_box_classes = np.array(im_box_classes)
                im_ho_pairs = np.stack(im_ho_pairs, axis=0)
                im_hoi_labels_onehot = np.stack(im_hoi_labels, axis=0)
                if im_state_labels:
                    im_state_labels = np.stack(im_state_labels, axis=0)

                im_boxes, u_idxs, inv_idxs = np.unique(im_boxes, return_index=True, return_inverse=True, axis=0)
                im_box_classes = im_box_classes[u_idxs]
                im_ho_pairs = np.stack([inv_idxs[im_ho_pairs[:, 0]], inv_idxs[im_ho_pairs[:, 1]]], axis=1)

                gt_img_data = GTImgData(filename=fname, img_size=np.array(img_size),
                                        boxes=im_boxes, box_classes=im_box_classes, ho_pairs=im_ho_pairs,
                                        labels=im_hoi_labels_onehot, ps_labels=im_state_labels)
            else:
                gt_img_data = GTImgData(filename=fname, img_size=None)  # FIXME

            split_data.append(gt_img_data)

        return split_data


class HicoDetDriver:
    def __init__(self):
        """
        Relevant class attributes:
            - null_interaction: the name of the null interaction
            - wn_action_dict [dict]: The 119 WordNet entries for all actions. Keys are wordnets IDs and each element contains:
                - 'wname' [str]: The name of the wordnet entry this actions refers to. It is in the form VERB.v.NUM, where VERB is the verb
                    describing the action and NUM is an index used to disambiguate between homonyms.
                - 'id' [int]: A number I have not understood the use of.
                - 'count' [int]: Another number I have not understood the use of.
                - 'syn' [list]: Set of synonyms
                - 'def' [str]: A definition
                - 'ex' [str]: An example (sometimes not provided)
                EXAMPLE: key: v00007012, entry:
                    {'id': 1, 'wname': 'blow.v.01', 'count': 6, 'syn': ['blow'], 'def': 'exhale hard', 'ex': 'blow on the soup to cool it down'}
            - action_dict [dict]: The 117 possible actions, including a null one. They are fewer than the entries in the WordNet dictionary
                because some action can have different meaning and thus two different WordNet entries. Keys are verbs in the base form and
                entries consist of:
                    - 'ing' [str]: -ing form of the verb (unchanged for the null one).
                    - 'wn_ids' [list(str)]: The WordNet IDs (AKA keys in `wn_action_dict`) corresponding to this verb (empty for the null one).
            - interaction_list [list(dict)]: The 600 interactions in HICO-DET. Each element consists of:
                - 'obj' [str]: The name of the object of the action (i.e., the target).
                - 'act' [str]: The verb describing the action (key in `action_dict`).
                - 'act_wid' [str]: The WordNet ID of the action (key in `wn_action_dict`), or None for the null interaction.
            - split_data [dict(dict)]: One entry per split, with keys in ['train', 'test']. Each entry is a dictionary with the following items:
                - 'img_dir' [str]: Path to the folder containing the images
                - 'annotations' [list(dict)]: Annotations for each image, thus structured:
                    - 'file' [str]: The file name
                    - 'orig_img_size' [array]: Image size expressed in [width, height, depth]
                    - 'interactions' [list(dict)]: Each entry has:
                            - 'id' [int]: The id of the interaction in `interaction_list`.
                            - 'invis' [bool]: Whether the interaction is invisible or not. It does NOT necesserily mean that it is not in the image.
                        If 'invis' is False then there are three more fields:
                            - 'hum_bbox' [array]: Hx4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to a human.
                            - 'obj_bbox' [array]: Ox4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to an object.
                            - 'conn' [array]: Cx2 with a pair of human-object indices for each interaction
                Other entries might be added to this dictionary for caching reasons.
        """

        self.data_dir = os.path.join(cfg.data_root, 'HICO-DET')
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')
        self.null_interaction = '__no_interaction__'

        train_annotations, test_annotations, interaction_list, wn_act_dict, act_dict = self.load_annotations()
        self.split_img_dir = {'train': os.path.join(self.data_dir, 'images', 'train2015'),
                              'test': os.path.join(self.data_dir, 'images', 'test2015')}
        self.split_annotations = {'train': train_annotations, 'test': test_annotations}
        self.interaction_list = interaction_list
        self.wn_action_dict = wn_act_dict
        self.action_dict = act_dict

    def load_annotations(self):
        def _parse_split(_split):
            # The many "-1"s are due to original values being suited for MATLAB.
            _annotations = []
            for _src_ann in src_anns['bbox_%s' % _split]:
                _ann = {'file': _src_ann[0],
                        'orig_img_size': np.array([int(_src_ann[1][field]) for field in ['width', 'height', 'depth']], dtype=np.int),
                        'interactions': []}
                for _inter in np.atleast_1d(_src_ann[2]):
                    _new_inter = {
                        'id': int(_inter['id']) - 1,
                        'invis': bool(_inter['invis']),
                    }
                    if not _new_inter['invis']:
                        _new_inter['hum_bbox'] = np.atleast_2d(np.array([_inter['bboxhuman'][c] - 1 for c in ['x1', 'y1', 'x2', 'y2']],
                                                                        dtype=np.int).T)
                        _new_inter['obj_bbox'] = np.atleast_2d(np.array([_inter['bboxobject'][c] - 1 for c in ['x1', 'y1', 'x2', 'y2']],
                                                                        dtype=np.int).T)
                        _new_inter['conn'] = np.atleast_2d(np.array([coord - 1 for coord in _inter['connection']], dtype=np.int))
                    _ann['interactions'].append(_new_inter)
                _annotations.append(_ann)
            return _annotations

        try:
            with open(self.path_pickle_annotation_file, 'rb') as f:
                d = pickle.load(f)
                train_annotations = d['train']
                test_annotations = d['test']
                interaction_list = d['interaction_list']
                wn_act_dict = d['wn_act_dict']
                act_dict = d['act_dict']
        except FileNotFoundError:
            src_anns = loadmat(os.path.join(self.data_dir, 'anno_bbox.mat'), squeeze_me=True)

            train_annotations = _parse_split(_split='train')
            test_annotations = _parse_split(_split='test')

            interaction_list, wn_act_dict, act_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({'train': train_annotations,
                             'test': test_annotations,
                             'interaction_list': interaction_list,
                             'wn_act_dict': wn_act_dict,
                             'act_dict': act_dict,
                             }, f)

        # Substitute 'no_interaction' with the specified null interaction string, if needed.
        act_dict[self.null_interaction] = act_dict.get('no_interaction', self.null_interaction)
        del act_dict['no_interaction']
        act_dict = {k: act_dict[k] for k in sorted(act_dict.keys())}
        for inter in interaction_list:
            if inter['act'] == 'no_interaction':
                inter['act'] = self.null_interaction
            if inter['obj'] == 'hair_drier':
                inter['obj'] = 'hair_dryer'

        return train_annotations, test_annotations, interaction_list, wn_act_dict, act_dict

    @staticmethod
    def parse_interaction_list(src_interaction_list):
        wact_dict = {}
        interaction_list = []
        act_dict = {}

        for i, interaction_ann in enumerate(src_interaction_list):
            fields = interaction_ann[-2].dtype.fields
            act_wann = {}
            act_wid = None
            if fields is None:  # Null interaction
                for j, s in enumerate(interaction_ann):
                    if j < 3:
                        if j > 0:
                            assert s == 'no_interaction'
                        assert isinstance(s, str)
                    else:
                        assert s.size == 0
            else:
                for f in fields:
                    fvalue = str(interaction_ann[-2][f])
                    try:
                        fvalue = int(fvalue)
                    except ValueError:
                        pass

                    if f == 'name':
                        act_wann['wname'] = fvalue
                    elif f == 'wid':
                        act_wid = fvalue
                    elif f == 'syn':
                        act_wann[f] = list(set(fvalue.split(' ')))
                    elif f == 'ex':
                        act_wann[f] = fvalue if fvalue != '[]' else ''
                    else:
                        act_wann[f] = fvalue

                # Add to the wordnet action dictionary
                assert wact_dict.setdefault(act_wid, act_wann) == act_wann, '\n%s\n%s' % (wact_dict[act_wid], act_wann)

            assert 'name' not in act_wann

            # Add to the action dictionary
            act, act_ing = interaction_ann[1], interaction_ann[2]
            d_act = act_dict.setdefault(act, {'ing': act_ing, 'wn_ids': []})
            assert d_act['ing'] == act_ing
            if act_wid is not None:
                act_dict[act]['wn_ids'] = sorted(set(act_dict[act]['wn_ids'] + [act_wid]))

            # Add to the interaction list
            new_action_ann = {'obj': interaction_ann[0], 'act': act, 'act_wid': act_wid}
            interaction_list.append(new_action_ann)

        # Sort
        wact_dict = {k: wact_dict[k] for k in sorted(wact_dict.keys())}
        act_dict = {k: act_dict[k] for k in sorted(act_dict.keys())}

        return interaction_list, wact_dict, act_dict


if __name__ == '__main__':
    h = HicoDetHake()
    print('Done.')

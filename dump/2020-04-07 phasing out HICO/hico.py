import os
import pickle
from typing import Dict, List

import imagesize
import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset, GTImgData


class Hico(HoiDataset):
    def __init__(self):
        driver = HicoDriver()  # type: HicoDriver

        object_classes = sorted(set([inter['obj'] for inter in driver.interaction_list]))
        action_classes = list(driver.predicate_dict.keys())
        null_action = driver.null_interaction
        interactions_classes = [[inter['pred'], inter['obj']] for inter in driver.interaction_list]

        train_annotations = driver.split_annotations['train']
        train_annotations[np.isnan(train_annotations)] = 0
        train_annotations[train_annotations < 0] = 0
        test_annotations = driver.split_annotations['test']
        test_annotations[np.isnan(test_annotations)] = 0
        test_annotations[test_annotations < 0] = 0
        annotations_per_split = {'train': train_annotations, 'test': test_annotations}

        super().__init__(object_classes=object_classes, action_classes=action_classes, null_action=null_action,
                         interactions_classes=interactions_classes)
        self.split_filenames = driver.split_filenames
        self.split_labels = annotations_per_split
        self.split_img_dir = driver.split_img_dir
        self.split_img_dims = driver.split_img_dims  # (w, h)
        self._img_data_per_split = {s: [GTImgData(filename=fn, img_size=dims, labels=l)
                                        for fn, dims, l in zip(self.split_filenames[s], self.split_img_dims[s], self.split_labels[s])]
                                    for s in ['train', 'test']}  # type: Dict[str, List[GTImgData]]

    def get_img_data(self, split) -> List[GTImgData]:
        return self._img_data_per_split[split]

    def get_img_path(self, split, fname):
        return os.path.join(self.split_img_dir[split], fname)


class HicoDriver:
    def __init__(self):
        """
        Relevant class attributes:
            - null_interaction: the name of the null interaction
            - wn_action_dict [dict]: The 119 WordNet entries for all actions. Keys are wordnets IDs and each element contains:
                - 'wname' [str]: The name of the wordnet entry this actions refers to. It is in the form VERB.v.NUM, where VERB is the verb
                    describing the action and NUM is an index used to disambiguate between homonyms.
                - 'syn' [list]: Set of synonyms
                - 'def' [str]: A definition
                - 'add_def' [str]: An additional definition (sometimes not provided)
                EXAMPLE: key: v00007012, entry:
                    {'wname': 'blow.v.01', 'syn': ['blow'], 'def': 'exhale hard'}
            - predicate_dict [dict]: The 117 possible actions, including a null one. They are fewer than the entries in the WordNet dictionary
                because some predicate can have different meaning and thus two different WordNet entries. Keys are verbs in the base form and
                entries consist of:
                    - 'ing' [str]: -ing form of the verb (unchanged for the null one).
                    - 'wn_ids' [list(str)]: The WordNet IDs (AKA keys in `wn_action_dict`) corresponding to this verb (empty for the null one).
            - interaction_list [list(dict)]: The 600 interactions in HICO-DET. Each element consists of:
                - 'obj' [str]: The name of the object of the action (i.e., the target).
                - 'pred' [str]: The verb describing the action (key in `predicate_dict`).
                - 'pred_wid' [str]: The WordNet ID of the action (key in `wn_action_dict`), or None for the null interaction.
            - split_annotations [dict(ndarray)]: One entry per split, with keys in ['train', 'test'].
                Each entry is a matrix with dimensions [num_images, num_interactions] and each cell ij has a value in {-1, 0, 1} according to whether
                action j is a hard negative, uncertain/unknown or a hard positive in image i.
        """

        self.data_dir = os.path.join(cfg.data_root, 'HICO')
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')
        self.path_pickle_img_dim_file = os.path.join(self.data_dir, 'img_dims.pkl')
        self.null_interaction = '__no_interaction__'

        # Annotations
        train_annotations, train_fns, test_annotations, test_fns, interaction_list, wn_pred_dict, pred_dict = self.load_annotations()
        self.split_img_dir = {'train': os.path.join(self.data_dir, 'images', 'train2015'),
                              'test': os.path.join(self.data_dir, 'images', 'test2015')}
        self.split_annotations = {'train': train_annotations, 'test': test_annotations}
        self.split_filenames = {'train': train_fns, 'test': test_fns}
        self.interaction_list = interaction_list
        self.wn_predicate_dict = wn_pred_dict
        self.predicate_dict = pred_dict
        self.split_img_dims = self.load_img_dims()

    def load_annotations(self):
        try:
            with open(self.path_pickle_annotation_file, 'rb') as f:
                d = pickle.load(f)
                train_annotations = d[f'train_anno']
                train_fns = d[f'train_fn']
                test_annotations = d[f'test_anno']
                test_fns = d[f'test_fn']
                interaction_list = d['interaction_list']
                wn_pred_dict = d['wn_pred_dict']
                pred_dict = d['pred_dict']
        except FileNotFoundError:
            # 'anno_train': 600 x 38118 matrix. Associates to each training set images action labels. Specifically, cell (i,j) can contain one
            #       of the four values -1, 0, 1 or NaN according to whether action i is a hard negative, a soft negative/positive,
            #       a hard positive or unknown in image j.
            # 'anno_test': 600 x 9658 matrix. Same format for the training set one.
            # 'list_train' and 'list_set' are respectively 38118- and 9658- dimensional vectors of file names.
            src_anns = loadmat(os.path.join(self.data_dir, 'anno.mat'), squeeze_me=True)

            train_annotations = src_anns['anno_train'].T
            train_fns = [fn for fn in src_anns['list_train']]
            test_annotations = src_anns['anno_test'].T
            test_fns = [fn for fn in src_anns['list_test']]
            interaction_list, wn_pred_dict, pred_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({f'train_anno': train_annotations,
                             f'train_fn': train_fns,
                             f'test_anno': test_annotations,
                             f'test_fn': test_fns,
                             'interaction_list': interaction_list,
                             'wn_pred_dict': wn_pred_dict,
                             'pred_dict': pred_dict,
                             }, f)

        assert train_annotations.shape[0] == len(train_fns)
        assert test_annotations.shape[0] == len(test_fns)

        # Substitute 'no_interaction' with the specified null interaction string, if needed.
        pred_dict[self.null_interaction] = pred_dict.get('no_interaction', self.null_interaction)
        del pred_dict['no_interaction']
        pred_dict = {k: pred_dict[k] for k in sorted(pred_dict.keys())}
        for inter in interaction_list:
            if inter['pred'] == 'no_interaction':
                inter['pred'] = self.null_interaction
            if inter['obj'] == 'hair_drier':
                inter['obj'] = 'hair_dryer'

        return train_annotations, train_fns, test_annotations, test_fns, interaction_list, wn_pred_dict, pred_dict

    def load_img_dims(self, use_imagesize=False):
        try:
            with open(self.path_pickle_img_dim_file, 'rb') as f:
                split_img_dims = pickle.load(f)
        except FileNotFoundError:
            if use_imagesize:
                # This doesn't work if the image is rotated.
                split_img_dims = {split: np.array([imagesize.get(os.path.join(self.split_img_dir[split], fn)) for fn in fns])
                                  for split, fns in self.split_filenames.items()}
            else:
                split_img_dims = {}
                for split, fns in self.split_filenames.items():
                    all_img_dims = []
                    for fn in fns:
                        image = Image.open(os.path.join(self.split_img_dir[split], fn))
                        try:
                            image = ImageOps.exif_transpose(image)
                        except TypeError:
                            pass
                        img_wh = np.asarray(image).shape[:2][::-1]
                        all_img_dims.append(img_wh)
                    split_img_dims[split] = np.array(all_img_dims)
            with open(self.path_pickle_img_dim_file, 'wb') as f:
                pickle.dump(split_img_dims, f)
        return split_img_dims

    @staticmethod
    def parse_interaction_list(src_interaction_list):
        wpred_dict = {}
        interaction_list = []
        pred_dict = {}

        for i, interaction_ann in enumerate(src_interaction_list):
            fields = interaction_ann[-2].dtype.fields
            pred_wann = {}
            pred_wid = None
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
                        pred_wann['wname'] = fvalue
                    elif f == 'wid':
                        pred_wid = fvalue
                    elif f == 'syn':
                        pred_wann[f] = list(set(fvalue.split(' ')))
                    elif f == 'ex':
                        pred_wann[f] = fvalue if fvalue != '[]' else ''
                    else:
                        pred_wann[f] = fvalue

                # Add to the wordnet predicate dictionary
                assert wpred_dict.setdefault(pred_wid, pred_wann) == pred_wann, '\n%s\n%s' % (wpred_dict[pred_wid], pred_wann)

            assert 'name' not in pred_wann

            # Add to the predicate dictionary
            pred, pred_ing = interaction_ann[1], interaction_ann[2]
            d_pred = pred_dict.setdefault(pred, {'ing': pred_ing, 'wn_ids': []})
            assert d_pred['ing'] == pred_ing
            if pred_wid is not None:
                pred_dict[pred]['wn_ids'] = sorted(set(pred_dict[pred]['wn_ids'] + [pred_wid]))

            # Add to the interaction list
            new_action_ann = {'obj': interaction_ann[0], 'pred': pred, 'pred_wid': pred_wid}
            interaction_list.append(new_action_ann)

        # Sort
        wpred_dict = {k: wpred_dict[k] for k in sorted(wpred_dict.keys())}
        pred_dict = {k: pred_dict[k] for k in sorted(pred_dict.keys())}

        return interaction_list, wpred_dict, pred_dict


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
    h = Hico()
    print('Done.')

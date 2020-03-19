import json
import os

from config import cfg


class VCocoDriver:
    def __init__(self):
        self.data_dir = os.path.join(cfg.data_root, 'V-COCO')
        self.null_interaction = '__no_interaction__'

        # Annotations
        self.annotations, self.image_infos, self.categories, self.hoi_data = self.load_annotations()
        # self.split_img_dir = {Splits.TRAIN: os.path.join(self.data_dir, 'images', 'train2015'),
        #                       Splits.TEST: os.path.join(self.data_dir, 'images', 'test2015')}
        # self.split_annotations = {Splits.TRAIN: train_annotations, Splits.TEST: test_annotations}
        # self.split_filenames = {Splits.TRAIN: train_fns, Splits.TEST: test_fns}
        # self.interaction_list = interaction_list
        # self.wn_predicate_dict = wn_pred_dict
        # self.predicate_dict = pred_dict

    def load_annotations(self):
        with open(os.path.join(self.data_dir, 'instances_vcoco_all_2014.json'), 'r') as f:
            bbox_data = json.load(f)
        annotations = bbox_data['annotations']
        image_infos = bbox_data['images']
        categories = bbox_data['categories']

        hoi_data = {}
        with open(os.path.join(self.data_dir, 'vcoco', 'vcoco_train.json'), 'r') as f:
            hoi_data['train'] = json.load(f)
        with open(os.path.join(self.data_dir, 'vcoco', 'vcoco_val.json'), 'r') as f:
            hoi_data['val'] = json.load(f)
        with open(os.path.join(self.data_dir, 'vcoco', 'vcoco_test.json'), 'r') as f:
            hoi_data['test'] = json.load(f)
        hoi_class_data = {hoi_data[split] for split in ['train', 'val', 'test']}

        num_categories = len(hoi_data['train'])
        assert num_categories == hoi_data['val'] == hoi_data['test']
        for i in range(num_categories):
            for k in ['action_name', 'role_name', 'include']:
                assert hoi_data['train'][i][k] == hoi_data['val'][i][k] == hoi_data['test'][i][k]
        # TODO
        return annotations, image_infos, categories, hoi_data


if __name__ == '__main__':
    h = VCocoDriver()
    print('Done.')

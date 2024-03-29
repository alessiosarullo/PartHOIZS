import os
from typing import List, NamedTuple, Union

import numpy as np


class GTImgData(NamedTuple):
    filename: str
    img_size: np.ndarray  # width, height
    labels: Union[None, np.ndarray] = None  # R
    boxes: Union[None, np.ndarray] = None  # N x 4
    box_classes: Union[None, np.ndarray] = None  # N
    ho_pairs: Union[None, np.ndarray] = None  # R x 2, [hum_idx, obj_idx]
    ps_labels: Union[None, np.ndarray] = None  # R x S, multilabel


class HoiDataset:
    def __init__(self, object_classes, action_classes, null_action, interactions_classes=None, interactions=None):
        assert (interactions_classes is None or interactions is None)  # specify only one of them

        self.objects = object_classes
        self.object_index = {obj: i for i, obj in enumerate(self.objects)}

        # Actions
        self.null_action = null_action
        self.actions = action_classes
        self.action_index = {act: i for i, act in enumerate(self.actions)}
        assert self.action_index[self.null_action] == 0

        # Interactions
        if interactions is not None:
            self.interactions = interactions.astype(np.int)
        else:
            assert interactions_classes is not None
            self.interactions = np.array([[self.action_index[act], self.object_index[obj]] for act, obj in interactions_classes])  # [a, o]

        # It must be possible NOT to interact with any object
        assert sorted(self.interactions[self.interactions[:, 0] == 0, 1].tolist()) == list(range(self.num_objects))

        self.oa_to_interaction = np.full([self.num_objects, self.num_actions], fill_value=-1, dtype=np.int)
        self.oa_to_interaction[self.interactions[:, 1], self.interactions[:, 0]] = np.arange(self.num_interactions)

    def get_img_data(self, split) -> List[GTImgData]:
        raise NotImplementedError

    def get_img_path(self, split, fname):
        raise NotImplementedError

    def get_fname_id(self, fname) -> int:
        return int(os.path.splitext(fname)[0].split('_')[-1])

    @property
    def labels_are_actions(self) -> bool:
        raise NotImplementedError

    @property
    def human_class(self) -> int:
        return self.object_index['person']

    @property
    def interactions_str(self):
        return [f'{self.actions[a]} {self.objects[o]}' for a, o in self.interactions]

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def interaction_to_object_mat(self):
        interactions_to_obj = np.zeros((self.num_interactions, self.num_objects))
        interactions_to_obj[np.arange(self.num_interactions), self.interactions[:, 1]] = 1
        return interactions_to_obj

    @property
    def interaction_to_action_mat(self):
        interactions_to_act = np.zeros((self.num_interactions, self.num_actions))
        interactions_to_act[np.arange(self.num_interactions), self.interactions[:, 0]] = 1
        return interactions_to_act

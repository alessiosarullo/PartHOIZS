import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset
from lib.dataset.utils import Splits
from lib.timer import Timer


class AbstractHoiDatasetSplit(Dataset):
    def __init__(self, split):
        assert split != Splits.VAL
        self.split = split

    @property
    def num_objects(self):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError

    @property
    def num_interactions(self):
        raise NotImplementedError

    @property
    def num_images(self):
        raise NotImplementedError

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class HoiDatasetSplit(AbstractHoiDatasetSplit):
    def __init__(self, split, full_dataset: HoiDataset, object_inds=None, action_inds=None):
        super().__init__(split)
        self.full_dataset = full_dataset  # type: HoiDataset
        self.keep_inds = self.holdout_inds = None  # These will be set later, if needed.

        object_inds = sorted(object_inds) if object_inds is not None else range(self.full_dataset.num_objects)
        self.objects = [full_dataset.objects[i] for i in object_inds]
        self.seen_objects = np.array(object_inds, dtype=np.int)

        action_inds = sorted(action_inds) if action_inds is not None else range(self.full_dataset.num_actions)
        self.actions = [full_dataset.actions[i] for i in action_inds]
        self.seen_actions = np.array(action_inds, dtype=np.int)

        seen_op_mat = self.full_dataset.oa_pair_to_interaction[self.seen_objects, :][:, self.seen_actions]
        seen_interactions = set(np.unique(seen_op_mat).tolist()) - {-1}
        self.seen_interactions = np.array(sorted(seen_interactions), dtype=np.int)
        self.interactions = self.full_dataset.interactions[self.seen_interactions, :]  # original action and object inds

        try:
            pc_feats_file = h5py.File(self._get_precomputed_feats_fn(self.split), 'r')
            self.pc_img_feats = pc_feats_file['img_feats'][:]
        except OSError:
            self.pc_img_feats = None

        self.img_labels = self.full_dataset.split_img_labels[self.split]
        if self.seen_interactions.size < self.full_dataset.num_interactions:
            all_labels = self.img_labels
            self.img_labels = np.zeros_like(all_labels)
            self.img_labels[:, self.seen_interactions] = all_labels[:, self.seen_interactions]
        self.non_empty_inds = np.flatnonzero(np.any(self.img_labels, axis=1))

    def hold_out(self, ratio):
        if cfg.no_filter_bg_only:
            num_imgs = self.num_images
            image_ids = np.arange(num_imgs)
        else:
            num_imgs = len(self.non_empty_inds)
            image_ids = self.non_empty_inds
        num_keep_imgs = num_imgs - int(num_imgs * ratio)
        keep_inds = np.random.choice(image_ids, size=num_keep_imgs, replace=False)
        self.keep_inds = keep_inds
        self.holdout_inds = np.setdiff1d(image_ids, keep_inds)

    @property
    def img_feat_dim(self):
        return self.pc_img_feats.shape[1]

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
    def num_images(self):
        return self.full_dataset.split_img_labels[self.split].shape[0]

    def _collate(self, idx_list, device):
        Timer.get('GetBatch').tic()
        idxs = np.array(idx_list)
        feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
        if self.split != Splits.TEST:
            labels = torch.tensor(self.img_labels[idxs, :], dtype=torch.float32, device=device)
        else:
            labels = None
        Timer.get('GetBatch').toc()
        return feats, labels, []

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=None, holdout_set=False, **kwargs):
        if self.pc_img_feats is None:
            raise NotImplementedError('This is only possible with precomputed features.')

        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        if drop_last is None:
            drop_last = False if self.split == Splits.TEST else True
        batch_size = batch_size * num_gpus

        if self.split == Splits.TEST:
            ds = self
        else:
            if self.keep_inds is None:
                assert self.holdout_inds is None and holdout_set is False
                ds = self
            else:
                ds = Subset(self, self.holdout_inds if holdout_set else self.keep_inds)
        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: self._collate(x, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def get_img(self, img_id):
        img_fn = os.path.join(self.full_dataset.get_img_dir(self.split), self.full_dataset.split_filenames[self.split][img_id])
        img = Image.open(img_fn).convert('RGB')
        return img

    def __getitem__(self, idx):
        # This should only be used by the data loader (see above).
        return idx

    def __len__(self):
        return self.num_images

    def _get_precomputed_feats_fn(self, split):
        raise NotImplementedError

    @classmethod
    def instantiate_full_dataset(cls) -> HoiDataset:
        raise NotImplementedError

    @classmethod
    def get_splits(cls, act_inds=None, obj_inds=None):
        splits = {}
        full_dataset = cls.instantiate_full_dataset()

        train_split = cls(split=Splits.TRAIN, full_dataset=full_dataset, object_inds=obj_inds, action_inds=act_inds)
        if cfg.val_ratio > 0:
            train_split.hold_out(ratio=cfg.val_ratio)
        splits[Splits.TRAIN] = train_split
        splits[Splits.TEST] = cls(split=Splits.TEST, full_dataset=full_dataset)

        train_str = Splits.TRAIN.value.capitalize()
        if obj_inds is not None:
            print(f'{train_str} objects ({train_split.seen_objects.size}):', train_split.seen_objects.tolist())
        if act_inds is not None:
            print(f'{train_str} actions ({train_split.seen_actions.size}):', train_split.seen_actions.tolist())
        if obj_inds is not None or act_inds is not None:
            print(f'{train_str} interactions ({train_split.seen_interactions.size}):', train_split.seen_interactions.tolist())

        return splits

import h5py
from lib.dataset.cocoa import Cocoa
import os
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    args = parser.parse_args()
    fn = args.fn
    assert fn.split('.')[-2].endswith('train2014')

    f = h5py.File(fn, 'r')
    # ['boxes', 'fname_ids', 'keypoints', 'kp_boxes', 'kp_feats', 'person_feats', 'scores']
    ids_to_idx = {}
    for i, fid in enumerate(f['fname_ids'][:]):
        ids_to_idx.setdefault(fid, []).append(i)

    cocoa = Cocoa()
    for split in ['train', 'test']:
        new_fn = fn.replace('train2014', split)
        if os.path.exists(new_fn):
            print('Overwriting', new_fn)
        new_f = h5py.File(new_fn, 'w')

        split_ids = sorted({cocoa.get_fname_id(imd.filename) for imd in cocoa.get_img_data(split)})
        h5_idxs = np.array(sorted([idx for sid in split_ids for idx in ids_to_idx.get(sid, [])]))

        for k in f:
            new_f.create_dataset(name=k, data=f[k][:][h5_idxs])

        new_f.close()

    # Sanity check
    train_f = h5py.File(fn.replace('train2014', 'train'), 'r')
    test_f = h5py.File(fn.replace('train2014', 'test'), 'r')
    assert [k for k in f] == [k for k in train_f] == [k for k in test_f]
    assert [f[k].shape[0] for k in f] == [train_f[k].shape[0] + test_f[k].shape[0] for k in f]

    train_f.close()
    test_f.close()
    f.close()
    print('Done.')


if __name__ == '__main__':
    main()

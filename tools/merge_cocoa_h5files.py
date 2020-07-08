import argparse
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    args = parser.parse_args()
    fn = args.fn
    assert 'cocoa' in fn

    f_tr = h5py.File(fn + '_train.h5', 'r')
    f_te = h5py.File(fn + '_test.h5', 'r')
    assert not (set(f_te.keys()) - set(f_tr.keys()))
    assert all(['label' in x for x in set(f_tr.keys()) - set(f_te.keys())])
    # ['boxes', 'fname_ids', 'keypoints', 'kp_boxes', 'kp_feats', 'person_feats', 'scores']

    new_fn = fn.replace('cocoa', 'cocoaall') + '_test.h5'
    assert not os.path.exists(new_fn)
    new_f = h5py.File(new_fn, 'w')
    for k in f_te:
        new_f.create_dataset(name=k, data=np.concatenate((f_tr[k][:], f_te[k][:]), axis=0))

    new_f.close()
    f_tr.close()
    f_te.close()


if __name__ == '__main__':
    main()

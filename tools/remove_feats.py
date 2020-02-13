import argparse

import h5py


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    source_fn = args.fn
    f = h5py.File(source_fn, 'r')
    g = h5py.File(source_fn.split('.')[0] + '__nofeats.h5', 'w')

    for k in f:
        if 'feat' not in k:
            g.create_dataset(k, data=f[k])

    f.close()
    g.close()


if __name__ == '__main__':
    main()

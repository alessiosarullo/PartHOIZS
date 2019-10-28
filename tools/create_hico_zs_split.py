import pickle
import sys
import argparse

import numpy as np

from lib.dataset.hico import Hico


def main():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int)
    args = parser.parse_args()
    idx = args.idx

    np.random.seed(idx)

    hico = Hico()
    num_seen_obj = 40
    num_seen_act = 59

    with open('imagenet1000classes.txt', 'r') as f:
        inet_categories = [' '.join(l.strip('{}, \n').split()[1:]).strip("'").split(', ')
                           for l in f.readlines()]
        inet_classes = {c for cat in inet_categories for c in cat}

    hico_objects = [obj.replace('_', ' ') for obj in hico.objects]
    common_str = set(hico_objects) & inet_classes
    common = {i for i, n in enumerate(hico_objects) if n in common_str}
    print(len(common))
    hico_only = set(range(hico.num_objects)) - common - {hico.human_class}

    seen_obj = np.random.choice(np.array(list(hico_only)), size=num_seen_obj - len(common) - 1, replace=False)
    seen_obj = np.sort(np.concatenate([seen_obj, np.array([hico.human_class]), np.array(list(common))]))
    assert hico.human_class in seen_obj

    seen_act = np.random.choice(np.arange(1, hico.num_actions), size=num_seen_act - 1, replace=False)
    seen_act = np.sort(np.concatenate([np.array([0]), seen_act]))

    u_seen_obj = np.unique(seen_obj)
    u_seen_act = np.unique(seen_act)
    assert seen_obj.size == num_seen_obj and seen_act.size == num_seen_act
    assert u_seen_obj.size == num_seen_obj and u_seen_act.size == num_seen_act
    d = {'train': {'obj': seen_obj, 'pred': seen_act}}
    filename = f'zero-shot_inds/seen_inds_{idx}.pkl.push'
    with open(filename, 'wb') as f:
        pickle.dump(d, f)
    print(f'File {filename} created.')


if __name__ == '__main__':
    main()

import os
import pickle
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from config import cfg
from lib.dataset.hico import HicoSplit, Hico
from lib.dataset.utils import Splits
from lib.models.abstract_model import AbstractModel
from scripts.utils import get_all_models_by_name


def save():
    cfg.parse_args(fail_if_missing=False)
    cfg.load()

    if cfg.seenf >= 0:
        inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
        act_inds = sorted(inds_dict[Splits.TRAIN.value]['act'].tolist())
        obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())
    else:
        obj_inds = act_inds = None

    splits = HicoSplit.get_splits(obj_inds=obj_inds, act_inds=act_inds)
    train_split, val_split, test_split = splits[Splits.TRAIN], splits[Splits.VAL], splits[Splits.TEST]

    # Model
    model = get_all_models_by_name()[cfg.model](train_split)  # type: AbstractModel

    ckpt = torch.load(cfg.best_model_file, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    _, act_class_embs = model.gcn()
    act_class_embs = act_class_embs.cpu().numpy()

    os.makedirs(cfg.output_analysis_path, exist_ok=True)
    np.save(os.path.join(cfg.output_analysis_path, 'act_embs'), act_class_embs)


def show():
    # sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl/asl1/2019-09-25_10-25-31_SINGLE']
    # sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_Ra/Ra-10-03/2019-09-25_10-25-51_SINGLE/']
    sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl_Ra/asl1_Ra-10-03/2019-09-25_14-21-33_SINGLE']
    print(sys.argv)
    cfg.parse_args(fail_if_missing=False)
    cfg.load()

    dataset = Hico()
    n = 10
    print(' ' * 3, end='  ')
    for i in range(n):
        print(f'{i:<20d}', end=' ')
    for i, a in enumerate(dataset.actions):
        if i % n == 0:
            print()
            print(f'{i // n:3d}', end=': ')
        print(f'{a:20s}', end=' ')
    print()

    inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
    seen_act_inds = np.array(sorted(inds_dict[Splits.TRAIN.value]['act'].tolist()))
    unseen_act_inds = np.setdiff1d(np.arange(dataset.num_actions), seen_act_inds)

    act_class_embs = np.load(os.path.join(cfg.output_analysis_path, 'act_embs.npy'))
    perplexity = 20.0
    act_emb_2d = TSNE(perplexity=perplexity).fit_transform(act_class_embs)

    fig, ax = plt.subplots()
    for ainds, c in [(seen_act_inds, 'b'),
                     (unseen_act_inds, 'r')]:
        x, y = act_emb_2d[ainds, 0], act_emb_2d[ainds, 1]
        ax.scatter(x, y, c=c)
        for i, txt in enumerate(ainds):
            ax.annotate(txt, (x[i], y[i]))
    # ax.set_title(f'Perplexity = {perplexity}')
    ax.axis('off')
    print(f'Perplexity = {perplexity}')

    plt.show()


if __name__ == '__main__':
    if torch.cuda.is_available():
        save()
    else:
        show()

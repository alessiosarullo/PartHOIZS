import pickle

import numpy as np

from config import cfg
from lib.dataset.hicodet_hake import HicoDetHake



def get_zs_classes(hico_hake: HicoHake, fname=None):
    if fname is None:
        fname = cfg.seen_classes_file

    hh = hico_hake
    inds_dict = pickle.load(open(fname, 'rb'))
    seen_objs = inds_dict['train']['obj']
    seen_acts = inds_dict['train']['act']
    seen_interactions = np.setdiff1d(np.unique(hh.oa_pair_to_interaction[seen_objs, :][:, seen_acts]), -1)
    unseen_objs = np.setdiff1d(np.arange(hh.num_objects), seen_objs)
    unseen_acts = np.setdiff1d(np.arange(hh.num_actions), seen_acts)
    unseen_interactions = np.setdiff1d(np.arange(hh.num_interactions), seen_interactions)
    return seen_objs, unseen_objs, seen_acts, unseen_acts, seen_interactions, unseen_interactions

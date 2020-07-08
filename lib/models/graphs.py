import pickle

import numpy as np
import torch

from lib.dataset.cocoa import Cocoa, CocoaSplit
from lib.dataset.hico_cocoa import HicoCocoa, HicoCocoaSplit
from lib.dataset.hicodet_hake import HicoDetHake
from lib.dataset.hoi_dataset import HoiDataset
from lib.dataset.utils import get_obj_mapping
from lib.dataset.vcoco import VCocoSplit, VCoco
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.mine_interactions import get_interactions_from_ext_src, check_hoi_coverage


def get_states_vars_from_hake(hh, sym):
    if not isinstance(hh, HicoDetHake):
        raise NotImplementedError

    if sym:
        _src_a_o_s_cooccs = np.zeros((hh.num_actions, hh.num_objects, hh.num_symstates))
        _states = hh.symstates
    else:
        _src_a_o_s_cooccs = np.zeros((hh.num_actions, hh.num_objects, hh.num_states))
        _states = hh.states
    _src_num_ao_pairs = np.zeros((hh.num_actions, hh.num_objects), dtype=np.int)
    for split in ['train', 'test']:
        for gt_img_data in hh.get_img_data(split=split):
            pstate_labels = gt_img_data.ps_labels
            hoi_labels = gt_img_data.labels
            if pstate_labels is None or hoi_labels is None:
                continue

            assert np.all((pstate_labels == 0) | (pstate_labels == 1))
            pstate_labels = pstate_labels.astype(np.bool)
            hoi_labels = hoi_labels.astype(np.int)
            act_labels = hh.interactions[hoi_labels, 0]
            ho_obj_labels = hh.interactions[hoi_labels, 1]
            assert act_labels.shape[0] == pstate_labels.shape[0] == ho_obj_labels.shape[0] \
                   and act_labels.ndim == ho_obj_labels.ndim == 1 and pstate_labels.shape[1] == hh.num_states
            for a, o, ps_mask in zip(act_labels, ho_obj_labels, pstate_labels):  # for each pair
                if o < 0:
                    continue
                if sym:
                    ps_mask = np.array([ps_mask[js].any() for i, js in enumerate(hh.symstates_inds)])
                ps = np.flatnonzero(ps_mask)
                _src_a_o_s_cooccs[a, o, ps] += 1
                _src_num_ao_pairs[a, o] += 1
    return _src_a_o_s_cooccs, _src_num_ao_pairs, _states


def get_vcoco_graphs(vcoco_split: VCocoSplit, source_ds: HoiDataset, to_torch=True, verbose=False, sym=True, ext_interactions=False):
    if ext_interactions:
        ext_interactions = get_interactions_from_ext_src(hoi_ds=vcoco_split.full_dataset)
        _ext_objects_per_action = {}
        for a, o in ext_interactions:
            _ext_objects_per_action.setdefault(a, set()).add(o)
        _ext_objects_per_action = {k: sorted(v) for k, v in _ext_objects_per_action.items()}

        def _get_interacting_objects_from_act(_a_src, _a_coco):
            return np.array(_ext_objects_per_action.get(_a_coco, []))
    else:
        def _get_interacting_objects_from_act(_a_src, _a_coco):
            return [h_to_v_obj_mapping[o_src] for o_src in np.flatnonzero(source_ds.oa_to_interaction[:, _a_src] >= 0)]

    src_a_o_s_cooccs, src_num_ao_pairs, states = get_states_vars_from_hake(hh=source_ds, sym=sym)

    vcoco_ds = vcoco_split.full_dataset
    v_to_h_act_mapping = {'hit_instr': 'swing',
                          'lay_instr': 'lie_on',
                          'look_obj': 'inspect',
                          'skateboard_instr': 'ride',
                          'ski_instr': 'ride',
                          'snowboard_instr': 'ride',
                          'surf_instr': 'ride',
                          'work_on_computer_instr': 'type_on'
                          }
    h_to_v_obj_mapping = get_obj_mapping(source_ds.objects, coco_to_hico=False)
    v_to_h_obj_mapping = get_obj_mapping(source_ds.objects, coco_to_hico=True)

    vcoco_interactions_from_hico = []
    act_obj_state_cooccs = np.zeros((vcoco_ds.num_actions, vcoco_ds.num_objects, src_a_o_s_cooccs.shape[-1]))
    num_vcoco_ao_pairs_in_src = np.zeros((vcoco_ds.num_actions, vcoco_ds.num_objects), dtype=np.int)
    for a_coco, a_coco_str in enumerate(vcoco_ds.actions):
        if a_coco == 0:
            continue
        role = a_coco_str.split('_')[-1]
        if role == 'agent':
            continue

        do_mapping = (a_coco_str in v_to_h_act_mapping)
        a_coco_str_no_role = '_'.join(a_coco_str.split('_')[:-1])

        # Get corresponding action index in source dataset
        if do_mapping:
            a_src_str = v_to_h_act_mapping[a_coco_str]
        elif role == 'instr' and any([a.split('_')[0] == a_coco_str.split('_')[0] for i, a in enumerate(vcoco_ds.actions) if i != a_coco]):
            a_src_str = a_coco_str_no_role + '_with'
        else:
            a_src_str = a_coco_str_no_role
        try:
            a_src = source_ds.action_index[a_src_str]
        except KeyError:
            continue
        assert a_src >= 0

        if do_mapping:
            # Possibly restrict interactions to specific objects
            objs_coco = _get_interacting_objects_from_act(_a_src=a_src,
                                                          _a_coco=a_coco if v_to_h_act_mapping[a_coco_str] != 'ride'
                                                          else vcoco_ds.action_index['ride_instr'])
            objs_coco = [o_coco for o_coco in objs_coco if a_coco_str_no_role in vcoco_ds.objects[o_coco]] or objs_coco
        else:
            objs_coco = _get_interacting_objects_from_act(_a_src=a_src, _a_coco=a_coco)

        for o_coco in objs_coco:
            vcoco_interactions_from_hico.append((a_coco, o_coco))

            o_src = v_to_h_obj_mapping[o_coco]

            # Get relevant part states
            act_obj_state_cooccs[a_coco, o_coco, :] = src_a_o_s_cooccs[a_src, o_src, :]
            num_vcoco_ao_pairs_in_src[a_coco, o_coco] = src_num_ao_pairs[a_src, o_src]

    interactions = np.unique(np.concatenate([vcoco_split.interactions, np.array(sorted(vcoco_interactions_from_hico))], axis=0), axis=0)

    isolated_actions_inds = np.setdiff1d(np.arange(vcoco_ds.num_actions), np.unique(interactions[:, 0]))
    if verbose:
        print('Isolated actions:', ', '.join([f'{vcoco_ds.actions[i]}{"" if i in vcoco_split.seen_actions else "*"}'
                                              for i in isolated_actions_inds]))
        print('Uncovered interactions:\n',
              '\n\t'.join([f'{vcoco_ds.actions[a]:25s} {vcoco_ds.objects[o]:20s}'
                           for a, o in sorted({(a, o) for a, o in vcoco_ds.interactions.tolist()} -
                                              {(a, o) for a, o in interactions.tolist()}
                                              )
                           ]))
        print('Interactions:\n',
              '\n\t'.join([f'{vcoco_ds.actions[a]:25s} {vcoco_ds.objects[o]:20s}' for a, o in interactions]))

    oa_adj = np.zeros([vcoco_ds.num_objects, vcoco_ds.num_actions], dtype=np.float32)
    oa_adj[interactions[:, 1], interactions[:, 0]] = 1
    oa_adj[:, 0] = 0
    if verbose:
        _act_obj_state_cooccs = act_obj_state_cooccs.sum(axis=1)
        print('Action-state pairs:\n',
              '\n\t'.join([f'{vcoco_ds.actions[a]:25s} {states[ps]:35s} {_act_obj_state_cooccs[a, ps] * 100:.0f}'
                           for a, ps in np.stack(np.where(_act_obj_state_cooccs), axis=1)]))

    if to_torch:
        oa_adj = torch.from_numpy(oa_adj)
        act_obj_state_cooccs = torch.from_numpy(act_obj_state_cooccs)
        num_vcoco_ao_pairs_in_src = torch.from_numpy(num_vcoco_ao_pairs_in_src)
    return oa_adj, act_obj_state_cooccs, num_vcoco_ao_pairs_in_src


def get_cocoa_graphs(cocoa_split: CocoaSplit, source_ds: HoiDataset, to_torch=True, verbose=False, sym=True, ext_interactions=False):
    def _get_cocoa_to_hh_act_mapping():
        if not isinstance(source_ds, HicoDetHake):
            raise NotImplementedError

        hh = source_ds
        we = WordEmbeddings()

        hhe = we.get_embeddings(hh.actions)
        cae = we.get_embeddings(cocoa.actions)

        sims = hhe @ cae.T

        wndict = hh.driver.wn_action_dict
        adict = hh.driver.action_dict
        hhsynsets = [sorted({s.replace('_', ' ') for wnid in adict[a]['wn_ids'] for s in wndict[wnid]['syn']})
                     for a in hh.actions]
        synsims = np.array([[np.mean(we.get_embeddings(hhsynsets[i], del_on_miss=True, verbose=False) @ cae[j, :])
                             if i > 0 else 0
                             for i in range(hh.num_actions)] for j in range(cocoa.num_actions)]
                           ).T

        assignment = [hh.action_index[hh.null_action]]
        for j in range(1, cocoa.num_actions):
            wem_s = sims[:, j]
            syn_s = synsims[:, j]

            wem_best = np.argmax(wem_s[1:]) + 1
            if hh.actions[wem_best] == cocoa.actions[j]:  # same action
                best = wem_best
            else:
                sem_s = (wem_s + syn_s) / 2
                sem_rank = np.argsort(sem_s[1:])[::-1] + 1
                best = sem_rank[0]
            assignment.append(best)
        assert len(assignment) == cocoa.num_actions
        return assignment

    if ext_interactions:
        ext_interactions = get_interactions_from_ext_src(hoi_ds=cocoa_split.full_dataset)
        _ext_objects_per_action = {}
        for a, o in ext_interactions:
            _ext_objects_per_action.setdefault(a, set()).add(o)
        _ext_objects_per_action = {k: sorted(v) for k, v in _ext_objects_per_action.items()}

        def _get_interacting_objects_from_act(_a_src, _a_coco):
            return np.array(_ext_objects_per_action.get(_a_coco, []))
    else:
        def _get_interacting_objects_from_act(_a_src, _a_coco):
            return [hico_to_coco_obj_mapping[o_src] for o_src in np.flatnonzero(source_ds.oa_to_interaction[:, _a_src] >= 0)]

    src_a_o_s_cooccs, src_num_ao_pairs, states = get_states_vars_from_hake(hh=source_ds, sym=sym)

    cocoa = cocoa_split.full_dataset
    cocoa_to_hico_act_mapping = _get_cocoa_to_hh_act_mapping()
    hico_to_coco_obj_mapping = get_obj_mapping(source_ds.objects, coco_to_hico=False)
    coco_to_hico_obj_mapping = get_obj_mapping(source_ds.objects, coco_to_hico=True)

    cocoa_interactions_from_hico = []
    act_obj_state_cooccs = np.zeros((cocoa.num_actions, cocoa.num_objects, src_a_o_s_cooccs.shape[-1]))
    num_cocoa_ao_pairs_in_src = np.zeros((cocoa.num_actions, cocoa.num_objects), dtype=np.int)
    for a_coco, a_coco_str in enumerate(cocoa.actions):
        if a_coco == 0:
            continue

        a_src = cocoa_to_hico_act_mapping[a_coco]
        objs_coco = _get_interacting_objects_from_act(_a_src=a_src, _a_coco=a_coco)

        for o_coco in objs_coco:
            cocoa_interactions_from_hico.append((a_coco, o_coco))

            o_src = coco_to_hico_obj_mapping[o_coco]

            # Get relevant part states
            act_obj_state_cooccs[a_coco, o_coco, :] = src_a_o_s_cooccs[a_src, o_src, :]
            num_cocoa_ao_pairs_in_src[a_coco, o_coco] = src_num_ao_pairs[a_src, o_src]

    interactions = np.unique(np.concatenate([cocoa_split.interactions, np.array(sorted(cocoa_interactions_from_hico))], axis=0), axis=0)

    isolated_actions_inds = np.setdiff1d(np.arange(cocoa.num_actions), np.unique(interactions[:, 0]))
    if verbose:
        print('Isolated actions:', ', '.join([f'{cocoa.actions[i]}{"" if i in cocoa_split.seen_actions else "*"}'
                                              for i in isolated_actions_inds]))
        print('Unvcovered interactions:\n',
              '\n\t'.join([f'{cocoa.actions[a]:25s} {cocoa.objects[o]:20s}'
                           for a, o in sorted({(a, o) for a, o in cocoa.interactions.tolist()} -
                                              {(a, o) for a, o in interactions.tolist()}
                                              )
                           ]))
        print('Interactions:\n',
              '\n\t'.join([f'{cocoa.actions[a]:25s} {cocoa.objects[o]:20s}' for a, o in interactions]))

    oa_adj = np.zeros([cocoa.num_objects, cocoa.num_actions], dtype=np.float32)
    oa_adj[interactions[:, 1], interactions[:, 0]] = 1
    oa_adj[:, 0] = 0
    if verbose:
        _act_obj_state_cooccs = act_obj_state_cooccs.sum(axis=1)
        print('Action-state pairs:\n',
              '\n\t'.join([f'{cocoa.actions[a]:25s} {states[ps]:35s} {_act_obj_state_cooccs[a, ps] * 100:.0f}'
                           for a, ps in np.stack(np.where(_act_obj_state_cooccs), axis=1)]))

    if to_torch:
        oa_adj = torch.from_numpy(oa_adj)
        act_obj_state_cooccs = torch.from_numpy(act_obj_state_cooccs)
        num_cocoa_ao_pairs_in_src = torch.from_numpy(num_cocoa_ao_pairs_in_src)
    return oa_adj, act_obj_state_cooccs, num_cocoa_ao_pairs_in_src


def get_hicococoa_graphs(hc_split: HicoCocoaSplit, to_torch=True, verbose=False, ext_interactions=False, **kwargs):
    hc_ds = hc_split.full_dataset

    if ext_interactions:
        mined_interactions = get_interactions_from_ext_src(hoi_ds=hc_ds)
        interactions = np.unique(np.concatenate([hc_split.interactions, mined_interactions], axis=0), axis=0)
        if verbose:
            check_hoi_coverage(hoi_ds=hc_ds, tr_interactions=hc_split.interactions, ext_interactions=mined_interactions)
    else:
        interactions = hc_ds.interactions

    oa_adj = np.zeros([hc_ds.num_objects, hc_ds.num_actions], dtype=np.float32)
    oa_adj[interactions[:, 1], interactions[:, 0]] = 1
    oa_adj[:, 0] = 0
    if to_torch:
        oa_adj = torch.from_numpy(oa_adj)
    return oa_adj, None, None


def main():
    test = 4
    if test == 0:
        raise NotImplementedError
    elif test == 1:
        raise NotImplementedError
    elif test == 2:
        vcoco = VCoco()
        hh = HicoDetHake()
        d = pickle.load(open(f'zero-shot_inds/vcoco_seen_inds_{1}.pkl.push', 'rb'))
        ainds = d['train']['act']
        oinds = d['train'].get('obj', np.arange(vcoco.num_objects))
        coco_split = VCocoSplit(split='train', full_dataset=vcoco, action_inds=ainds, object_inds=oinds)

        sym = True
        _oa, _, _ = get_vcoco_graphs(vcoco_split=coco_split, source_ds=hh, verbose=True, sym=sym, to_torch=False,
                                     ext_interactions=True
                                     )
    elif test == 3:
        cocoa = Cocoa()
        hh = HicoDetHake()
        coco_split = CocoaSplit(split='train', full_dataset=cocoa)

        sym = True
        _oa, _, _ = get_cocoa_graphs(cocoa_split=coco_split, source_ds=hh, verbose=True, sym=sym, to_torch=False,
                                     ext_interactions=True
                                     )
    elif test == 4:
        from analysis.utils import plt, plot_mat

        vcoco = VCoco()
        hh = HicoDetHake()
        d = pickle.load(open(f'zero-shot_inds/vcoco_seen_inds_{1}.pkl.push', 'rb'))
        ainds = d['train']['act']
        oinds = d['train'].get('obj', np.arange(vcoco.num_objects))
        coco_split = VCocoSplit(split='train', full_dataset=vcoco, action_inds=ainds, object_inds=oinds)

        sym = True
        _oa, aos_cooccs, num_ao_pairs = get_vcoco_graphs(vcoco_split=coco_split, source_ds=hh, verbose=True, sym=sym, to_torch=False)
        as_cooccs = aos_cooccs.sum(axis=1)
        num_acts = num_ao_pairs.sum(axis=1)
        ao = _oa.T

        thr = 0.5
        as_r_act = as_cooccs / np.maximum(1, num_acts[:, None])
        as_c = as_cooccs / np.maximum(1, as_cooccs.sum(axis=0, keepdims=True))
        ao_r = ao / np.maximum(1, ao.sum(axis=1, keepdims=True))
        ao_c = ao / np.maximum(1, ao.sum(axis=0, keepdims=True))

        # ############## PART STATES
        use_interactions = False

        if not use_interactions:
            nec = as_r_act > thr
            suf = as_c > thr

            print('#' * 100)
            states = hh.symstates if sym else hh.states
            for zsi in np.setdiff1d(np.arange(vcoco.num_actions), ainds):
                pstates_with_cond = {}
                for n in np.flatnonzero(nec[zsi, :]):
                    pstates_with_cond.setdefault(n, []).append('N')
                for s in np.flatnonzero(suf[zsi, :]):
                    pstates_with_cond.setdefault(s, []).append('S')
                print(f'{vcoco.actions[zsi]:25s}:', ', '.join([f'{states[k]} ({"".join(pstates_with_cond[k])})'
                                                               for k in sorted(pstates_with_cond.keys())]))

            # Rows normalised by the total number of actions, so element (i, j) is the percentage of time action i co-occurs with state j.
            plot_mat(as_r_act,
                     xticklabels=hh.symstates if sym else hh.states,
                     yticklabels=vcoco.actions,
                     zero_color=[1, 1, 1],
                     alternate_labels=False,
                     vrange=(0, 1),
                     annotate=False,
                     figsize=(20, 10),
                     plot=False
                     )

            plot_mat(as_c,
                     xticklabels=hh.symstates if sym else hh.states,
                     yticklabels=vcoco.actions,
                     zero_color=[1, 1, 1],
                     alternate_labels=False,
                     vrange=(0, 1),
                     annotate=False,
                     figsize=(20, 10),
                     plot=False
                     )

            plot_mat(nec.astype(np.float) / 3 + suf.astype(np.float) / 3 * 2,
                     xticklabels=hh.symstates if sym else hh.states,
                     yticklabels=vcoco.actions,
                     zero_color=[1, 1, 1],
                     alternate_labels=False,
                     vrange=(0, 1),
                     annotate=False,
                     figsize=(20, 10),
                     plot=False
                     )
        else:
            interactions = vcoco.interactions
            # interaction_str = [f'{vcoco.actions[a] if i == 0 or a != interactions[i - 1, 0] else "-" * len(vcoco.actions[a])} {vcoco.objects[o]}'
            #                    for i, (a, o) in enumerate(interactions)]
            interaction_str = vcoco.interactions_str
            i_s_cooccs = aos_cooccs[interactions[:, 0], interactions[:, 1], :]  # C x S_sym
            num_inters = num_ao_pairs[interactions[:, 0], interactions[:, 1]]  # C

            s_i_norm_over_s = (i_s_cooccs / np.maximum(1, num_inters[:, None])).T
            s_i_norm_over_ao = (i_s_cooccs / np.maximum(1, i_s_cooccs.sum(axis=0, keepdims=True))).T
            nec = s_i_norm_over_s
            suf = s_i_norm_over_ao

            plot_mat(nec,
                     xticklabels=interaction_str,
                     yticklabels=hh.symstates if sym else hh.states,
                     zero_color=[1, 1, 1],
                     alternate_labels=True,
                     vrange=(0, 1),
                     annotate=False,
                     figsize=(20, 10),
                     plot=False
                     )

            plot_mat(suf,
                     xticklabels=interaction_str,
                     yticklabels=hh.symstates if sym else hh.states,
                     zero_color=[1, 1, 1],
                     alternate_labels=True,
                     vrange=(0, 1),
                     annotate=False,
                     figsize=(20, 10),
                     plot=False
                     )

            plot_mat((s_i_norm_over_s > thr).astype(np.float) / 3 + (s_i_norm_over_ao > thr).astype(np.float) / 3 * 2,
                     xticklabels=interaction_str,
                     yticklabels=hh.symstates if sym else hh.states,
                     zero_color=[1, 1, 1],
                     alternate_labels=True,
                     vrange=(0, 1),
                     annotate=False,
                     figsize=(20, 10),
                     plot=False
                     )

        # ################### OBJECTS
        # plot_mat(ao_r,
        #          xticklabels=vcoco.objects,
        #          yticklabels=vcoco.actions,
        #          neg_color=[1, 1, 1],
        #          zero_color=[1, 0, 1],
        #          alternate_labels=False,
        #          vrange=(0, 1),
        #          annotate=max(as_r_act.shape) <= 60,
        #          figsize=(20, 10),
        #          plot=False
        #          )
        #
        # plot_mat(ao_c,
        #          xticklabels=vcoco.objects,
        #          yticklabels=vcoco.actions,
        #          neg_color=[1, 1, 1],
        #          zero_color=[1, 0, 1],
        #          alternate_labels=False,
        #          vrange=(0, 1),
        #          annotate=False,
        #          figsize=(20, 10),
        #          plot=False
        #          )

        # ao_r = (ao_r > thr).astype(np.float)
        # ao_c = (ao_c > thr).astype(np.float)
        # plot_mat(ao_r / 3 + ao_c / 3 * 2,
        #          xticklabels=vcoco.objects,
        #          yticklabels=vcoco.actions,
        #          neg_color=[1, 1, 1],
        #          zero_color=[1, 0, 1],
        #          alternate_labels=False,
        #          vrange=(0, 1),
        #          annotate=False,
        #          figsize=(20, 10),
        #          plot=False
        #          )

        # # Normalise over both rows and columns
        # ap_rc = 1 / np.sqrt(ap.sum(axis=0, keepdims=True)) * ap * 1 / np.sqrt(ap.sum(axis=1, keepdims=True))
        # ap_rc[np.isnan(ap_rc)] = 0
        # plot_mat(ap_rc,
        #          xticklabels=hh.symstates if sym else hh.states,
        #          yticklabels=vcoco.actions,
        #          neg_color=[1, 1, 1],
        #          zero_color=[1, 0, 1],
        #          alternate_labels=False,
        #          vrange=(0, 1),
        #          annotate=max(ap.shape) <= 60,
        #          figsize=(20, 10),
        #          plot=False
        #          )

        plt.show()


if __name__ == '__main__':
    main()

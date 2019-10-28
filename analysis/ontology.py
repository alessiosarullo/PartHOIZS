import json
import numpy as np
import h5py
import os


def load_vg_sgg_dicts():
    with open(os.path.join('data', 'VG-SGG', 'VG-SGG-dicts.json'), 'r') as f:
        dbdict = json.load(f)
        idx_to_pred_dict = dbdict['idx_to_predicate']
        idx_to_label_dict = dbdict['idx_to_label']
        pred_to_idx = dbdict['predicate_to_idx']
        label_to_idx = dbdict['label_to_idx']
        assert '0' not in idx_to_pred_dict and '0' not in idx_to_label_dict
        idx_to_pred = ['__no_rel__'] + [idx_to_pred_dict[str(i + 1)] for i in range(len(idx_to_pred_dict))]
        idx_to_label = ['__background__'] + [idx_to_label_dict[str(i + 1)] for i in range(len(idx_to_label_dict))]
    return idx_to_pred, idx_to_label, pred_to_idx, label_to_idx


def load_allowed_rels(version):
    p2i, l2i = load_vg_sgg_dicts()[2:]

    print('Results version: v%d.' % version)
    with open('data/Ontology/RESULTSv%d' % version, 'r') as f:
        fc = f.readlines()
        blanks = [i for i, l in enumerate(fc) if l.strip() == '']
        assert len(blanks) == 2
        kb_rel_lines = fc[blanks[0] + 1:blanks[1]]
        kb_rels = np.ones((151, 151, 51), dtype=np.bool)
        for l in kb_rel_lines:
            s, p_str, o, b = l.split()
            s = s.replace('_', ' ')
            p_str = p_str.replace('_', ' ')
            o = o.replace('_', ' ')
            b = (b == 'true')
            si = l2i.get(s, 0)
            pi = p2i[p_str]
            oi = l2i.get(o, 0)
            kb_rels[si, oi, pi] = b
        print('Number absurd relations: %d.' % np.sum(kb_rels == 0))
        kb_rels[0, :, :] = False
        kb_rels[:, 0, :] = False
    return kb_rels


def read_from_gt(split, image_index=None):
    with h5py.File(os.path.join('data', 'VG-SGG', 'VG-SGG.h5'), 'r') as db:
        img_split = db['_data_split'][:]
        entity_classes = np.squeeze(db['labels'][:])
        preds = np.squeeze(db['actions'][:])
        rels = np.squeeze(db['relationships'][:])
        ifr, ilr = np.squeeze(db['img_to_first_rel'][:]), np.squeeze(db['img_to_last_rel'][:])
        ifb, ilb = np.squeeze(db['img_to_first_box'][:]), np.squeeze(db['img_to_last_box'][:])
        entity_boxes = db['boxes_1024'][:]

        # xc, yc, w, h -> x1, y1, x2, y2
        entity_boxes[:, :2] = entity_boxes[:, :2] - entity_boxes[:, 2:] // 2
        entity_boxes[:, 2:] = entity_boxes[:, :2] + entity_boxes[:, 2:]

        print('Total rels: %d.' %
              sum([l - f + 1 for s, f, l in zip(img_split, ifr, ilr) if s == split and f >= 0 and l >= 0]))

        im_split_idxs = np.flatnonzero(img_split == split)
        if image_index is not None:
            assert len(set(image_index) - set(im_split_idxs)) == 0  # The given index has to be a subset of the _data_split.
            print('Dumping %d images.' % len(set(im_split_idxs) - set(image_index)))
            im_split_idxs = image_index

        print('Total rels in indexed images: %d.' %
              sum([l - f + 1 for f, l in zip(ifr[im_split_idxs], ilr[im_split_idxs]) if f >= 0 and l >= 0]))

        rel_split_mask = np.zeros_like(preds, dtype=np.bool)
        rel_to_split_img = -np.ones_like(preds, dtype=np.int)
        rels_per_split_img = {}
        im_w_rel_idxmask = np.zeros_like(im_split_idxs, dtype=np.bool)  # Images with relationships in this _data_split
        for i, im_i in enumerate(im_split_idxs):
            if ifr[im_i] >= 0:  # filter images with no relationships
                assert ifb[im_i] >= 0
                im_w_rel_idxmask[i] = 1
                rel_split_mask[ifr[im_i]:ilr[im_i] + 1] = 1
                rel_to_split_img[ifr[im_i]:ilr[im_i] + 1] = im_i
                rels_per_split_img[im_i] = set(range(ifr[im_i], ilr[im_i] + 1))
        assert np.all((rel_split_mask == 0) == (rel_to_split_img == -1))
        assert np.all(rel_to_split_img[rel_split_mask] >= 0)

        del rels_per_split_img
        final_preds = preds[rel_split_mask]
        final_rels = rels[rel_split_mask, :]
        final_rel_to_img = rel_to_split_img[rel_split_mask]
        rel_classes = np.stack((entity_classes[final_rels[:, 0]],
                                final_preds,
                                entity_classes[final_rels[:, 1]]), axis=1)
        rel_boxes = np.concatenate((entity_boxes[final_rels[:, 0], :],
                                    entity_boxes[final_rels[:, 1], :]), axis=1)

        _final_im2last = np.concatenate([np.flatnonzero(final_rel_to_img[:-1] != final_rel_to_img[1:]), [len(final_rel_to_img) - 1]])
        _final_im2first = np.concatenate([[0], _final_im2last[:-1] + 1])
        final_im2last = -np.ones_like(im_split_idxs)
        final_im2last[im_w_rel_idxmask] = _final_im2last
        final_im2first = -np.ones_like(im_split_idxs)
        final_im2first[im_w_rel_idxmask] = _final_im2first
        gt_dict = {'rel_classes': rel_classes,
                   'rel_scores': np.ones((rel_classes.shape[0],)),
                   'rel_boxes': rel_boxes,
                   'i2first_rel': final_im2first.astype(ifr.dtype),
                   'i2last_rel': final_im2last.astype(ilr.dtype),
                   'i2first_box': ifb[im_split_idxs],
                   'i2last_box': ilb[im_split_idxs],
                   'entity_classes': entity_classes,
                   'entity_boxes': entity_boxes
                   }

    return gt_dict


def check_on_kb(rels, pred_class_index, box_class_index):
    kb_rels = load_allowed_rels(VERSION)

    inds = []
    for i, (s, p, o) in enumerate(rels):
        if not kb_rels[s, o, p]:
            inds.append(i)
    absurd_rels, absurd_rel_counts = np.unique(rels[np.array(inds), :], axis=0, return_counts=True)
    print('Relationship instances to be filtered: %d/%d.' % (len(inds), rels.shape[0]))
    print('Relationships triplets to be filtered: %d/%d.' % (len(absurd_rels), len(np.unique(rels, axis=0))))
    num_to_filter_per_rel = []
    for p, p_str in enumerate(pred_class_index):
        inds = (absurd_rels[:, 1] == p)
        num_to_filter_per_rel.append(np.sum(inds))
        print('{0} {1} ({3} | u {2}) {0}'.format('=' * 20, p_str, num_to_filter_per_rel[-1], np.sum(absurd_rel_counts[inds])))
        tmp = np.concatenate([absurd_rels[inds], absurd_rel_counts[:, None][inds, :]], axis=1)
        print('\n'.join(['%s %s %s (%d)' % (box_class_index[s], pred_class_index[pr], box_class_index[o], n) for s, pr, o, n in tmp]))
    print('#' * 80)
    print('Summary (triplets):')
    print('\n'.join(['%-15s %5d' % (pred_class_index[i], n) for i, n in enumerate(num_to_filter_per_rel) if n > 0]))
    print('-' * (15 + 1 + 5))
    print('%-15s %5d' % ('Total', sum(num_to_filter_per_rel)))

    num_most_common = 100
    inds = np.argsort(absurd_rel_counts)[::-1][:num_most_common]
    tmp = np.concatenate([np.arange(num_most_common)[:, None], absurd_rels[inds], absurd_rel_counts[:, None][inds, :]], axis=1)
    print('\nTop %d most common:' % num_most_common)
    print('\n'.join(['%4d) [%4d] %s %s %s' % (i + 1, n, box_class_index[s], pred_class_index[pr], box_class_index[o]) for i, s, pr, o, n in tmp]))

    print('\n\nResults version: v%d.' % VERSION)
    assert np.sum(np.array(num_to_filter_per_rel)) == absurd_rels.shape[0]


def main():
    pred_class_index, box_class_index = load_vg_sgg_dicts()[:2]
    gt_dict = read_from_gt(split=0)
    check_on_kb(gt_dict['rel_classes'], pred_class_index, box_class_index)


def stats():
    gt_pred_classes, gt_object_classes = load_vg_sgg_dicts()[:2]
    ont_object_classes = set()
    with open('data/Ontology/RESULTSv%d' % VERSION, 'r') as f:
        fc = f.readlines()
        blanks = [i for i, l in enumerate(fc) if l.strip() == '']
        assert len(blanks) == 2
        kb_rel_lines = fc[blanks[0] + 1:blanks[1]]
        for l in kb_rel_lines:
            s, p_str, o, b = l.split()
            s = s.replace('_', ' ')
            p_str = p_str.replace('_', ' ')
            o = o.replace('_', ' ')
            b = (b == 'true')
            ont_object_classes.add(s)
            ont_object_classes.add(o)
    missing_object_classes = sorted(set(gt_object_classes) - ont_object_classes)
    print(len(missing_object_classes), '\n'.join(missing_object_classes))


if __name__ == '__main__':
    VERSION = 5
    main()
    # stats()

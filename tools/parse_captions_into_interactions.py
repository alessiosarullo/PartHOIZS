import argparse
import json
import os
import pickle
import sys

from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from config import cfg
from lib.dataset.hico import Hico


def parse_captions(captions):
    # If not downlowaded already, run:
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('universal_tagset')
    # nltk.download('wordnet')

    # stop_words = stopwords.words('english') + list(get_stop_words('en'))
    person_words = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',
                    'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend', 'guard', 'little girl',
                    'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child', 'snowboarder', 'surfer', 'tennis player'}

    hd = Hico()
    preds = hd.actions
    pred_verbs = [preds[0]] + [p.split('_')[0] for p in preds[1:]]
    predset = set(pred_verbs)
    objs_per_pred = {p: set() for p in preds}
    for i_cap, caption in enumerate(captions):
        if i_cap % 1000 == 0:
            print(i_cap)

        tokens = word_tokenize(caption)
        tagged_tokens = pos_tag(tokens, tagset='universal')

        person_found = False
        for i_tokens, w in enumerate(tokens):
            if wn.morphy(w, wn.NOUN) in person_words:
                person_found = True
            else:
                verb = wn.morphy(w, wn.VERB)
                if verb in predset:
                    break
        else:
            continue

        if not person_found or i_tokens + 1 == len(tokens):
            continue

        following_tagged_tokens = tagged_tokens[i_tokens + 1:]
        if following_tagged_tokens[0][1] == 'ADP':
            preposition = following_tagged_tokens[0][0]
            following_tagged_tokens = following_tagged_tokens[1:]
        else:
            preposition = None

        p_obj_sentence = []
        for w, pos in following_tagged_tokens:
            if pos == 'NOUN' and w not in ['front', 'top']:
                p_obj_sentence.append(w)
            elif pos in {'NOUN', 'PRON', 'ADJ', 'DET'}:
                continue
            else:
                break

        if not p_obj_sentence:
            continue
        else:
            p_obj = ' '.join(p_obj_sentence)

        for i_pred, orig_p in enumerate(preds):
            if pred_verbs[i_pred] == verb:
                p_tokens = orig_p.split('_')
                if len(p_tokens) > 1 and (preposition is None or preposition != p_tokens[1]):
                    continue
                objs_per_pred[orig_p].add(p_obj)

    for p, objs in objs_per_pred.items():
        print('%20s:' % p, sorted(objs))

    return objs_per_pred


def vg():
    data_dir = os.path.join(cfg.data_root, 'VG')
    try:
        with open(os.path.join(cfg.cache_root, 'vg_region_descriptions.txt'), 'r') as f:
            region_descr = [l.strip() for l in f.readlines()]
    except FileNotFoundError:
        region_descr = json.load(open(os.path.join(data_dir, 'region_descriptions.json'), 'r'))
        region_descr = [r['phrase'] for rd in region_descr for r in rd['regions']]
        with open(os.path.join(cfg.cache_root, 'vg_region_descriptions.txt'), 'w') as f:
            f.write('\n'.join(region_descr))
    print('\n'.join(region_descr[:10]))
    print()

    objs_per_pred = parse_captions(region_descr)
    with open(os.path.join(cfg.cache_root, 'vg_predicate_objects.pkl'), 'wb') as f:
        pickle.dump(objs_per_pred, f)
    print(len(region_descr))


def vcap():
    d = json.load(open(os.path.join(cfg.data_root, 'VideoCaptions', 'train.json'), 'r'))
    captions = [s for v in d.values() for s in v['sentences']]
    print('\n'.join(captions[:10]))
    print()

    objs_per_pred = parse_captions(captions)
    # with open(os.path.join(cfg.cache_root, 'vcap_predicate_objects.pkl'), 'wb') as f:
    #     pickle.dump(objs_per_pred, f)
    # print(len(captions))


def main():
    funcs = {
        'vg': vg,
        'vcap': vcap,
    }
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=funcs.keys())
    namespace = parser.parse_known_args()
    func = vars(namespace[0])['func']
    sys.argv = sys.argv[:1] + namespace[1]
    print(sys.argv)

    cfg.parse_args(fail_if_missing=False)
    funcs[func]()


if __name__ == '__main__':
    main()

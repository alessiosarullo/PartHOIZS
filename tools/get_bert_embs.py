import argparse
import pickle

import torch
from transformers import *

from lib.dataset.hico_hake import HicoHake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if args.debug:
        try:  # PyCharm debugging
            print('Starting remote debugging (resume from debug server)')
            import pydevd_pycharm
            pydevd_pycharm.settrace('130.88.195.105', port=16008, stdoutToServer=True, stderrToServer=True)
            print('Remote debugging activated.')
        except:
            print('Remote debugging failed.')
            raise

    hh = HicoHake()
    sentences = [f'People {hh.actions[a]} the {hh.objects[o]}' for a, o in hh.interactions if hh.actions[a] != hh.null_action]
    cache_fn = 'cache/bert_hoi_embs.pkl'

    try:
        with open(cache_fn, 'rb') as f:
            last_hidden_states, input_ids = pickle.load(f)
    except FileNotFoundError:
        # Transformers has a unified API for 10 transformer architectures and 30 pretrained weights.
        #          Model          | Tokenizer          | Pretrained weights shortcut
        MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased'),
                  (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
                  (GPT2Model, GPT2Tokenizer, 'gpt2'),
                  (CTRLModel, CTRLTokenizer, 'ctrl'),
                  (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
                  (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
                  (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
                  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
                  (RobertaModel, RobertaTokenizer, 'roberta-base'),
                  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
                  ]
        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the
        # PyTorch model `RobertaModel`

        # Load pretrained model/tokenizer
        model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, 'bert-base-uncased'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        # Encode text
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        with torch.no_grad():
            input_ids = []
            last_hidden_states = []
            for s in sentences:
                input_ids_s = torch.tensor([tokenizer.encode(s, add_special_tokens=True)])
                output = model(input_ids_s)

                input_ids.append(input_ids_s)
                last_hidden_states.append(output[0])  # Models outputs are now tuples
        with open(cache_fn, 'wb') as f:
            pickle.dump([last_hidden_states, input_ids], f)
    print('Done.')


if __name__ == '__main__':
    main()

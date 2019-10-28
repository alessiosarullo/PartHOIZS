import os
import pickle
import re

import numpy as np

from config import cfg


class WordEmbeddings:
    def __init__(self, source='numberbatch', dim=None, normalize=False):
        """
        Attributes:
            embeddings: [array] NxD matrix consisting of N D-dimensional embeddings
            vocabulary: [list] the N words composing the vocabulary, sorted according to `embeddings`'s rows
        """

        self.loaders = {'numberbatch': {'parser': self.parse_numberbatch,
                                        'src_file': 'numberbatch-en.txt',
                                        },
                        'glove': {'parser': self.parse_glove,
                                  'src_file': 'glove.6B.%dd.txt',
                                  'default_dim': 300,
                                  },
                        'word2vec': {'parser': self.parse_word2vec,
                                     'src_file': 'GoogleNews-vectors-negative300.bin',
                                     },
                        }

        self.source = source

        try:
            dim = dim or self.loaders[source].get('default_dim', None)
            self._embeddings, self.vocabulary = self.load(source.lower(), dim)
        except KeyError:
            raise ValueError('Unknown source %s. Possible sources:' % source, list(self.loaders.keys()))
        if normalize:
            norms = np.linalg.norm(self._embeddings, axis=1)
            norms[norms == 0] = 1
            self._embeddings /= norms[:, None]
            self.normalised = True
        else:
            self.normalised = False

        self.word_index = {v: i for i, v in enumerate(self.vocabulary)}
        self.dim = self._embeddings.shape[1]

    def embedding(self, word: str, none_on_miss=True):
        try:
            if word == 'hair drier':
                word = 'hair dryer'
            idx = self.word_index[word]
        except KeyError:
            if none_on_miss:
                return None
            else:
                return np.zeros_like(self._embeddings[0, :])
        return self._embeddings[idx, :]

    def get_embeddings(self, words, retry='avg'):
        # vectors = np.random.standard_normal((len(words), self.dim))  # This is what they do in NeuralMotifs, but I'm not sure it's a good idea.
        vectors = np.zeros((len(words), self.dim))
        fails = []
        replacements = {}
        for i, word in enumerate(words):
            if not word.startswith('_'):
                word = word.replace('_', ' ').strip()
            emb = self.embedding(word, none_on_miss=True)
            if emb is None and retry and not word.startswith('_'):
                tokens = word.split(' ')
                if retry == 'longest':
                    repl_word = sorted(tokens, key=lambda x: len(x), reverse=True)[0]
                    replacements[word] = repl_word
                    emb = self.embedding(repl_word, none_on_miss=True)
                elif retry == 'first':
                    repl_word = tokens[0]
                    replacements[word] = repl_word
                    emb = self.embedding(repl_word, none_on_miss=True)
                elif retry == 'last':
                    repl_word = tokens[-1]
                    replacements[word] = repl_word
                    emb = self.embedding(repl_word, none_on_miss=True)
                elif retry == 'avg':
                    if not self.normalised:
                        raise ValueError('Average embeddings requires normalisation.')
                    embs = [self.embedding(token, none_on_miss=True) for token in tokens]
                    if embs:
                        emb = sum(embs) / len(embs)
                        emb_norm = np.linalg.norm(emb)
                        emb = emb / emb_norm if emb_norm > 0 else emb
                    else:
                        emb = None
                else:
                    raise ValueError(retry)

            if emb is not None:
                vectors[i] = emb
            else:
                fails.append(word)

        if retry:
            for w in fails:
                if not w.startswith('_'):
                    del replacements[w]
        if replacements:
            print('These words were not found and have been replaced: %s.' % ', '.join(['%s -> %s' % (k, v) for k, v in replacements.items()]))
        if fails:
            print('Default embedding will be used for %s.' % ', '.join(fails))
        return vectors.astype(np.float32, copy=False)

    def load(self, source, dim):
        src_fn = self.loaders[source]['src_file']
        if dim is not None:
            src_fn = src_fn % dim
        path_cache_file = os.path.join(cfg.cache_root, os.path.splitext(src_fn)[0] + '_cache.pkl')
        try:
            with open(path_cache_file, 'rb') as f:
                print('Loading cached %s embeddings.' % source)
                embedding_mat, vocabulary = pickle.load(f)
        except FileNotFoundError:
            print('Parsing and caching %s embeddings.' % source)
            embedding_mat, vocabulary = self.loaders[source]['parser'](os.path.join(cfg.embedding_dir, src_fn))
            # Cleaning
            clean_words_inds = [i for i, word in enumerate(vocabulary) if not bool(re.search(r"[^a-zA-Z0-9_'\-]", word))]
            vocabulary = [vocabulary[i].replace('_', ' ').strip() for i in clean_words_inds]
            embedding_mat = embedding_mat[clean_words_inds, :]
            with open(path_cache_file, 'wb') as f:
                pickle.dump((embedding_mat, vocabulary), f)
        return embedding_mat, vocabulary

    @staticmethod
    def parse_glove(src_file):
        with open(src_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

        embeddings, vocabulary = [], []
        for i, line in enumerate(lines):
            tokens = line.split()
            embeddings.append(np.array([float(x) for x in tokens[1:]]))
            vocabulary.append(tokens[0])

        embedding_mat = np.stack(embeddings, axis=0)
        return embedding_mat, vocabulary

    @staticmethod
    def parse_numberbatch(src_file):
        """
        Format (from https://github.com/commonsense/conceptnet-numberbatch):
            The first line of the file contains the dimensions of the matrix:
                1984681 300
            Each line contains a term label followed by 300 floating-point numbers, separated by spaces:
                /c/en/absolute_value -0.0847 -0.1316 -0.0800 -0.0708 -0.2514 -0.1687 -...
                /c/en/absolute_zero 0.0056 -0.0051 0.0332 -0.1525 -0.0955 -0.0902 0.07...
                /c/en/absoluteless 0.2740 0.0718 0.1548 0.1118 -0.1669 -0.0216 -0.0508...
                /c/en/absolutely 0.0065 -0.1813 0.0335 0.0991 -0.1123 0.0060 -0.0009 0...
                /c/en/absolutely_convergent 0.3752 0.1087 -0.1299 -0.0796 -0.2753 -0.1...
        :return: see __init__
        """

        with open(src_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

        embedding_mat = np.zeros([int(dim) for dim in lines[0].split()])
        vocabulary = []
        for i, line in enumerate(lines[1:]):
            tokens = line.split()
            embedding_mat[i, :] = np.array([float(x) for x in tokens[1:]])

            # This is needed if words start with '/c/LANGUAGE_TAG/'. It's not the case in the english-only version.
            # word_id_tokens = tokens[0].split('/')
            # assert len(word_id_tokens) == 4, (word_id_tokens, i, line)
            # vocabulary.append(word_id_tokens[-1])

            vocabulary.append(tokens[0])

        return embedding_mat, vocabulary

    @staticmethod
    def parse_word2vec(src_file):
        """
        `model`'s parameters are:
            - vectors [ndarray]: N x vector_size vector of embeddings. N = 3 billion.
            - vocab [dict(str, Vocab)]: keys are words (the same in `index2word`). Values are objects that have two attributes: 'count' and
                'index'. Seems like, for some reason, count + index = len(`vocab`), always. Don't understand what they're supposed to represent.
            - vector_size [int]: it's 300.
            - index2word [list(str)]: list of words.
            - vectors_norm [None]: don't know why it's None. I guess it's here for caching reasons.
        """
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(src_file, binary=True)
        embedding_mat = model.vectors
        vocabulary = model.index2word
        return embedding_mat, vocabulary


def main():
    # we = WordEmbeddings(source='glove', normalize=True, dim=200)
    we = WordEmbeddings(source='word2vec', normalize=True)
    e = we.get_embeddings(['ride', 'chair', 'bike'])
    print(e[0].dot(e[1]), e[0].dot(e[2]))


if __name__ == '__main__':
    main()

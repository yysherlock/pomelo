import sys,os
import numpy as np
from six.moves import xrange


class PlainCorpus(object):
    def __init__(self, corpus_fn):
        self.corpus_fn = corpus_fn
        self.constructed = False

    def constructed(self):
        self.corpus_size = self.get_corpus_length()
        self.constructed = True

    def get_corpus_length(self):
        length = 0
        with open(self.corpus_fn) as f:
            for line in f: length += 1
        return length

    def sample_corpus(self, sample_size, sample_fn):
        """
        Random sample some sentences from plain text corpus, which consists one sentences
        each line.
        Args:
            corpus_size: #sentences of the corpus
            sample_size: #sentences you want to sample
        """
        corpus_size = self.get_corpus_length() if not self.constructed else self.corpus_size
        x = sorted(np.random.choice(corpus_size, sample_size, replace=False))

        with open(self.corpus_fn) as f, open(sample_fn, 'w') as outf:
            cnt = 0
            line = f.readline()
            for i in xrange(sample_size):
                while cnt < i:
                    line = f.readline()
                    cnt += 1
                outf.write(line)

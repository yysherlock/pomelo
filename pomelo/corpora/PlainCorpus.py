import sys,os
import re
import random
import numpy as np
from six.moves import xrange
from gensim import utils
from collections import defaultdict

class PlainCorpus(object):
    def __init__(self, corpus_fn):
        if not os.path.exists(corpus_fn):
            raise FileNotFoundError("Not Found this '{}' corpus! Please check!".format(corpus_fn))
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

    def corpus_split(self, fnames, props):
        size = self.corpus_size if self.constructed else self.get_corpus_length()
        sids = list(range(size))
        random.shuffle(sids)
        start = 0
        d = defaultdict(int)
        for i,num in enumerate([int(p*size) for p in props]):
            for k in sids[start:start+num]: d[k] = i
            start += num
        with open(self.corpus_fn) as f:
            ff = [open(fn,'w') for fn in fnames]
            for i,line in enumerate(f):
                ff[d[i]].write(line)
            for fl in ff: fl.close()

    def create_vocab(self, fname):
        wordset = set()
        with open(self.corpus_fn) as f, open(fname,'w') as outf:
            for line in f:
                for token in line.strip().split():
                    wordset.add(token)
            for word in wordset: outf.write(word+"\n")

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

    def tokenize(self,content, BytesOrNot=False):
        """
        Tokenize a piece of text.
        Return list of tokens as utf8 bytestrings. Ignore words shorted than 2 or longer
        that 15 characters (not bytes!).
        """
    # https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py#L166
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
        if BytesOrNot:
            return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore')
                if 2 <= len(token) <= 15 and not token.startswith('_')] # return a list of bytes of characters
        else: return list(utils.tokenize(content, lower=True, errors='ignore')) # return a list of strings

    def get_texts(self):
        PAT_DG = re.compile('(\d+,\d+)|(\d+\.\d+)|(\d+)', re.UNICODE)
        DIGIT = "dddggg"
        with open(self.corpus_fn) as f:
            for line in f: # for each sentence
                line = self.replace_tag(line, PAT_DG, DIGIT)
                yield self.tokenize(line)

    def replace_tag(self, line, pat, tag):
        for match in pat.finditer(line.strip()):
            line = line.replace(match.group(), tag)
        return line
    # (\d+,\d+)|(\d+\.\d+)|(\d+)

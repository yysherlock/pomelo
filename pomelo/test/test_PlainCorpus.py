import logging
import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from pomelo.corpora import PlainCorpus

output_dir = '/home/luozhiyi/projects/WhatHappen/data/'

class PlainCorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.big_corpus = PlainCorpus.PlainCorpus('/home/luozhiyi/data/causal/Causal_Sentences.txt')
        self.small_corpus = PlainCorpus.PlainCorpus('/home/luozhiyi/projects/WhatHappen/data/sampled_causal_sentences.txt')

    def test_sample_plain_corpus(self):
        corpus = self.corpus
        sample_fn = output_dir + 'sampled_causal_sentences.txt'
        sample_size = 100000
        corpus.sample_corpus(sample_size, sample_fn)
        assert os.path.exists(sample_fn) == True
        assert PlainCorpus.PlainCorpus(sample_fn).get_corpus_length() == sample_size

    def test_tokenize(self):
        with open('pp.txt') as f:
            for line in f:
                print(self.small_corpus.tokenize(line))

    def test_get_texts(self):
        DIGIT = "dddggg"
        outfn = 'processed1.txt'
        with open(outfn,'w') as outf:
            for text in self.small_corpus.get_texts():
                outf.write(' '.join(text).replace(DIGIT, "DG"))
                outf.write('\n')

    def test_corpus_split(self):
        processed_corpus = PlainCorpus.PlainCorpus('/home/luozhiyi/projects/WhatHappen/data/processed.txt')
        props = [0.05, 0.05, 0.9]
        output_dir = "/home/luozhiyi/projects/WhatHappen/data/"
        fnames = [output_dir+"ptb.test.txt", output_dir+"ptb.valid.txt", output_dir+"ptb.train.txt"]
        processed_corpus.corpus_split(fnames, props)

    def test_create_vocab(self):
        processed_corpus = PlainCorpus.PlainCorpus('/home/luozhiyi/projects/WhatHappen/data/processed.txt')
        output_dir = "/home/luozhiyi/projects/WhatHappen/data/"
        processed_corpus.create_vocab(output_dir+"vocab.ptb.txt")

def get_suite(cls, test_names):
    suite = unittest.TestSuite()
    for test_name in test_names: suite.addTest(cls(test_name))
    #suite.addTest(PlainCorpusTestCase('test_sample_plain_corpus'))
    return suite

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    #unittest.main()
    runner = unittest.TextTestRunner()
    suite1 = get_suite(PlainCorpusTestCase, ['test_sample_plain_corpus'])
    suite2 = get_suite(PlainCorpusTestCase, ['test_tokenize'])
    suite3 = get_suite(PlainCorpusTestCase, ['test_get_texts'])
    suite4 = get_suite(PlainCorpusTestCase, ['test_corpus_split'])
    suite5 = get_suite(PlainCorpusTestCase, ['test_create_vocab'])
    print(runner.run( suite5 ))

import logging
import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from pomelo.corpora import PlainCorpus

output_dir = '/home/luozhiyi/projects/WhatHappen/data/'
class PlainCorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.corpus = PlainCorpus.PlainCorpus('/home/luozhiyi/data/causal/Causal_Sentences.txt')

    def test_sample_plain_corpus(self):
        corpus = self.corpus
        sample_fn = output_dir + 'sampled_causal_sentences.txt'
        sample_size = 100000
        corpus.sample_corpus(sample_size, sample_fn)
        assert os.path.exists(sample_fn) == True
        assert PlainCorpus.PlainCorpus(sample_fn).get_corpus_length() == sample_size

def get_suite(cls, test_names):
    suite = unittest.TestSuite()
    for test_name in test_names: suite.addTest(cls(test_name))
    #suite.addTest(PlainCorpusTestCase('test_sample_plain_corpus'))
    return suite

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    #unittest.main()
    runner = unittest.TextTestRunner()
    print(runner.run(
            get_suite(PlainCorpusTestCase, ['test_sample_plain_corpus'])
            ))

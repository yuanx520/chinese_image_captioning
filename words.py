import os
import numpy as np
import cPickle as pickle
import codecs
import re

class HzTable():
    def __init__(self, vocab_size, dim_embed, max_sent_len, save_file):
        self.idx2word = []
        self.word2idx = {}
        self.word2vec = {}
        self.word_freq = []
        self.num_words = vocab_size
        self.dim_embed = dim_embed
        self.max_num_words = vocab_size
        self.max_sent_len = max_sent_len 
        self.save_file = save_file
    def filter_word2vec(self):
        """ Remove unseen words from the word embedding. """
        word2vec = {}
        for w in self.word2idx:
            word2vec[w] = self.word2vec[w] 
        self.word2vec = word2vec

    def build(self, freq_file):
        """ Build the vocabulary by selecting the words that occur frequently in the given sentences, and compute the frequencies of these words. """
	freq_wd = np.loadtxt(freq_file,dtype=str)
	idx = 1
        self.word2vec['.'] = 0.01 * np.random.randn(self.dim_embed)
        self.idx2word.append('.')
        self.word2idx['.'] = 0
        self.word_freq.append(int(30000) * 1.0)
	iter_freq = iter(freq_wd)
	for x in iter_freq:
	    word = x
	    freq = next(iter_freq)
            #print freq
            self.word2vec[word] = 0.01 * np.random.randn(self.dim_embed)
	    if int(freq) > 0:
                #self.word2vec[word] = 0.01 * np.random.randn(self.dim_embed)            	
                self.idx2word.append(word)
            	self.word2idx[word] = idx
            	self.word_freq.append(int(freq) * 1.0)
	    idx = idx + 1
        self.num_words=len(self.idx2word)
        self.word_freq /= np.sum(self.word_freq)
        self.word_freq = np.log(self.word_freq)
        self.word_freq -= np.max(self.word_freq)
        self.filter_word2vec()
    
    def symbolize_sent(self, words):
        """ Translate a sentence into the indicies of its words. """
        indices = np.zeros(self.max_sent_len).astype(np.int32)-1
        masks = np.zeros(self.max_sent_len)
        Len=min(self.max_sent_len,len(words))
        """ !!!!  """
        indices[:Len] = np.array(words[:Len])
        indices=indices+1
        masks[:Len] = 1.0
        return indices, masks
    
    def indices_to_sent(self, indices):
        """ Translate a vector of indicies into a sentence. """
        words = [self.idx2word[i] for i in indices]
        if words[-1] != '.':
            words.append('.')
        punctuation = np.argmax(np.array(words) == '.') + 1
        words = words[:punctuation]
        res = ' '.join(words)
        res = res.replace(' ,', ',')
        res = res.replace(' ;', ';')
        res = res.replace(' :', ':')
        res = res.replace(' .', '.')
        return res

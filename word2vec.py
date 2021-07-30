# import modules & set up logging
import logging
import json
import os
from gensim.models import word2vec

vec_size = 5

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('sentence.txt')

model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=vec_size)

model.save(u'sentence.model')

#model_2 = word2vec.Word2Vec.load('sentence.model')
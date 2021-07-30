import logging
import json
import os
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = 'assignment_training_data_word_segment.json'
sentence_list = json.load(open(path, 'r'))

fo = open("sentence.txt", "a", encoding="utf-8")
for i in range(len(sentence_list)):
    for j in range(len(sentence_list[i]['words'])):
        fo.write(sentence_list[i]['words'][j])
        fo.write(' ')
    fo.write('\n')
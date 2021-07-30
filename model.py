import numpy as np
import struct
import time
import json
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence
import lstm_predict
import rule

def predict(data_path, mode) :
    maxlen = 20
    model = load_model('my_model.h5')
    sentence_list = json.load(open(data_path, 'r'))

    preds = rule.predict(data_path)

    x = np.zeros(len(preds)* 100, dtype=list)
    cnt = 0

    if mode == 2 :
        for i in range(len(preds)) :
            if preds[i] != [] :
                for j in range(len(preds[i])) :
                    x[cnt] = sentence_list[i]['indexes'] + preds[i][j]
                    cnt += 1
    elif mode == 3 :
        for i in range(len(sentence_list)) :
            for j in sentence_list[i]['times'] :
                for k in sentence_list[i]['attributes'] :
                    for l in sentence_list[i]['values'] :
                        if not [j, k, l] in preds[i] :
                            x[cnt] = sentence_list[i]['indexes'] + [j, k, l]
                            cnt += 1

    if mode == 2 or mode == 3 :
        x = sequence.pad_sequences(x[:cnt], maxlen=maxlen)
        y = lstm_predict.predict(x, model)

    cnt = 0
    
    if mode == 2 :
        for i in range(len(preds)) :
            if preds[i] != [] :
                k = 0
                for j in range(len(preds[i])) :
                    if y[cnt] == 0 :
                        preds[i].pop(j - k)
                        k += 1
                    cnt += 1
    elif mode == 3 :
        for i in range(len(sentence_list)) :
            for j in sentence_list[i]['times'] :
                for k in sentence_list[i]['attributes'] :
                    for l in sentence_list[i]['values'] :
                        if not [j, k, l] in preds[i] :
                            if y[cnt] == 1 :
                                preds[i].append([j,k,l])
    """
    N = len(sentence_list)
    rightnum=0
    predictnum=0
    for i in range(N):
        for j in range(len(preds[i])):
            predictnum += 1
            if(preds[i][j] in sentence_list[i]['results']):
                rightnum += 1
    resultnum=0
    for i in range(N):
        for j in range(len(sentence_list[i]['results'])):
            resultnum += 1
    rate1=rightnum/predictnum
    print(rightnum,predictnum,resultnum)
    print('正确率（准）：',rate1)
    rate2=rightnum/resultnum
    print('正确率（全）：',rate2)
    print(2*rate1*rate2 / (rate1 + rate2))
    """
    print('Printing results...')
    f_res = open("assignment_test_data_word_segment", 'w')
    for i in range(len(sentence_list)):
        sentence_list[i]['results'] = preds[i]

    json.dump(sentence_list, f_res)
    f_res.close()
    print('Print completed!')
    return preds
    

if __name__ == "__main__":
    print('请输入预测模式：')
    print('\t1.仅基于规则预测')
    print('\t2.使用训练和规则结合预测，优先准确率(p)')
    print('\t3.使用训练和规则结合预测，优先全面率(r)')
    mode = int(input())
    while(mode != 1 and mode != 2 and mode != 3):
        print('请重新输入：')
        print('\t1.仅基于规则预测')
        print('\t2.使用训练和规则结合预测，优先准确率(p)')
        print('\t3.使用训练和规则结合预测，优先全面率(r)')
        mode = int(input())
    predict('assignment_test_data_word_segment.json', mode)


import numpy as np
import struct
import time
import json
import lstm_predict
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence

def found(sentence_index):
    sp_indexes = [14, 22, 24, 32, 88, 270, 284]  #0time,1atteibute,2value, 14为‘占’，2425为括号，32为‘增长’，53为‘比’，270为‘增长率’，284为’降低‘
    key_indexes = [0,1,2,14,22,24, 32, 88, 270, 284]
    main_indexes = [0,1,2]
    predict=[]
    x=sentence_index
    length=len(x)
    time=np.zeros(100)
    t=0
    attribute=np.zeros(100)
    a=0
    value=np.zeros(100)
    v=0
    ctimenum = 0
    cvaluenum = 0
    sp=0
    for i in range(length):
        if(x[i] in sp_indexes):
            sp=1
            break
    #是否有特别需要注意的词组
    if(sp==0):
        for i in range(length):
            if (x[i] == 1):#对每个attribute
                #print('attribute为：',i)
                j=i
                t=0
                v=0
                while(j>=0 and x[j]!=0):
                    j=j-1
                if(j==-1):
                    x[i]=100
                else:
                    while(j>=0 and x[j]!=1 and x[j]!=2):
                        if(x[j]==0):
                            time[t]=j
                            t += 1
                        j=j-1
                #向前找时间
                j=i+1
                while(j<length and x[j]!=0 and x[j]!=1 and x[j]!=2):
                    j += 1
                if(j==length or x[j]!=2):
                    x[i]=100
                else:
                    while(j<length and x[j]!=1 and x[j]!=0):
                        if(x[j]==2):
                            value[v]=j
                            v += 1
                        j += 1
                #向后找value
                if(v>0 and t==v):
                    for k in range(v):
                        predict.append([int(time[t-1-k]),i,int(value[k])])


    else:
        if(x[i]==14 or x[i]==22 or x[i]==32 or x[i]==88 or x[i]==270 or x[i]==284):
            j=i
            while(j<length and x[j]!=2 and x[j]!=0 and x[j]!=1):
                j=j+1
            if(j==length or x[j]==0):
                x[i]=100
                return found(x)
            elif(x[j]==1):
                x[i]=100
                x[j]=100
                while(j<length and x[j]!=0 and x[j]!=1):
                    if(x[j]==2):
                        x[j]=100
                    j += 1
                return found(x)
            else:
                x[i]=100
                while (j < length and x[j] != 0 and x[j] != 1):
                    if (x[j] == 2):
                        x[j] = 100
                    j += 1
                j=i
                while(j>=0 and x[j]!=0 and x[j]!=1 and x[j]!=2):
                    j=j-1
                if(x[j]==1):
                    x[j]=100
                return found(x)
        elif (x[i] == 24) :
            j = i
            while(x[j] != 0 and j > 0) :
                j -= 1
            if (x[j] == 0) :
                x[j] = 100
                x[i] = 100
            return found(x)

    return predict

def predict(path) :
    time_start = time.time()

    print("Loading data...")
    sentence_list = json.load(open(path, 'r'))
    N=np.shape(sentence_list)[0]

    x=sentence_list
    for i in range(N):
        x[i]=sentence_list[i]['indexes']

    sentence_list = json.load(open(path, 'r'))
    y=sentence_list

    """ 
    for i in range(N):
        y[i]=sentence_list[i]['results'] 
    """

    sentence_list = json.load(open(path, 'r'))
    z=sentence_list
    for i in range(N):
        z[i]=[]
    predict_z=z
    time_load_finish = time.time()
    print("Loading complete. Time: " + str(time_load_finish - time_start))

    for i in range(N):
        predict_z[i]=found(x[i])
    """
    rightnum=0
    predictnum=0
    for i in range(N):
        for j in range(len(predict_z[i])):
            predictnum += 1
            if(predict_z[i][j] in y[i]):
                rightnum += 1
    resultnum=0
    for i in range(N):
        for j in range(len(y[i])):
            resultnum += 1
    rate1=rightnum/predictnum
    print(rightnum,predictnum,resultnum)
    print('正确率（准）：',rate1)
    rate2=rightnum/resultnum
    print('正确率（全）：',rate2)
    print(2*rate1*rate2 / (rate1 + rate2))
    """
    return predict_z
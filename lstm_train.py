from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence
import json
import numpy as np

def plot_acc_and_loss(history):
    from matplotlib import pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validtion acc')
    plt.legend()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

max_features = 10000 # 我们只考虑最常用的10k词汇
maxlen = 20 

path = 'assignment_training_data_word_segment.json'
sentence_list = json.load(open(path, 'r'))

train_rate = 0.9

train_max_num = int(len(sentence_list) * 100 * train_rate)
test_max_num = int(len(sentence_list) * 100 * (1-train_rate))

x_train = np.zeros(train_max_num, dtype=list)
y_train = np.zeros(train_max_num, dtype=np.int8)

x_test = np.zeros(test_max_num, dtype=list)
y_test = np.zeros(test_max_num, dtype=np.int8)

train_num=0
for i in range(int(len(sentence_list) * train_rate)):
    for time in sentence_list[i]['times']:
        for attribute in sentence_list[i]['attributes']:
            for value in sentence_list[i]['values']:
                y_train[train_num] = int([time, attribute, value] in sentence_list[i]['results'])
                x_train[train_num] = sentence_list[i]['indexes'] + [time, attribute, value]
                train_num+=1

test_num=0
for i in range(int(len(sentence_list) * train_rate), len(sentence_list)):
    for time in sentence_list[i]['times']:
        for attribute in sentence_list[i]['attributes']:
            for value in sentence_list[i]['values']:
                x_test[test_num] = sentence_list[i]['indexes'] + [time, attribute, value]
                y_test[test_num] = int([time, attribute, value] in sentence_list[i]['results'])
                test_num+=1

x_train = x_train[:train_num]
y_train = y_train[:train_num]
x_test = x_test[:test_num]
y_test = y_test[:test_num]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) #长了就截断，短了就补0

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2
)


old_model = load_model('my_model.h5')

#plot_acc_and_loss(history)
preds = (model.predict(x_test) > 0.5).astype("int32")
old_preds = (old_model.predict(x_test) > 0.5).astype("int32")

A = 0       # num of samples whose ground truth label is 1
A_hat = 0   # num of samples whose predicted label is 1
B = 0       # A and A_hat

for j in range(preds.shape[0]):
    if preds[j] == 1:
        A_hat += 1
    if y_test[j] == 1:
        A += 1
    if preds[j] == 1 and y_test[j] == 1:
        B += 1

p = B / A_hat
r = B / A

f1 = 2 * p * r / (p + r)

for j in range(old_preds.shape[0]):
    if old_preds[j] == 1:
        A_hat += 1
    if y_test[j] == 1:
        A += 1
    if old_preds[j] == 1 and y_test[j] == 1:
        B += 1

old_p = B / A_hat
old_r = B / A

old_f1 = 2 * old_p * old_r / (old_p + old_r)

if f1 > old_f1:
    model.save('my_model.h5')
    old_p = p
    old_r = r
    old_f1 = f1

print("precision: " + str(p))
print("recall: " + str(r))
print("f1 score: " + str(f1))

print("best precision: " + str(old_p))
print("best recall: " + str(old_r))
print("best f1 score: " + str(old_f1))
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence
import json
import numpy as np

def predict(vec, model) :
    return (model.predict(vec) > 0.5).astype("int32")
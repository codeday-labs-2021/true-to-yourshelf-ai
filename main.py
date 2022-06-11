import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#
# [[Data Needs To Be Put Here To Use Later]]
#

emotions = [['Happy'], ['Sad'], ['Mad'], ['Angry'], ['Scared'], ['Surprised'], ['Adventerous']]

emotion_encoder = MultiLabelBinarizer()
emotions_encoded = emotion_encoder.fit_transform(emotions)
num_emotions = len(emotions_encoded)
print(emotion_encoder.classes_)
print(emotions_encoded[0])

train_size = int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))

train_tags = tags_encoded[]:train_size]
test_tags = tags_encoded[train_size:]

from tensorflow.keras.preprocessing import text

class TextPreprocessor(object):
  def __init__(self, vocab_size):
    self._vocab_size = vocab_size
    self._tokenizer = None

  def create_tokenizer(self, text_list):
    tokenizer = text.Tokenizer(num_words=self._vocab_size)
    tokenizer.fit_on_texts(text_list)
    self.tokenizer = tokenizer

  def transform_text(self, text_list):
    text_matrix = self._tokenizer.texts_to_matrix(text_list)
    return text_matrix

from preprocessing import TextPreprocessor

train_qs = data['text'].values[:train_size]
test_qs = data['text'].values[train_size:]

VOCAB_SIZE = 400
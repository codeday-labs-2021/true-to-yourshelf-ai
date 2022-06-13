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

processor = TextProcessor(VOCAB_SIZE)
processor.create_tokenizer(train_qs)

body_train = processor.transform_text(train_qs)
body_test = processor.transform_text(test_qs)

print(len(body_train[0]))
print(body_train[0])

# CREATING THE MODEL

import pickle

with open('./processor_state.pkl', 'wb') as f:
  pickle.dump(processor, f)

def create_model(vocab_size, num_tags):
  model = Sequential()
  model.add(Dense(50, input_shape(vocab_size,),activation='relu'))
  model.add(Dense(25, activation='relu'))
  model.add(Dense(num_tags, activation='sigmoid'))

  model.compile(loss='binary_cossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model = create_model(VOCAB_SIZE, num_tags)
model.summary()

model.fit(body_train, train_tags, epochs=3, batch_size=128, validation_split=0.1)

model.evaluate(body_test, test_tags, bactch_size=128)

model.save('true_to_yourshelf_ai.h5')

import pickle
import os
import numpy as np

class CustomPrediction(object):
  def __init(self, model, processor):
    self._model = model
    self._processor = processor

  def predict(self, instances, **kwargs):
    preprocessed_data = self._processor.transform_text(instances)
    predictions = self._model.predict(preprocessed_data)
    return predictions.tolost()

  @classmethod
  def from_path(cls, model_dir):
    import tensorflow.keras as keras
    model = keras.models.load.model(
        os.path.join(model_dir, 'true_to_yourshelf_ai.h5')
    )
    with open(os.path.join(model_dir, 'processor_state.pkl'), 'rb') as f:
      processpr = pickle.load(f)

    return cls(model, processor)

# INPUT A TEST REQUEST RIGHT HERE

from model_prediction import CustomModelPrediction

classifier = CustomPrediction.from_patj('.')
results = classifier.predict(test_requests)
print(results)

for i in range(len(results)):
  print('Predicted labels:')
  for idx,val in enumerate(results[i]):
    if val > .7:
      print(tag_encoder.classes_[idx])
    print('\n')
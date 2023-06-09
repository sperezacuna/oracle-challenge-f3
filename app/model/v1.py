from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy
from keras.losses import BinaryCrossentropy
from keras import Input

from app.config import PARAMETERS

class PairSampleLabellerV1Model():
  def __init__(self):
    self.model = Sequential()
    self.model.add(Input(shape=(PARAMETERS['input-scope'],6,)))
    self.model.add(Dense(1024))
    self.model.add(BatchNormalization()) 
    self.model.add(LeakyReLU())
    self.model.add(Dropout(0.25))
    self.model.add(Dense(512))
    self.model.add(BatchNormalization()) 
    self.model.add(LeakyReLU())
    self.model.add(Dropout(0.25))
    self.model.add(Dense(128))
    self.model.add(BatchNormalization()) 
    self.model.add(LeakyReLU())
    self.model.add(Dropout(0.25))
    self.model.add(Dense(32))
    self.model.add(BatchNormalization()) 
    self.model.add(LeakyReLU())
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(
      optimizer=Adam(learning_rate=0.02),
      loss=BinaryCrossentropy(),
      metrics=[BinaryAccuracy()]
    )

  def predict(self, X):
    return self.model.predict(X, verbose=0)

  def fit(self, samples, labels, numEpochs, verbose):
    return self.model.fit(samples, labels,
              epochs = numEpochs,
              batch_size = PARAMETERS['batch-size'],
              verbose=verbose,
              shuffle=False)
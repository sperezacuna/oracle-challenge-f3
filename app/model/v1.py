from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape, Lambda
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D, LeakyReLU, BatchNormalization, GaussianNoise
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras import regularizers

from app.config import DATASET_PROPERTIES

def scheduler(epoch, lr):
  if epoch < 30:
    return lr
  elif epoch == 30:
    return 0.005
  else:
    if lr < 0.2:
      return lr * 1.0001
    else:
      return lr

class PairSampleLabellerModel():
  def __init__(self):
    self.model = Sequential()
    self.model.add(Dense(500, input_dim=6, activity_regularizer=regularizers.l2(0.01)))
    self.model.add(GaussianNoise(0.01))
    self.model.add(BatchNormalization()) 
    self.model.add(LeakyReLU()) 
    self.model.add(Dense(16, activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Flatten())
    self.model.add(BatchNormalization()) 
    self.model.add(LeakyReLU())
    self.model.add(Dense(1)) 
    self.model.add(Activation('sigmoid'))
    optimizer = Adam(learning_rate=0.02)
    self.amplify_lr = LearningRateScheduler(scheduler)
    self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

  def predict(self, X):
    return self.model.predict(X, verbose=1)

  def fit(self, samples, labels, numEpochs):
    history = self.model.fit(samples, labels, 
              epochs = numEpochs, 
              batch_size = DATASET_PROPERTIES['batch-size'],
              verbose=1, 
              callbacks=[self.amplify_lr],
              shuffle=False)
    return history
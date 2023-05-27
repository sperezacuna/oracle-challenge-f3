import os
import json
import pandas as pd
import numpy as np
from app.common.transform import cleanDataframe, calculateLabels

class IncrementalDataset():
  def __init__(self, training_csv, testing_csv):
    training_dataframe = pd.read_csv(training_csv, index_col=0, parse_dates=['Time'])
    training_dataframe = cleanDataframe(training_dataframe)
    testing_dataframe = pd.read_csv(testing_csv, index_col=0, parse_dates=['Time'])
    self.dataset = pd.concat([training_dataframe, testing_dataframe], ignore_index=True)
    self.dataset = calculateLabels(self.dataset)
    self.dataset['label'] = self.dataset['label'].astype(int)
    self.dataset['Time'] = self.dataset['Time'].apply(lambda x: x.date().toordinal())
    self.dataset.to_csv(os.path.join(os.path.dirname(__file__), f'../../data/processed/full_set.csv'))
    convertToJson(self.dataset.iloc[3509:])
    print(self.dataset.dtypes)
  def __len__(self):
    return len(self.dataset)
  def show(self):
    print(self.dataset)
  def getBatch(self, last_idx):
    X, Y = [], []
    for i in range(0, last_idx-2):
      x_i = np.column_stack((self.dataset.iloc[i]['Time'], self.dataset.iloc[i]['Open'], self.dataset.iloc[i]['High'], self.dataset.iloc[i]['Low'], self.dataset.iloc[i]['Close'], self.dataset.iloc[i]['Volume'])).flatten()
      y_i = self.dataset.iloc[i]['label']
      X.append(x_i)
      Y.append(y_i)
    for i in range(last_idx-2, last_idx+1):
      x_i = np.column_stack((self.dataset.iloc[i]['Time'], self.dataset.iloc[i]['Open'], self.dataset.iloc[i]['High'], self.dataset.iloc[i]['Low'], self.dataset.iloc[i]['Close'], self.dataset.iloc[i]['Volume'])).flatten()
      y_i = 0.5
      X.append(x_i)
      Y.append(y_i)
    X, Y = np.array(X), np.array(Y)
    return X, Y
  def getSample(self, idx):
    X = []
    X.append(np.column_stack((self.dataset.iloc[idx]['Time'], self.dataset.iloc[idx]['Open'], self.dataset.iloc[idx]['High'], self.dataset.iloc[idx]['Low'], self.dataset.iloc[idx]['Close'], self.dataset.iloc[idx]['Volume'])).flatten())
    return np.array(X)

def convertToJson(df):
  list_dict = []
  for index, row in list(df.iterrows()):
    list_dict.append(dict(row))
  results_dir = os.path.join(os.path.dirname(__file__), f'../../results')
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  with open(f'{results_dir}/perfect.json', 'w') as f:
    f.write(json.dumps({
      "target": { i: int(row['label']) for i, row in enumerate(list_dict) }
    }))

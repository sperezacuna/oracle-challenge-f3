import os
import json
import pandas as pd
from app.common.transform import cleanDataframe, calculateLabels
from app.config import DATALOAD_CONFIG

class IncrementalDataset():
  def __init__(self, training_csv, testing_csv):
    training_dataframe = pd.read_csv(training_csv, index_col=0, parse_dates=[1])
    training_dataframe = cleanDataframe(training_dataframe)
    training_dataframe.to_csv('out.csv') # TEMPORALLY
    testing_dataframe = pd.read_csv(testing_csv, index_col=0, parse_dates=[1])
    self.dataset = pd.concat([training_dataframe, testing_dataframe], ignore_index=True)
    self.dataset['label'] = self.dataset['label'].astype('boolean')
    self.dataset = calculateLabels(self.dataset)
    self.dataset.to_csv(os.path.join(os.path.dirname(__file__), f'../../data/processed/full_set.csv'))
    convertToJson(self.dataset.iloc[3509:])
  def show(self):
    print(self.dataset)
  def __getitem__(self, idx):
    return self.dataset.iloc[idx].to_list()
  def getSlice(max_idx):
    return self.dataset.iloc[:max_idx]

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

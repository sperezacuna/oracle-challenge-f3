import os
import json
import pandas as pd
import numpy as np

from app.common.transform import cleanDataframe, createFullDataset
from app.config import PARAMETERS

class IncrementalDataset():
  def __init__(self, training_csv, testing_csv):
    training_dataframe = pd.read_csv(training_csv, index_col=0, parse_dates=['Time'])
    training_dataframe = cleanDataframe(training_dataframe)
    testing_dataframe = pd.read_csv(testing_csv, index_col=0, parse_dates=['Time'])
    self.dataset = createFullDataset(training_dataframe, testing_dataframe, generateLabels=True)
    saveDatasetCsv(self.dataset)
    saveGroundTruthJson(self.dataset)
  def __len__(self):
    return len(self.dataset)
  def getPackagedSamples(self, first_idx, last_idx, qtty):
    sampleBatch = np.empty(shape=(last_idx-first_idx+1, qtty, 6))
    labelBatch = np.empty(shape=(last_idx-first_idx+1, 1))
    for i in range(first_idx, last_idx+1):
      labelBatch[i-first_idx, 0] = self.dataset.iloc[i]['label']
      for j in reversed(range(0, qtty)):
        sampleBatch[i-first_idx, j] = np.array([
          self.dataset.iloc[i-j]['Time'],
          self.dataset.iloc[i-j]['Open'],
          self.dataset.iloc[i-j]['High'],
          self.dataset.iloc[i-j]['Low'],
          self.dataset.iloc[i-j]['Close'],
          self.dataset.iloc[i-j]['Volume']
        ])
    return sampleBatch, labelBatch

def saveDatasetCsv(dataset):
  dataset.to_csv(os.path.join(os.path.dirname(__file__), f'../../data/processed/full_set.csv'))

def saveGroundTruthJson(dataset):
  tesinng_ground_truth_df = dataset[PARAMETERS['testing_offset']:]
  list_dict = []
  for index, row in list(tesinng_ground_truth_df.iterrows()):
    list_dict.append(dict(row))
  results_dir = os.path.join(os.path.dirname(__file__), f'../../results')
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  with open(f'{results_dir}/ground-truth.json', 'w') as f:
    f.write(json.dumps({
      "target": { i: int(row['label']) for i, row in enumerate(list_dict) }
    }))

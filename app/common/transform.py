import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def cleanDataframe(dataframe):
  dataframe['label'] = dataframe['label'].astype('boolean')
  # log(dataframe)
  dataframe = deleteOutliers(dataframe)
  # log(dataframe)
  try:
    assert dataframe.apply(isValid, axis=1).all(), "[!] Data has errors"
  except AssertionError as error:
    print(error)
    dataframe = dataframe.apply(fixSample, axis=1)
    print("Finished naive correction of errors")
  dataframe = fillGaps(dataframe)
  try:
    assert dataframe.apply(isValid, axis=1).all(), "[!] Data has errors"
  except AssertionError as error:
    print(error)
    dataframe = dataframe.apply(fixSample, axis=1)
    print("Finished naive correction of errors")
  assert dataframe.apply(isValid, axis=1).all(), "[!] Data has errors"
  # log(dataframe)
  return dataframe

def calculateLabels_prev(dataframe):
  for i in range(len(dataframe)-3):
    label = dataframe.loc[i, 'Close'] < dataframe.loc[i+3, 'Close']
    if pd.isna(dataframe.loc[i, 'label']):
      dataframe.loc[i, 'label'] = label
    elif label != dataframe.loc[i, 'label']: 
      dataframe.loc[i+3, 'Close'] = dataframe.loc[i, 'Close']+0.00001
  for i in range(len(dataframe)-3, len(dataframe)):
    dataframe.loc[i, 'label'] = False
  return dataframe

def calculateLabels(dataframe):
  counter = 0
  for i in range(len(dataframe)-3):
    if dataframe.loc[i, 'Close'] < dataframe.loc[i+3, 'Close']:
      if (dataframe.loc[i, 'Volume'] < dataframe.loc[i+3, 'Volume']):
        label = True
      elif (dataframe.loc[i, 'High']-dataframe.loc[i, 'Low'] < dataframe.loc[i+3, 'High']-dataframe.loc[i+3, 'Low']):
        label = True
      else:
        label = False
    else:
      label = False
    dataframe.loc[i, 'label'] = label
  for i in range(len(dataframe)-3, len(dataframe)):
    dataframe.loc[i, 'label'] = False
  return dataframe

def createFullDataset(training_dataframe, testing_dataframe, generateLabels):
  dataset = pd.concat([training_dataframe, testing_dataframe], ignore_index=True)
  dataset['Time'] = dataset['Time'].apply(lambda x: x.date().toordinal())
  if generateLabels:
    dataset = calculateLabels(dataset)
    print("Finished setting correct labels")
  print("Finished merging datasets")
  return dataset

def deleteOutliers(dataframe):
  for indicator in ["Open", "High", "Low", "Close"]:
    q1 = dataframe[indicator].quantile(0.25)
    q3 = dataframe[indicator].quantile(0.75)
    min_indicator = q1 - 1.5 * (q3 - q1)
    max_indicator = q3 + 1.5 * (q3 - q1)
    dataframe[indicator] = dataframe[indicator].mask((dataframe[indicator] < min_indicator) | (dataframe[indicator] > max_indicator))
  print("Finished deleting outliers")
  return dataframe

def isValid(sample):
  if not pd.isna(sample['Low']):
    if not pd.isna(sample['High']) and sample['Low'] > sample['High']:
      return False
    if not pd.isna(sample['Open']) and sample['Open'] < sample['Low']:
      return False
    if not pd.isna(sample['Close']) and sample['Close'] < sample['Low']:
      return False
  if not pd.isna(sample['High']):
    if not pd.isna(sample['Open']) and sample['Open'] > sample['High']:
      return False
    if not pd.isna(sample['Close']) and sample['Close'] > sample['High']:
      return False
  if not pd.isna(sample['Volume']) and sample['Volume'] < 0:
    return False
  return True

def fillGaps(dataframe):
  dataframe = transitiveFillOpenClose(dataframe)
  dataframe = bayesianFillHighLowVolume(dataframe)
  print("Finished filling gaps")
  return dataframe

def transitiveFillOpenClose(dataframe):
  for i in range(0, len(dataframe)):
    if not i == 0 and pd.isna(dataframe.loc[i, 'Open']):
      dataframe.loc[i, 'Open'] = dataframe.loc[i-1, 'Close']
      assert dataframe.loc[i, 'Open'] == dataframe.loc[i-1, 'Close']
    if not i == len(dataframe) and pd.isna(dataframe.loc[i, 'Close']):
      dataframe.loc[i, 'Close'] = dataframe.loc[i+1, 'Open']
      assert dataframe.loc[i, 'Close'] == dataframe.loc[i+1, 'Open']
  return dataframe

def bayesianFillHighLowVolume(dataframe):
  imputer = IterativeImputer()
  imputer.fit(dataframe[['Open', 'High', 'Low', 'Close', 'Volume']])
  dataframe[['Open', 'High', 'Low', 'Close', 'Volume']] = imputer.transform(dataframe[['Open', 'High', 'Low', 'Close', 'Volume']])
  return dataframe

def fixSample(sample):
  if sample['Volume'] < 0:
    sample['Volume'] = 0
  if sample['Close'] < sample['Open']:
    if sample['High'] < sample['Open']: sample['High'] = sample['Open']
    if sample['Low'] > sample['Close']: sample['Low'] = sample['Close']
  else:
    if sample['High'] < sample['Close']: sample['High'] = sample['Close']
    if sample['Low'] > sample['Open']: sample['Low'] = sample['Open']
  return sample

def log(dataframe):
  print(dataframe.describe())
  print()
  print(dataframe.isna().sum())
  print()
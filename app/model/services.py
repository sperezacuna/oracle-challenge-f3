import numpy as np
from app.config import DATASET_PROPERTIES

def performIncrementalInferenceLearning(incrementalDataset, model):
  samples, labels = incrementalDataset.getBatch(DATASET_PROPERTIES['testing_offset'])
  history = model.fit(samples, labels, numEpochs=DATASET_PROPERTIES['initial-epochs'])
  all_preds = []
  for i in range (DATASET_PROPERTIES['testing_offset'], len(incrementalDataset)):
    topredict = incrementalDataset.getSample(i)
    prediction = model.predict(topredict)
    print(f'[!] Completed prediction for item {i}/{len(incrementalDataset)}: {int(prediction.item(0))}')
    all_preds.append(int(prediction.item(0)))
    samples, labels = incrementalDataset.getBatch(i)
    history = model.fit(samples, labels, numEpochs=DATASET_PROPERTIES['epochs-per-sample'])
  return all_preds

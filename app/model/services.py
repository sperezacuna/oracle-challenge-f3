import numpy as np
from app.config import DATASET_PROPERTIES
from app.model.v1 import PairSampleLabellerModel

def performIncrementalInferenceLearning(incrementalDataset):
  all_preds = []
  for i in range (DATASET_PROPERTIES['testing_offset'], len(incrementalDataset)):
    model = PairSampleLabellerModel()
    samples, labels = incrementalDataset.getBatch(i)
    history = model.fit(samples, labels, numEpochs=DATASET_PROPERTIES['initial-epochs'])
    topredict = incrementalDataset.getSample(i)
    prediction = model.predict(topredict)
    print(f'[!] Completed prediction for item {i}/{len(incrementalDataset)}: {int(prediction.item(0))}')
    all_preds.append(int(prediction.item(0)))
  return all_preds

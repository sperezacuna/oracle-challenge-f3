import numpy as np
from app.config import PARAMETERS

def performIncrementalInferenceLearning(dataset, model):
  all_preds = []
  # pictogram = ""
  wrong_count = false_positives = false_negatives = true_positives = 0
  first_training_idx = PARAMETERS['input-scope']
  last_training_idx  = PARAMETERS['testing_offset'] - PARAMETERS['label-lookahead']
  trainingSamples, trainingLabels = dataset.getPackagedSamples(first_training_idx, last_training_idx, qtty=PARAMETERS['input-scope'])
  print("> Started base model training")
  model.fit(trainingSamples, trainingLabels, numEpochs=PARAMETERS['initial-epochs'], verbose=2)
  print("> Finished base model training, performing inference and relearning...")
  for sample_idx in range (PARAMETERS['testing_offset'], len(dataset)):
    last_training_idx  = sample_idx-PARAMETERS['label-lookahead']
    if (PARAMETERS['enable-sliding-window']):
      first_training_idx = last_training_idx - PARAMETERS['window-size']
    trainingSamples, trainingLabels = dataset.getPackagedSamples(first_training_idx, last_training_idx, qtty=PARAMETERS['input-scope'])
    testingSample,   testingLabel   = dataset.getPackagedSamples(sample_idx, sample_idx, qtty=PARAMETERS['input-scope'])
    model.fit(trainingSamples, trainingLabels, numEpochs=PARAMETERS['epochs-per-sample'], verbose=0)
    prediction = model.predict(testingSample)
    out = int(bool(round(prediction.item(0)) > 0))
    all_preds.append((out))
    if out!=testingLabel[0,0]:
      wrong_count += 1
      if (testingLabel[0,0] == 1): false_negatives +=1
      else: false_positives +=1
    elif testingLabel[0,0]:
      true_positives += 1
    print(f'Completed prediction for sample {sample_idx-PARAMETERS["testing_offset"]+1}/{len(dataset)-PARAMETERS["testing_offset"]}: {prediction.item(0):.4f} {"✅" if out==testingLabel else "❌"}')
    print(f'> {(100-wrong_count/(sample_idx+1-PARAMETERS["testing_offset"])*100):.4f}% of current predictions are correct')
    if wrong_count:
      print(f'> {(false_negatives/(false_positives+false_negatives)*100):.4f}% of errors are false negatives')
      print(f'> {(false_positives/(false_positives+false_negatives)*100):.4f}% of errors are false positives')
    # pictogram += "✅" if out==testingLabel else "❌"
    print('' + pictogram)
  return all_preds
import os
import sys
import json
import uuid
import getopt
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from app.common.dataload import IncrementalDataset
from app.model.services import performIncrementalInferenceLearning

def help():
  print("Usage: python main.py generate_model.py [-h] [-m MODELTYPE]\n")
  print("\tCreates and performs continuous inference over the testing dataset\n")
  print("Options:")
  print("\t-m, --model MODELTYPE\tEstablish the model iteration to use [default is v1]")
  print("\t-h, --help\tShow this help message and exit")

def main(argv):
  try:
    arguments, values = getopt.getopt(argv, "h", ["help"])
    modeltype == "v1"
    for currentArgunemt, currentValue in arguments:
      if currentArgunemt in ("-m", "--modeltype"):
        modelType = currentValue
      if currentArgunemt in ("-h", "--help"):
        help()
        sys.exit(0)
    if modelType == "v1":
      from app.model.v1 import PairSampleLabellerV1Model
      model = PairSampleLabellerV1Model()
    else:
      print("[!] Invalid model type")
      sys.exit(1)
  except getopt.error as err:
    print("[!] " + str(err))
    sys.exit(1)
  dataset = IncrementalDataset(os.path.join(os.path.dirname(__file__), "data/raw/training_set.csv"),
                               os.path.join(os.path.dirname(__file__), "data/raw/testing_set.csv"))
  results = performIncrementalInferenceLearning(dataset, model)
  results_dir = os.path.join(os.path.dirname(__file__), f'results/{modelType}')
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  with open(f'{results_dir}/{uuid.uuid4().hex}.json', 'w') as f:
    f.write(json.dumps({
      "target": { i: label for i, label in enumerate(results) }
    }))

if __name__ == "__main__":
  main(sys.argv[1:])
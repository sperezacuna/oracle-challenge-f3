import os
import sys
import json
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import getopt

from app.common.dataload import IncrementalDataset
from app.model.v1 import PairSampleLabellerModel
from app.model.services import performIncrementalInferenceLearning

def help():
  print("Usage: main.py [TODO]\n")
  print("\tTODO description\n")
  print("Options:")
  print("\t-h, --help\tShow this help message and exit")

def main(argv):
  try:
    arguments, values = getopt.getopt(argv, "h", ["help"])
    for currentArgunemt, currentValue in arguments:
      if currentArgunemt in ("-h", "--help"):
        help()
        sys.exit(0)
    # PERFORM PROGRAM LOGIC
  except getopt.error as err:
    print("[!] " + str(err))
    sys.exit(1)
  dataset = IncrementalDataset(os.path.join(os.path.dirname(__file__), "data/raw/training_set.csv"),
                               os.path.join(os.path.dirname(__file__), "data/raw/testing_set.csv"))
  results = performIncrementalInferenceLearning(dataset)
  results_dir = os.path.join(os.path.dirname(__file__), f'results/default-model')
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  with open(f'{results_dir}/current.json', 'w') as f:
    f.write(json.dumps({
      "target": { i: label for i, label in enumerate(results) }
    }))

if __name__ == "__main__":
  main(sys.argv[1:])
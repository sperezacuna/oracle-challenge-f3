import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import getopt

from app.common.dataload import IncrementalDataset

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
  dataset.show()
  print("Done")

if __name__ == "__main__":
  main(sys.argv[1:])
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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

if __name__ == "__main__":
  main(sys.argv[1:])
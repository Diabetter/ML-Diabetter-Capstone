import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--value-1')
parser.add_argument('--value-2')
args = parser.parse_args()

print("Hello")
print(args.value_2)
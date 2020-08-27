import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from apps.algorithms import Algorithm
from apps.algorithm_loader import Algorithm_Loader
import apps.data_generator as dg
import pickle

with open("300_5_dataset_results.pkl", "rb") as input_file:
    e = pickle.load(input_file)

print(e)
print(e.GC)
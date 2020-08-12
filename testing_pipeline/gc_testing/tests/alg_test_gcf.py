import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from apps.algorithms import Algorithm
from apps.algorithm_loader import Algorithm_Loader
import apps.data_generator as dg

# generates a dataset of f=5, t=300 values
dataset1 = dg.dataset(features=5)
dataset1.gen_var_data()

# we can run a specific algorithm with specific parameters
args_gcf = {"train_epochs": 50, "learning_rate": 0.01, "batch_size": 32, "p":15, "q":15, 'verbose':0}
args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'gcf':args_gcf}}
alg_load1 = Algorithm_Loader(args)
# print out granger causality matrix
print(alg_load1.dataset.GC)
print(alg_load1.dataset.dependencies['dep1'])
print(alg_load1)

dataset2 = dg.dataset(features=5)
dataset2.gen_lorenz96_data()
# or we run a specific algorithm with default parameters
args = {'dataset':dataset2, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'gcf':{'verbose':0}}}
#args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2'}

alg_load2 = Algorithm_Loader(args)
print(alg_load2.dataset.GC)
print(alg_load2.dataset.dependencies['dep1'])
print(alg_load2)

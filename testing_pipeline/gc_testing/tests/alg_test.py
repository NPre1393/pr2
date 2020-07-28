import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
#print(sys.path)
from apps.algorithms import Algorithm
from apps.algorithm_loader import Algorithm_Loader
import apps.data_generator as dg
from test_algs.gc_ame.gc_ame_alg import GC_AME

dataset1 = dg.dataset()
dataset1.gen_var_data()

args = {'data':dataset1.data, 'result_path':'result1', 'model_path':'result2'}
alg_load = Algorithm_Loader(args)
print(alg_load)

args_alg = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "granger_loss_weight": 0.05, "l2_weight": 12, "num_units": 23, "num_layers": 34}
gc = GC_AME(args_alg)
print(gc)
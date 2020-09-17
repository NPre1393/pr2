import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
#print(sys.path)
from apps.algorithm_loader import Algorithm_Loader
import apps.data_generator as dg
#from test_algs.gc_ame.gc_ame_alg import GC_AME
#from test_algs.dca_bi_final.bi_cgl import BI_CGL

config = 'config.ini'
dataset1 = dg.dataset(features=5)
dataset1.gen_var_data()
"""
args_gcf = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "p":200, "q":200}
args = {'dataset':dataset1.data, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'gcf':args_gcf}}
alg_load = Algorithm_Loader(args)
print(alg_load)
"""
#args_gcf = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "p":200, "q":200}
#args = {'data':dataset1.data, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'gcf':0}}
args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2', 'algorithms':config}
#args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2'}

alg_load = Algorithm_Loader(args)
print(alg_load.dataset.GC)
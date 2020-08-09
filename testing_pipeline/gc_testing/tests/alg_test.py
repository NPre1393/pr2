import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
#print(sys.path)
from apps.algorithms import Algorithm
from apps.algorithm_loader import Algorithm_Loader
import apps.data_generator as dg
#from test_algs.gc_ame.gc_ame_alg import GC_AME
#from test_algs.dca_bi_final.bi_cgl import BI_CGL

dataset1 = dg.dataset(features=5)
dataset1.gen_var_data()
"""
args_gcf = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "p":200, "q":200}
args = {'data':dataset1.data, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'gcf':args_gcf}}
alg_load = Algorithm_Loader(args)
print(alg_load)
"""
#args_gcf = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "p":200, "q":200}
#args = {'data':dataset1.data, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'gcf':0}}
args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'neunetnue':0}}
#args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2'}

alg_load = Algorithm_Loader(args)
print(alg_load.dataset.GC)
#print(alg_load.dataset.dependencies)
#print(alg_load.dataset.GC)



#print(alg_load)
#args_ame = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "granger_loss_weight": 0.05, "l2_weight": 12, "num_units": 23, "num_layers": 34}
# /content/ame/ame_starter/apps/main.py --dataset="boston_housing" --batch_size=32 --num_epochs=300 --learning_rate=0.001 --output_directory='/content/drive/My Drive/pr2/ame_output' --do_train --do_evaluate --num_units=16 --num_layers=1 --early_stopping_patience=32
#gc1 = GC_AME(args_ame)
#gc1.run()
#print(gc1)

#args_bicgl = {"train_epochs": 100, "learning_rate": 0.01, "batch_size": 32, "cuda":False}
#gc2 = BI_CGL(args_bicgl)
#print(gc2)
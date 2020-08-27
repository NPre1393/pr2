import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from apps.algorithm_loader import Algorithm_Loader
import apps.data_generator as dg

# generates a dataset of f=5, t=300 values
dataset1 = dg.dataset(features=5)
dataset1.gen_var_data()

# we can run a specific algorithm with specific parameters
args_bicgl = {
    "train_epochs":200, "learning_rate":0.01, "batch_size":50, 'y_dim':5, 'model':'DCM_DeepCausal', 'window':5,
    'pre_win':3, 'horizon':1, 'lowrank':50, 'p_list_number':40, 'p_list_layer':5, 'compress_p_list':'80', 'L1Loss': False, 'clip':1., 'dropout':0.1, 'seed':12345, 
    'gpu':0, 'cuda':False, 'optim':'adam', 'lr':0.01, 'lr_decay':0.99, 'start_decay_at':100, 'weight_decay':0, 'normalize':1, 'train':0.9,
    'valid':0.09, 'lambda1':0.1, 'verbose':1
}
args = {'dataset':dataset1, 'result_path':'./', 'model_path':'result2', 'algorithms':{'bicgl':args_bicgl}}
alg_load1 = Algorithm_Loader(args)
# print out granger causality matrix
print(alg_load1.dataset.GC)
print(alg_load1.dataset.dependencies['dep1'])
print(alg_load1)

dataset2 = dg.dataset(features=5)
dataset2.gen_lorenz96_data()
# or we run a specific algorithm with default parameters
args = {'dataset':dataset2, 'result_path':'result1', 'model_path':'result2', 'algorithms':{'bicgl':0}}
#args = {'dataset':dataset1, 'result_path':'result1', 'model_path':'result2'}

alg_load2 = Algorithm_Loader(args)
print(alg_load2.dataset.GC)
print(alg_load2.dataset.dependencies['dep1'])
print(alg_load2)
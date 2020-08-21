import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import pandas as pd
import pickle

from apps.data_generator import dataset
#from test_algs.gcf.Bivariate_GCF import run_main as run_gcf
#from test_algs.dca_bi_final.bi_cgl import run_main as run_bicgl


"""
    test algorithms implemented

    gcf:        granger causal features
    bicgl:      bi-directional causal graph learning
    gc_ame:     granger-causal attentive mixture of experts
    neural_gc:  neural granger causality
    neunetnue:  neural networks with non-uniform embedding

"""

class Algorithm_Loader:

    def __init__(self, args):

        """
        Constructor for algorithm loader object
        Args (required):
        :param dataset(obj)             data generator dataset object 
        
        Args (optional):
        :param result_path (string):    folder where GC results should be saved
        :param model_path (string):     folder where algorithm models should be saved
        :param algorithms (dict):       dict in format {'alg1':{{'param1':value,...}}}
        """

        self.args = args
        self.dataset = self.args["dataset"]
        self.result_path = self.args["result_path"]
        self.model_path = self.args["model_path"]
        if not self.args.get("algorithms"):
            self.algorithms = {'gcf':0, 'bicgl':0, 'gc_ame':0, 'neural_gc':0, 'neunetnue':0}
            self.args["algorithms"] = {'gcf':0, 'bicgl':0, 'gc_ame':0, 'neural_gc':0, 'neunetnue':0}
        else:
            self.algorithms = self.args["algorithms"]
        self.run_algorithms()
        self.save_result()

    def __repr__(self):
        repr = 'Algorithm Loader Information\n'\
            'Dataset dims = {}\n'\
            'result path = {}\n'\
            'model path = {}\n'\
            'algorithms to run: {}'\
            .format(self.dataset.data.shape, self.result_path, self.model_path, list(self.algorithms.keys()))
        return repr

    def run_algorithms(self):
        alg_keys = [*self.algorithms]
        for alg in alg_keys:
            fn_tocall = 'self.{}({})'.format(alg, self.algorithms[alg])
            eval(fn_tocall)

    def gcf(self, arguments):
        from test_algs.gcf.Bivariate_GCF import run_main as run_gcf

        """
            default arguments if non given
            "train_epochs": 50, 
            "learning_rate": 0.01, 
            "batch_size": 32,
            "inputbatchsize" = n*0.9,
            "p" = n*0.04,
            "q" = n*0.04,
            "verbose" = 1 [0:2]
        """
        alg_arguments = {
            'alg_loader':self,"train_epochs": 50, "learning_rate": 0.01, "batch_size": 32,'inputbatchsize': 
            int(self.dataset.n*0.9),'p': int(self.dataset.n*0.04),'q': int(self.dataset.n*0.04), 'verbose':1
        }

        if arguments:
            #arguments = {'alg_loader':self,"train_epochs": 50, "learning_rate": 0.01, "batch_size": 32,'inputbatchsize': int(self.dataset.n*0.9),'p': int(self.dataset.n*0.04),'q': int(self.dataset.n*0.04)}
            for arg in arguments.keys():
                alg_arguments[arg] = arguments[arg]
        print(alg_arguments)
        G, graph_dict, gc_dict, GC = run_gcf(alg_arguments)
        tmp = np.squeeze(np.asarray(GC))
        GC = pd.DataFrame(data=tmp, index=gc_dict.keys(), columns=gc_dict.keys())
        self.dataset.GC['gcf'] = GC

    def bicgl(self, arguments):
        from test_algs.dca_bi_final.bi_cgl import run_main as run_bicgl

        alg_arguments = {
            'alg_loader':self,"train_epochs":200, "learning_rate":0.01, "batch_size":50, 'y_dim':self.dataset.features, 'model':'DCM_DeepCausal', 'window':5,
            'pre_win':3, 'horizon':1, 'lowrank':50, 'p_list_number':40, 'p_list_layer':5, 'compress_p_list':'80', 'L1Loss': False, 'clip':1., 'dropout':0.1, 'seed':12345, 
            'gpu':0, 'save':self.model_path, 'cuda':False, 'optim':'adam', 'lr':0.01, 'lr_decay':0.99, 'start_decay_at':100, 'weight_decay':0, 'normalize':1, 'train':0.9,
            'valid':0.09, 'lambda1':0.1, 'verbose':1
        }

        if arguments:
            for arg in arguments.keys():
                alg_arguments[arg] = arguments[arg]

        GC = run_bicgl(alg_arguments)
        GC = pd.DataFrame(data=GC, index=self.dataset.data.columns.tolist(), columns=self.dataset.data.columns.tolist())
        self.dataset.GC['bigcl'] = GC

    def gc_ame(self, arguments):
        #from test_algs.gc_ame.gc_ame_alg import run_main as run_gc_ame
        pass

    def neural_gc(self, arguments):
        from test_algs.neural_gc.neural_gc import run_main as run_main_neural_gc

        alg_arguments = {
            'alg_loader':self,"train_epochs": 10000, "learning_rate": 0.01, "batch_size": 32,'hidden':10, 'lag':5, 'verbose':1, 'model':'mlp'
        }

        if arguments:
            for arg in arguments.keys():
                alg_arguments[arg] = arguments[arg]

        GC = run_main_neural_gc(alg_arguments)
        GC = pd.DataFrame(data=GC, index=self.dataset.data.columns.tolist(), columns=self.dataset.data.columns.tolist())
        self.dataset.GC['neural_gc'] = GC

    def neunetnue(self, arguments):
        from test_algs.neunetnue.neunetnue import run_main as run_main_neunetnue

        alg_arguments = {
            'alg_loader':self,"train_epochs": 10000, "learning_rate": 0.01, "batch_size": 32,'verbose':1, 'nr_of_procs':4
        }

        if arguments:
            for arg in arguments.keys():
                alg_arguments[arg] = arguments[arg]

        GC = run_main_neunetnue(alg_arguments)
        GC = pd.DataFrame(data=GC, index=self.dataset.data.columns.tolist(), columns=self.dataset.data.columns.tolist())
        self.dataset.GC['neunetnue'] = GC

    def save_result(self):
        with open(self.result_path+self.dataset.n+'_'+self.dataset.features+'_dataset_results', 'wb') as handle:
            pickle.dump(self.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_model(self, model, method):
        with open(self.result_path+method+'_nnmodel', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from apps.data_generator import dataset
from test_algs.gcf.Bivariate_GCF import run_main as run_gcf

"""
    gcf:        granger causal features
    bicgl:      bi-directional causal graph learning
    gc_ame:     granger-causal attentive mixture of experts
    neural_gc:  neural granger causality
    neunetnue:  neural networks with non-uniform embedding

"""

class Algorithm_Loader:

    def __init__(self, args):
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
        """
            default arguments if non given
            "train_epochs": 50, 
            "learning_rate": 0.01, 
            "batch_size": 32,
            "inputbatchsize" = n*0.9
            "p" = n*0.04
            "q" = n*0.04
        """
        if not arguments:
            arguments = {'alg_loader':self,"train_epochs": 50, "learning_rate": 0.01, "batch_size": 32,'inputbatchsize': int(self.dataset.n*0.9),'p': int(self.dataset.n*0.04),'q': int(self.dataset.n*0.04)}

        run_gcf(arguments)

    def bicgl(self, arguments):
        pass

    def gc_ame(self, arguments):
        pass

    def neural_gc(self, arguments):
        pass

    def neunetnue(self, arguments):
        pass

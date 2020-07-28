import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from apps.data_generator import dataset

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
        self.data = self.args["data"]
        self.result_path = self.args["result_path"]
        self.model_path = self.args["model_path"]

    def __repr__(self):
        repr = 'Algorithm Loader Information\n'\
            'Dataset dims = {}\n'\
            'result path = {}\n'\
            'model path = {}'\
            .format(self.data.shape, self.result_path, self.model_path)

        return repr

    def gcf(self):
        pass

    def bicgl(self):
        pass

    def gc_ame(self):
        pass

    def neural_gc(self):
        pass

    def neunetnue(self):
        pass

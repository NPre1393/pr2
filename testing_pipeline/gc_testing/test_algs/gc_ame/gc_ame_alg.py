import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from apps.algorithms import Algorithm

class GC_AME(Algorithm):

    def __init__(self, args):
        super(GC_AME, self).__init__(args)
        self.granger_loss_weight = self.args["granger_loss_weight"]
        self.l2_weight = self.args["l2_weight"]
        self.num_units = self.args["num_units"]
        self.num_layers = self.args["num_layers"]

    def __repr__(self):
        repr_s = super(GC_AME, self).__repr__()
        repr = '\nGranger Loss Weight = {}\n'\
            'L2 weight = {}\n'\
            'NN nr of HUs = {}\n'\
            'NN nr of layers = {}'\
            .format(self.granger_loss_weight, self.l2_weight, self.num_units, self.num_layers)

        return repr_s+repr

    def run(self):
        return None
import sys, os
root_dir = os.path.dirname(sys.path[0])
sys.path.append(root_dir)
sys.path.append(root_dir+'/test_algs/gcf/')
#print(os.path.dirname(sys.path[0]))
from argparse import ArgumentParser, Action, ArgumentTypeError
from apps.algorithms import Algorithm
from gcf.Bivariate_GCF import run_main
#from test_algs.gc_ame.ame_starter.apps.main import MainApplication

class GCF(Algorithm):

    def __init__(self, args):
        super(GCF, self).__init__(args)
        self.lag = self.args["lag"]
        self.p = self.args["p"]
        self.q = self.args["q"]

    def __repr__(self):
        repr_s = super(GCF, self).__repr__()
        repr = '\nGranger Loss Weight = {}\n'\
            'L2 weight = {}\n'\
            'NN nr of HUs = {}\n'\
            'NN nr of layers = {}'\
            .format(self.lag, self.p, self.q)
        return repr_s+repr

    def run(self):
        #!python /content/ame/ame_starter/apps/main.py --dataset="boston_housing" --batch_size=32 --num_epochs=300 --learning_rate=0.001 --output_directory='/content/drive/My Drive/pr2/ame_output' --do_train --do_evaluate --num_units=16 --num_layers=1 --early_stopping_patience=32
        run_main()


    def parse_params(self):
        pass
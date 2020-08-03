import sys, os
root_dir = os.path.dirname(sys.path[0])
sys.path.append(root_dir)
sys.path.append(root_dir+'/test_algs/gc_ame/')
#print(os.path.dirname(sys.path[0]))
from argparse import ArgumentParser, Action, ArgumentTypeError
from apps.algorithms import Algorithm
from ame_starter.apps.main import run_main
#from test_algs.gc_ame.ame_starter.apps.main import MainApplication

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
        #!python /content/ame/ame_starter/apps/main.py --dataset="boston_housing" --batch_size=32 --num_epochs=300 --learning_rate=0.001 --output_directory='/content/drive/My Drive/pr2/ame_output' --do_train --do_evaluate --num_units=16 --num_layers=1 --early_stopping_patience=32
        run_main()


    def parse_params(self):
        parser = ArgumentParser(description='AME starter project.')
        parser.add_argument("--dataset", default="boston_housing",
                            help="The data set to be loaded (mnist, boston_housing).")
        parser.add_argument("--seed", type=int, default=909,
                            help="Seed for the random number generator.")
        parser.add_argument("--output_directory", default="./models",
                            help="Base directory of all output files.")
        parser.add_argument("--model_name", default="forecast.h5.npz",
                            help="Base directory of all output files.")
        parser.add_argument("--load_existing", default="",
                            help="Existing model to load.")
        parser.add_argument("--n_jobs", type=int, default=4,
                            help="Number of processes to use where available for multitasking.")
        parser.add_argument("--learning_rate", default=0.0001, type=float,
                            help="Learning rate to use for training.")
        parser.add_argument("--l2_weight", default=0.0, type=float,
                            help="L2 weight decay used on neural network weights.")
        parser.add_argument("--num_epochs", type=int, default=150,
                            help="Number of epochs to train for.")
        parser.add_argument("--batch_size", type=int, default=300,
                            help="Batch size to use for training.")
        parser.add_argument("--early_stopping_patience", type=int, default=12,
                            help="Number of stale epochs to wait before terminating training")
        parser.add_argument("--num_units", type=int, default=8,
                            help="Number of neurons to use in DNN layers.")
        parser.add_argument("--num_layers", type=int, default=2,
                            help="Number of layers to use in DNNs.")
        parser.add_argument("--dropout", default=0.0, type=float,
                            help="Value of the dropout parameter used in training in the network.")
        parser.add_argument("--granger_loss_weight", type=float, default=0.03,
                            help="Weight of the granger causal loss [0.0, 1.0]. "
                                "Note: Main loss will be set to 1 - granger_loss_weight.")
        parser.add_argument("--fraction_of_data_set", type=float, default=1,
                            help="Fraction of time_series to use for folds.")
        parser.add_argument("--validation_set_fraction", type=float, default=0.27,
                            help="Fraction of time_series to hold out for the validation set.")
        parser.add_argument("--test_set_fraction", type=float, default=0.1,
                            help="Fraction of time_series to hold out for the test set.")
        parser.add_argument("--num_hyperopt_runs", type=int, default=35,
                            help="Number of hyperopt runs to perform.")
        parser.add_argument("--hyperopt_offset", type=int, default=0,
                            help="Offset at which to start the hyperopt runs.")

        parser.set_defaults(do_train=False)
        parser.add_argument("--do_train", dest='do_train', action='store_true',
                            help="Whether or not to train a model.")
        parser.set_defaults(do_hyperopt=False)
        parser.add_argument("--do_hyperopt", dest='do_hyperopt', action='store_true',
                            help="Whether or not to perform hyperparameter optimisation.")
        parser.set_defaults(do_evaluate=False)
        parser.add_argument("--do_evaluate", dest='do_evaluate', action='store_true',
                            help="Whether or not to evaluate a model.")
        parser.set_defaults(hyperopt_against_eval_set=False)
        parser.add_argument("--hyperopt_against_eval_set", dest='hyperopt_against_eval_set', action='store_true',
                            help="Whether or not to evaluate hyperopt runs against the evaluation set.")
        parser.set_defaults(copy_to_local=False)
        parser.add_argument("--copy_to_local", dest='copy_to_local', action='store_true',
                            help="Whether or not to copy the dataset to a local cache before training.")
        parser.set_defaults(do_hyperopt_on_lsf=False)
        parser.add_argument("--do_hyperopt_on_lsf", dest='do_hyperopt_on_lsf', action='store_true',
                            help="Whether or not to perform hyperparameter optimisation split into multiple jobs on LSF.")
        parser.set_defaults(do_merge_lsf=False)
        parser.add_argument("--do_merge_lsf", dest='do_merge_lsf', action='store_true',
                            help="Whether or not to merge LSF hyperopt runs.")
        parser.set_defaults(with_tensorboard=False)
        parser.add_argument("--with_tensorboard", dest='with_tensorboard', action='store_true',
                            help="Whether or not to serve tensorboard data.")

        parser.set_defaults(save_predictions=True)
        parser.add_argument("--do_not_save_predictions", dest='save_predictions', action='store_false',
                            help="Whether or not to save predictions.")
        parser.add_argument("--save_predictions", dest='save_predictions', action='store_true',
                            help="Whether or not to save predictions.")
        parser.set_defaults(save_attributions=True)
        parser.add_argument("--do_not_save_attributions", dest='save_attributions', action='store_false',
                            help="Whether or not to save attributions.")

        return vars(parser.parse_args())
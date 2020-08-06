import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from apps.algorithms import Algorithm

class BI_CGL(Algorithm):

    def __init__(self, args):
        super(BI_CGL, self).__init__(args)
        self.cuda = self.args["cuda"]

    def __repr__(self):
        repr_s = super(BI_CGL, self).__repr__()
        repr = '\nCuda enabled = {}'\
            .format(self.cuda)

        return repr_s+repr

    def parse_params(self, args):
        parser = argparse.ArgumentParser(description= 'SCGL')
        parser.add_argument('--data', type = str,  default = 'data/syntheticA/filter_norm_expression0.mat')
        parser.add_argument('--graph_path', type = str, default = 'data/syntheticA/groundtruth.mat')
        parser.add_argument('--GTu_path', type = str,  default = 'data/syntheticA/groundtruth.mat')
        parser.add_argument('--y_dim', type = int,  default = 65)

        parser.add_argument('--model', type = str, default = 'DCM_DeepCausal')

        parser.add_argument('--window', type = int, default = 5)
        parser.add_argument('--pre_win', type = int, default = 3)
        parser.add_argument('--horizon', type = int,  default = 1)


        parser.add_argument('--lowrank', type = int, default = 50)
        parser.add_argument('--p_list_number', type = int, default = 40)
        parser.add_argument('--p_list_layer', type = int, default = 5)
        parser.add_argument('--compress_p_list', type = str, default = '80')

        parser.add_argument('--L1Loss', type = bool, default = False)
        parser.add_argument('--clip', type = float,  default = 1.)

        parser.add_argument('--epochs', type = int, default = 200)
        parser.add_argument('--batch_size', type = int, default = 50)
        parser.add_argument('--dropout', type = float,  default = 0.1)
        parser.add_argument('--seed', type = int,  default = 12345)
        parser.add_argument('--gpu', type = int, default = 0)
        parser.add_argument('--save', type = str, default = 'save/model.pt')

        parser.add_argument('--cuda', type = str, default = False)
        parser.add_argument('--optim', type = str,  default = 'adam')
        parser.add_argument('--lr', type = float,  default = 0.01)
        parser.add_argument('--lr_decay', type = float,  default = 0.99)
        parser.add_argument('--start_decay_at', type = int,  default = 100)
        parser.add_argument('--weight_decay', type = float,  default = 0)

        parser.add_argument('--normalize', type = int,  default = 1)

        parser.add_argument('--train', type = float, default = 0.9)
        parser.add_argument('--valid', type = float,  default = 0.09)

        parser.add_argument('--lambda1', type = float,  default = 0.1)
        
        return vars(parser.parse_args())

    def run(self):
        pass
import sys, os
root_dir = os.path.dirname(sys.path[0])
sys.path.append(root_dir)
sys.path.append(root_dir+'/test_algs/neural_gc/')

import numpy as np
import torch
from models.cmlp import cMLP
from models.cmlp import train_model_adam as cmlp_adam
from models.cmlp import train_model_gista as cmlp_gista

from models.crnn import cRNN
from models.crnn import train_model_adam as crnn_adam
from models.crnn import train_model_gista as crnn_gista

from models.clstm import cLSTM
from models.clstm import train_model_adam as clstm_adam
from models.clstm import train_model_gista as clstm_gista

from synthetic import simulate_var
import matplotlib.pyplot as plt


def run_main(args):
    
    #device = torch.device('cuda')
    data = args['alg_loader'].dataset.data.to_numpy()
    #X = torch.tensor(data[np.newaxis], dtype=torch.float32, device=device)
    X = torch.tensor(data[np.newaxis], dtype=torch.float32)
    p = args['alg_loader'].dataset.features
    lag = args['lag']
    hidden = [args['hidden']]

    if args['verbose']==1:
        check_every = 1000
    else:
        check_every = 0

    if args.get('model')=='lstm':
        clstm = cLSTM(p, hidden=10)
        if args.get('opt') == 'gista':
            train_loss_list, train_mse_list = clstm_gista(clstm, X, lam=6.6, lam_ridge=1e-4, lr=0.005, max_iter=20000, check_every=check_every, truncation=5)
        else:
            train_loss_list = clstm_adam(clstm, X, lr=args['learning_rate'], niter=args['train_epochs'], check_every=check_every, verbose=args['verbose'])
    elif args.get('model')=='rnn':
        crnn = cRNN(p, hidden=hidden)
        if args.get('opt') == 'gista':
            train_loss_list, train_mse_list = crnn_gista(crnn, X, lam=6.3, lam_ridge=1e-4, lr=0.005, max_iter=20000, check_every=check_every, truncation=5)
        else:
            train_loss_list = crnn_adam(crnn, X, lr=args['learning_rate'], niter=args['train_epochs'], check_every=check_every, verbose=args['verbose'])
    else:
        cmlp = cMLP(p, lag, hidden)
        if args.get('opt') == 'gista':
            #lam = args.get('lam')
            #lam_ridge = args.get('lam_ridge')
            #max_iter = args.get('max_iter')
            #truncation = args.get('truncation')
            train_loss_list, train_mse_list = cmlp_gista(cmlp, X, lam=6.6, lam_ridge=1e-4, lr=0.005, penalty='GL', max_iter=20000, check_every=check_every, verbose=args['verbose'])
        else:
            train_loss_list = cmlp_adam(cmlp, X, lr=args['learning_rate'], niter=args['train_epochs'], check_every=check_every, verbose=args['verbose'])
        
        GC = cmlp.GC().cpu().data.numpy()
    
    print(GC)
    return GC

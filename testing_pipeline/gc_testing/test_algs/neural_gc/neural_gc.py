import sys, os
root_dir = os.path.dirname(sys.path[0])
sys.path.append(root_dir)
sys.path.append(root_dir+'/test_algs/neural_gc/')

import numpy as np
import torch
from models.cmlp import cMLP, cMLPSparse, train_model_adam, train_model_gista
from synthetic import simulate_var
import matplotlib.pyplot as plt


def run_main(args):
    
    device = torch.device('cuda')
    data = args['alg_loader'].dataset.data.to_numpy()
    X = torch.tensor(data[np.newaxis], dtype=torch.float32, device=device)
    p = args['alg_loader'].dataset.features
    lag = args['lag']
    hidden = [args['hidden']]
    cmlp = cMLP(p, lag, hidden).cuda(device=device)

    check_every = 1000
    train_loss_list = train_model_adam(cmlp, X, lr=1e-2, niter=10000, check_every=check_every)
    GC = cmlp.GC().cpu().data.numpy()
    return GC

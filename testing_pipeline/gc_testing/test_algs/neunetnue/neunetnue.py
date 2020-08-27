import sys, os
#root_dir = os.path.dirname(sys.path[0])
root_dir = os.path.dirname(os.path.realpath(__file__))

if sys.platform == 'win32':
    neunet_dir = root_dir+'\\test_algs\\neunetnue'
else:
    neunet_dir = root_dir+'/test_algs/neunetnue'

sys.path.append(root_dir)
sys.path.append(neunet_dir)

import matlab.engine
import numpy as np

def setup_paths_win(eng):
    # load matlab engine
    # have to manually add all folders to matlab file path
    eng.addpath(neunet_dir)
    eng.addpath(neunet_dir+'\\commonFunctions')
    eng.addpath(neunet_dir+'\\data')
    eng.addpath(neunet_dir+'\\exampleToolbox')
    eng.addpath(neunet_dir+'\\neunetnue')
    eng.addpath(neunet_dir+'\\neunetnue\\CODICE_NN\\Common')
    eng.addpath(neunet_dir+'\\neunetnue\\CODICE_NN\\Ffw')
    eng.addpath(neunet_dir+'\\neunetnue\\CODICE_NN\\Opt')
    eng.addpath(neunet_dir+'\\neunetnue\\UnsupervisedDrUtil')

def setup_paths_posix(eng):
    # load matlab engine
    # have to manually add all folders to matlab file path
    eng.addpath(neunet_dir)
    eng.addpath(neunet_dir+'/commonFunctions')
    eng.addpath(neunet_dir+'/data')
    eng.addpath(neunet_dir+'/exampleToolbox')
    eng.addpath(neunet_dir+'/neunetnue')
    eng.addpath(neunet_dir+'/neunetnue/CODICE_NN/Common')
    eng.addpath(neunet_dir+'/neunetnue/CODICE_NN/Ffw')
    eng.addpath(neunet_dir+'/neunetnue/CODICE_NN/Opt')
    eng.addpath(neunet_dir+'/neunetnue/UnsupervisedDrUtil')


def run_main(args):
    eng = matlab.engine.start_matlab()
    if sys.platform == 'win32':
        setup_paths_win(eng)
    else:
        setup_paths_posix(eng)
    #print(sys.path)
    #print(eng.path)
    outDir = '.'
    nr_of_processors = args['nr_of_procs']
    dat = args['alg_loader'].dataset.data.to_numpy()
    #print(dat)
    data = matlab.double(dat.tolist())
    #print(data)
    #output = eng.neunetnue_wrapper(eng.transpose(data),nr_of_processors, outDir)
    output = eng.neunetnue_wrapper(data,nr_of_processors, outDir)
    out = np.array(output._data).reshape(args['alg_loader'].dataset.features, args['alg_loader'].dataset.features)

    return out
    #np.savetxt(outDir+"\\neunetnue_output.csv", out, delimiter=",")
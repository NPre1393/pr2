import sys, os
root_dir = os.path.dirname(sys.path[0])
neunet_dir = root_dir+'\\test_algs\\neunetnue'
sys.path.append(root_dir)
sys.path.append(neunet_dir)

import matlab.engine
import numpy as np

def setup_paths(eng):
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

def run_main(args):
    eng = matlab.engine.start_matlab()

    setup_paths(eng)
    #print(sys.path)
    #print(eng.path)
    outDir = '.'
    nr_of_processors = 6
    dat = args['alg_loader'].dataset.data.to_numpy()
    #print(dat)
    data = matlab.double(dat.tolist())
    #print(data)
    #output = eng.neunetnue_wrapper(eng.transpose(data),nr_of_processors, outDir)
    output = eng.neunetnue_wrapper(data,nr_of_processors, outDir)
    out = np.array(output._data)
    print(out)
    #np.savetxt(outDir+"\\neunetnue_output.csv", out, delimiter=",")
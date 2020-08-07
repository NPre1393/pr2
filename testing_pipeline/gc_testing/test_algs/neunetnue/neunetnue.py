import matlab.engine
import numpy as np

def setup_paths(eng):
    # load matlab engine
    # have to manually add all folders to matlab file path
    eng.addpath('commonFunctions')
    eng.addpath('data')
    eng.addpath('exampleToolbox')
    eng.addpath('neunetnue')
    eng.addpath('neunetnue\\CODICE_NN\\Common')
    eng.addpath('neunetnue\\CODICE_NN\\Ffw')
    eng.addpath('neunetnue\\CODICE_NN\\Opt')
    eng.addpath('neunetnue\\UnsupervisedDrUtil')

def run_main(args):
    eng = matlab.engine.start_matlab()

    setup_paths(eng)

    outDir = '.'
    nr_of_processors = 6
    output = eng.neunetnue_wrapper(eng.transpose(args['alg_loader'].dataset.data),nr_of_processors)
    out = np.array(output._data)
    #np.savetxt(outDir+"\\neunetnue_output.csv", out, delimiter=",")
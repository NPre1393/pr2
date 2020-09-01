import sys, os
#root_dir = os.path.dirname(sys.path[0])
neunet_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(neunet_dir)

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
    dat = args['alg_loader'].dataset.data.to_numpy()
    outDir = args['alg_loader'].result_path
    nr_of_processors = args['nr_of_procs']
    if args['platform'] == 'octave':
        from oct2py import Oct2Py
        eng = Oct2Py()
        if sys.platform == 'win32':
            setup_paths_win(eng)
        else:
            setup_paths_posix(eng)        
        #data = eng.transpose(eng.double(dat.tolist()))
        data = eng.double(dat.tolist())
        out = eng.neunetnue_wrapper_octave(data,nr_of_processors, outDir)
    else:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        if sys.platform == 'win32':
            setup_paths_win(eng)
        else:
            setup_paths_posix(eng)          
        data = matlab.double(dat.tolist())
        output = eng.neunetnue_wrapper(data,nr_of_processors, outDir)
        out = np.array(output._data).reshape(args['alg_loader'].dataset.features, args['alg_loader'].dataset.features)

    return out

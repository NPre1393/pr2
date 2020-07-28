import matlab.engine
import numpy as np

# load matlab engine
eng = matlab.engine.start_matlab()

# have to manually add all folders to matlab file path
eng.addpath('commonFunctions')
eng.addpath('data')
eng.addpath('exampleToolbox')
eng.addpath('neunetnue')
eng.addpath('neunetnue\\CODICE_NN\\Common')
eng.addpath('neunetnue\\CODICE_NN\\Ffw')
eng.addpath('neunetnue\\CODICE_NN\\Opt')
eng.addpath('neunetnue\\UnsupervisedDrUtil')

dat_dir = 'exampleToolbox\\realization_5000p_1.mat'
loadedData = eng.load(dat_dir)
outDir = '.'
nr_of_processors = 6
output = eng.neunetnue_wrapper(eng.transpose(loadedData['data']),nr_of_processors)
out = np.array(output._data)
np.savetxt(outDir+"\\neunetnue_output.csv", out, delimiter=",")
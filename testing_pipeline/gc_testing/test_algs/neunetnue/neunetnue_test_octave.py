from oct2py import Oct2Py

# load oct2py engine
oct = Oct2Py()

# have to manually add all folders to ocatve file path
oct.addpath('commonFunctions')
oct.addpath('data')
oct.addpath('exampleToolbox')
oct.addpath('neunetnue')
oct.addpath('neunetnue\\CODICE_NN\\Common')
oct.addpath('neunetnue\\CODICE_NN\\Ffw')
oct.addpath('neunetnue\\CODICE_NN\\Opt')
oct.addpath('neunetnue\\UnsupervisedDrUtil')

dat_dir = 'exampleToolbox\\realization_5000p_1.mat'
loadedData = oct.load(dat_dir)
output = oct.neunetnue_wrapper(oct.transpose(loadedData.data),6,'outputdir')
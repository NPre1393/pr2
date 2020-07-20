% nameDataDir  = 'exampleToolbox\\';
% dataDir = ['C:\\Users\\mpres\\Documents\\MATLAB\\neunetnue\\' nameDataDir];
% dataFileName = 'realization_5000p_1';
% dataLabel = '';
% dataExtension = '.mat';
% listRealization = dir([dataDir [dataFileName '*' dataLabel '*' dataExtension]]);
% 
% dataLoaded = load([dataDir listRealization(1,1).name]);
%dat_dir = 'C:\\Users\\mpres\\Documents\\MATLAB\\neunetnue\\exampleToolbox\\realization_5000p_1.mat';
dat_dir = 'exampleToolbox\\realization_5000p_1.mat';
loadedData = load(dat_dir)
output = neunetnue_wrapper(transpose(loadedData.data),6,'outputdir')
%% Experiment using data from ...
% 
% Some explanations about how to set up the methods
% 
%     Method order: please take the order into account because afterwards
%     you should set autoPairwiseTarDriv or handPairwiseTarDriv that need
%     the precise order of the methods
% 
%     neunetnue
%%    neunetnue
% 
%     nnData                 = ...;
%     idTargets              = [1 2 3 4];
%     idDrivers              = [5 6 1 2];
%     idOtherLagZero         = [3,5,8,1];
%     modelOrder             = 8;
%     firstTermCaseVect      = [1 0];
%     secondTermCaseVect     = [1 1];
%     multi_bivAnalysis      = 'multiv';
%     eta                    = 0.01;
%     alpha                  = 0.01;
%     fracTrainSet           = 3/4;
%     actFunc                = {@sigmoid @identity};
%     numEpochs              = 30;
%     bias                   = 0;
%     epochs                 = 4000;
%     threshold              = 20/100;
%     dividingPoint          = 3/4;
%     valStep                = 5;
%     valThreshold           = 0.1/100;
%     learnAlg               = @resilientBackPropagation;
%     rbpIncrease            = 1.1;
%     rbpDecrease            = 0.9;
%     rangeW                 = 1;
%     coeffHidNodes          = 0.3;
%     % ******** Set the following fields together *******
%     paramsNonUniNeuralNet.genCondTermFun         = @generateConditionalTerm;%@generateCondTermLagZero
%     paramsNonUniNeuralNet.usePresent             = 0;
%     % **************************************************


    %% Set MuTE folder path including also all the subfolders, for instance
    clear all;
    %mutePath = ('C:\\Users\\mpres\\Documents\\MATLAB\\neunetnue\\'); % Adjust according to your path -> just an example: mutePath = '/home/alessandro/Scrivania/MuTE/';
%     cd(mutePath);
%     addpath(genpath(pwd));
    
    nameDataDir  = 'exampleToolbox\\';
    
    %% Set the directory in which the data files are stored. In this directory the outcome of the experiments will be stored too.
    %dataDir      = ['/Users/alessandromontalto/Dropbox/MuTE_onlineVersion/' nameDataDir]; % Adjust according to your path -> just as example: dataDir = ['/home/alessandro/Scrivania/MuTE/' nameMainDir];
    dataDir      = ['C:\\Users\\mpres\\Documents\\MATLAB\\neunetnue\\' nameDataDir];
    % *****************************************************
    %% PAY ATTENTION: if you are able to run the parallel session you can set numProcessors > 1
    numProcessors               = 6;
    % *****************************************************


        
    %% EXPERIMENTS
    

    cd(dataDir);

%%  Defining the strings to load the data files
    dataFileName    = 'realization_5000p_1';
    dataLabel       = '';
    dataExtension   = '.mat';
    

%%     making storing folders
    
    resDir          = [dataDir dataFileName '_' dataLabel '/'];
    if (~exist([dataDir 'resDir'],'dir'))
        mkdir(resDir);
    end
    copyDir   = [resDir 'entropyMatrices' dataLabel '/'];
    if (~exist([resDir 'copyDir'],'dir'))
        mkdir(copyDir);
    end
    

%%    defining result directories 

    cd(dataDir);
    resultDir           = [resDir 'results' dataLabel '/'];
    if (~exist([resDir 'resultDir'],'dir'))
        mkdir(resultDir);
    end
    
%%  Defining the experiment parameters
    channels             = 1:5;
    samplingRate         = 1;
    pointsToDiscard      = 0;
    listRealization      = dir([dataDir [dataFileName '*' dataLabel '*' dataExtension]]);
    autoPairwiseTarDriv  = 1;
    handPairwiseTarDriv  = 0;
    
%   neunetnue parameters
    threshold           = 0.008;
    valThreshold        = 0.6;
    numHiddenNodes      = 0.3;
    
%  
       %% STATISTICAL METHODS
    dataLoaded = load([dataDir listRealization(1,1).name]);

    fprintf('\n******************************\n\n');
    disp('Computing statistical methods...');
    fprintf('\n\n');

    tic
%     [output1,params1]               = parametersAndMethods(listRealization,samplingRate,pointsToDiscard,channels,autoPairwiseTarDriv,...
%                                       handPairwiseTarDriv,resultDir,dataDir,copyDir,numProcessors,...
%                                       'linue',[],[],[],5,'multiv',5,5,'bayesian',@linearEntropy,[1 0],[1 1],@generateConditionalTerm,0,...
%                                       'linnue',[],[],[],5,'multiv',@evaluateLinearNonUniformEntropy,[1 1],100,0.05,@generateConditionalTerm,0,...
%                                       'binue',[],[],[],5,'multiv',6,@conditionalEntropy,@quantization,[1 0],[1 1],100,0.05,20,...
%                                       @generateConditionalTerm,0,...
%                                       'binnue',[],[],[],5,'multiv',6,@evaluateNonUniformEntropy,@quantization,[1 1],100,0.05,@generateConditionalTerm,0,0,...
%                                       'neunetue',[],[],[],5,[1 1],'multiv',[],[],{@sigmoid @identity},30,0,4000,2/3,15,...
%                                       valThreshold,@resilientBackPropagation,1.1,0.9,1,numHiddenNodes,100,20,0.05,@generateConditionalTerm,0,...
%                                       'neunetnue',[],[],[],[],5,[1 0],[1 1],'multiv',[],[],{@sigmoid @identity},30,0,4000,threshold,2/3,15,...
%                                       valThreshold,@resilientBackPropagation,1.1,0.9,1,numHiddenNodes,@generateConditionalTerm,1,...
%                                       'nnue',[],[],[],5,'multiv',[1 1],100,'maximum',10,nnMexa64Path,mutePath,0.05,10,@generateConditionalTerm,0,...
%                                       'nnnue',[],[],[],5,'multiv',[1 1],100,'maximum',10,@nearNeiConditionalMutualInformation,...
%                                       @evalNearNeiTestSurrogates2rand,nnMexa64Path,mutePath,0.05,@generateConditionalTerm,0);

    [output,params] = parametersAndMethods_new(dataLoaded.data,samplingRate,pointsToDiscard,channels,autoPairwiseTarDriv,...
                                        handPairwiseTarDriv,resultDir,dataDir,copyDir,numProcessors,...
                                        'neunetnue',[],[],[],[],5,[1 0],[1 1],'biv',[],[],{@sigmoid @identity},30,0,4000,threshold,2/3,15,...
                                        valThreshold,@resilientBackPropagation,1.1,0.9,1,numHiddenNodes,@generateConditionalTerm,1);

    %close all
    toc
%     out_mat = outputToStore.reshapedMtx;
%     out_mat(find(out_mat~=0))=1

    fprintf('\n\n');
    disp('...computation done!');
    fprintf('\n\n');
    
    numTargets = size(dataLoaded.data,1);
    A = reshape(output.transferEntropy,numTargets-1,numTargets);
    if find(isnan(A));
        [idRow,idCol] = find(isnan(A));
        for idR = idRow
            for idC = idCol
                A(idR,idC) = 0;
            end
        end
    end
    reshapedMtx = zeros(numTargets);
    reshapedMtx(:,1) = [0;A(:,1)];
    for j = 2:numTargets-1
        reshapedMtx(:,j) = [A(1:j-1,j);0;A(j:end,j)];
    end
    reshapedMtx(:,numTargets) = [A(:,numTargets);0];
    
    out_mat = reshapedMtx;
    out_mat((out_mat~=0))=1
    %close all
%    cd(mutePath);
    %exit;


    
    
    
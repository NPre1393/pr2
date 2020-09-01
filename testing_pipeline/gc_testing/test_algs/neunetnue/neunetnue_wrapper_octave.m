function [out_mat] = neunetnue_wrapper(data,numProc,saveData)
    numProcessors = numProc;

    % time series data should be n=time, m=variables
    [n,m] = size(data);
    %%  Defining the experiment parameters
    channels = 1:m;
    samplingRate         = 1;
    pointsToDiscard      = 0;
    %listRealization      = dir([dataDir [dataFileName '*' dataLabel '*' dataExtension]]);
    autoPairwiseTarDriv  = 1;
    handPairwiseTarDriv  = 0;

%   neunetnue parameters
    threshold           = 0.008;
    valThreshold        = 0.6;
    numHiddenNodes      = 0.3;
    
%  
       %% STATISTICAL METHODS

    fprintf('\n******************************\n\n');
    disp('Computing NeuNetNUE method...');
    fprintf('\n\n');

    tic
    [output,params] = parametersAndMethods_octave(data,samplingRate,pointsToDiscard,channels,autoPairwiseTarDriv,...
                                        handPairwiseTarDriv,numProcessors,...
                                        'neunetnue',[],[],[],[],5,[1 0],[1 1],'biv',[],[],{@sigmoid @identity},30,0,4000,threshold,2/3,15,...
                                        valThreshold,@resilientBackPropagation,1.1,0.9,1,numHiddenNodes,@generateConditionalTerm,1);

    %close all
    toc

    fprintf('\n\n');
    disp('...computation done!');
    fprintf('\n\n');

    numTargets = size(data,2);
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
    out_mat((out_mat~=0))=1;
    
return;
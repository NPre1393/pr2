function [output,params] = parametersAndMethods(data,sampling,pointsToDiscard,channels,autoPairwiseTarDriv,handPairwiseTarDriv,numProcessors,varargin)

    % Data: Time series in the rows
    realizations        = 1;
    numSeries           = length(channels);
   
    % ***************************************************************************************************
    %% Setting methods
    
     method_caseVect = find(strcmp('neunetnue',varargin));
%    neunetnue = 0;
    %    if (~isempty(method_caseVect))
    neunetnue                                          = 1;
    nnIdTargets                                        = varargin{method_caseVect+2};
    nnIdDrivers                                        = varargin{method_caseVect+3};
    nnIdOtherLagZero                                   = varargin{method_caseVect+4};
    nnModelOrder                                       = varargin{method_caseVect+5};
    nnFirstTermCaseVect                                = varargin{method_caseVect+6};
    nnSecondTermCaseVect                               = varargin{method_caseVect+7};
    nnAnalysisType                                     = varargin{method_caseVect+8};
    nnEta                                              = varargin{method_caseVect+9};
    nnAlpha                                            = varargin{method_caseVect+10};
    nnActFunc                                          = varargin{method_caseVect+11};
    nnNumEpochs                                        = varargin{method_caseVect+12};
    nnBias                                             = varargin{method_caseVect+13};
    nnEpochs                                           = varargin{method_caseVect+14};
    nnThreshold                                        = varargin{method_caseVect+15};
    nnDividingPoint                                    = varargin{method_caseVect+16};
    nnValStep                                          = varargin{method_caseVect+17};
    nnValThreshold                                     = varargin{method_caseVect+18};
    nnLearnAlg                                         = varargin{method_caseVect+19};
    nnRbpIncrease                                      = varargin{method_caseVect+20};
    nnRbpDescrease                                     = varargin{method_caseVect+21};
    nnRangeW                                           = varargin{method_caseVect+22};
    nnCoeffHidNodes                                    = varargin{method_caseVect+23};
    nnGenCondTermFun                                   = varargin{method_caseVect+24};
    nnUsePresent                                       = varargin{method_caseVect+25};
%    else
%        nnData                                             = [];
%    end

    % ***************************************************************************************************
    %% Setting the parameters for each method:
    %if (neunetnue)
        paramsNonUniNeuralNet = createNeunetnueParams(numSeries,nnIdTargets,nnIdDrivers,nnIdOtherLagZero,nnModelOrder,nnFirstTermCaseVect,...
                                nnSecondTermCaseVect,nnAnalysisType,nnEta,nnAlpha,nnActFunc,nnNumEpochs,nnBias,nnEpochs,...
                                nnThreshold,nnDividingPoint,nnValStep,nnValThreshold,nnLearnAlg,nnRbpIncrease,nnRbpDescrease,...
                                nnRangeW,nnCoeffHidNodes,nnGenCondTermFun,nnUsePresent);
     %   if (autoPairwiseTarDriv == 1)
            [tarDrivRows] = allAgainstAll (channels);
            paramsNonUniNeuralNet.idTargets = tarDrivRows(1,:);
            paramsNonUniNeuralNet.idDrivers = tarDrivRows(2,:);
      %  end
    %end
    % ***************************************************************************************************
    %% Putting all the parameters in one structure
    
    %if (neunetnue)
        params.methods.neunetnue = paramsNonUniNeuralNet;
    %end
    
    % ***************************************************************************************************
    %% Calling methods
    
    output = callingMethods(data,params);

return;
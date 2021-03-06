function [W learnParams ] = gradientDescentWithMoment(W,gradW,learnParams)
% function [W learnParams ] = gradientDescentWithMoment(W,gradW,learnParams)
% W and gradW are cell arrays of matrices.
    eta   = learnParams.eta;
    alpha = learnParams.alpha;
    WLength=length(W);
    %firstIter = true;
    %deltaW=cell(1,numOfweightLayers);
    
    if (isfield(learnParams,'deltaW'))
        for i=1:WLength
            dW=-eta*gradW{i};
 %           dB=-eta*gradB{i};
            W{i}=W{i}+dW+alpha*learnParams.deltaW{i};
%            B{i}=B{i}+dB+alpha*learnParams.deltaB{i};
            learnParams.deltaW{i}=dW;
  %          learnParams.deltaB{i}=dB;
        end
    else
        setLearnParams(learnParams,'deltaW',cell(1,WLength));
   %     setLearnParams(learnParams,'deltaB',cell(1,numOfweightLayers));
        for i=1:WLength
            dW=-eta*gradW{i};
            %dB=-alpha*gradB{i};
            W{i}=W{i}+dW;
            %B{i}=ffwNet.B{i}+dB;
            learnParams.deltaW{i}=dW;
            %learnParams.deltaB{i}=dB;
        end
    end        
return;

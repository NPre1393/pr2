function [output] = callingMethods(dataNN,params)
% 
% Syntax:
% 
% callingMethods(data,methodCell,params);
% 
% Description:
% 
% the function calls each method and saves the results in the method folder

% extracting the global parameters
% globalParameters        = params.globalParameters;

methodNames      = fieldnames(params.methods);

methodName      = methodNames(1,1);
currentMethod   = str2func(methodName{1,1});
output     = currentMethod(dataNN,params.methods.('neunetnue'));



return;
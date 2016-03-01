function hiddenError = hiddenErrorGradient(hiddenNeurons, wHiddenOutput, outputErrorGrad)
% Fucntion for computing the error of each unit in the hidden layer;
% hiddenNeurons are the actual outputs of units in the hiddden layer;
% wHiddenOutput is a vector of weights of the connection from the actual unit to a unit
%                   in the next hidden (in this case, output) layer;
% outputErrorGrad is a vector of the errors of each unit in the output layer
  
  nHidden = rows(wHiddenOutput) - 1;
  % nOutput = columns(wHiddenOutput);
  % hiddenError = zeros(length(hiddenNeurons), 1);
  
  for j = 1:nHidden
    hiddenError(j) = hiddenNeurons(j) * (1 - hiddenNeurons(j)) * sum(outputErrorGrad .* (wHiddenOutput(j + 1, :))');
  end
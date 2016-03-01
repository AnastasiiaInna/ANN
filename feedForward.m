function [hiddenNeurons outputNeurons] = feedForward(inputNeurons, wInputHidden, wHiddenOutput)
  % Function for feeding inputs forward
  
  nHidden = columns(wInputHidden);
  nOutputs = columns(wHiddenOutput);
  
  hiddenNeurons = zeros(nHidden, 1);
  outputNeurons = zeros(nOutputs, 1);
  
  % Calculate Hidden Layer values
  % The first column of wInputHidden contains bias values
  for j = 1:nHidden
    hiddenNeurons(j) = sum(inputNeurons .* wInputHidden(2:end, j)) + wInputHidden(1, j);
  end
  hiddenNeurons = sigmoid(hiddenNeurons);
  
  % Calculate Output Layer values
  % The first column of wHiddenOutput contains bias values
  for k = 1:nOutputs
    outputNeurons(k) = sum(hiddenNeurons .* wHiddenOutput(2:end, k)) + wHiddenOutput(1, k);
  end
  outputNeurons = sigmoid(outputNeurons);
end
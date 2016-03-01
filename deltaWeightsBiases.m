function [deltaHiddenOutput deltaInputHidden deltaBiasHiddenOutput deltaBiasInputHidden] = deltaWeightsBiases(inputNeurons, hiddenNeurons, outputNeurons, wHiddenOutput, deltaHiddenOutput, deltaInputHidden, desiredValue, learningRate, momentum)
% Function for modifying weights and biases according to the units in both hidden and output layers
  
  outputError = zeros(length(outputNeurons), 1);
  outputError = outputErrorGradient(outputNeurons, desiredValue);
  for k = 1:length(outputNeurons)
    deltaHiddenOutput(:, k) = learningRate * outputError(k) * hiddenNeurons + momentum * deltaHiddenOutput(:, k);
  end
  
  hiddenError = zeros(length(hiddenNeurons), 1);
  hiddenError = hiddenErrorGradient(hiddenNeurons, wHiddenOutput, outputError);
  for j = 1:length(hiddenNeurons)
    deltaInputHidden(:, j) = learningRate * hiddenError(j) * inputNeurons + momentum * deltaInputHidden(:, j) ;
  end
  
  deltaBiasHiddenOutput = learningRate * outputError';
  deltaBiasInputHidden = learningRate * hiddenError;
  
end
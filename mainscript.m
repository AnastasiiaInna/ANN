[tvec tlab tstv tstl] = readSets(); 
[mu trmx] = prepTransform(tvec, 40);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);

tlab = tlab + 1;
tstl = tstl + 1;

% rand();
% rndstate = rand("state");
% save rndstate.txt rndstate;

load rndstate.txt
rand("state", rndstate);

% --- Setup the parameters --- %
nInputs  = 40;  % 20x20 Input Images of Digits
nHidden = 40;   % 25 hidden units
nOutputs = 10;  % 10 labels, from 1 to 10  
 
learningRate0 = 0.9;
nEpoch = 100;
currentEpoch = 1;

# --- Create random samples --- #

nSamples = rows(tvec);
[tvecRandom tlabRandom] = randomSamples(tvec, tlab, nSamples);

# --- #

% [MSE Accuracy Error Rejected wInputHidden wHiddenOutput] = trainingEpoch(tvecRandom, tlabRandom, wInputHidden, wHiddenOutput, learningRate);
prevAccuracy = 0;
prevPrevAccuracy = 0;
Accuracy = 0;

learningRate = learningRate0;

wInputHidden = initializeWeights(nInputs, nHidden);   % first row of W handles the "bias" terms
wHiddenOutput = initializeWeights(nHidden, nOutputs); % first row of W handles the "bias" terms

currentEpoch = 1;
t = mktime(localtime(time())); % for learning times
while ((prevPrevAccuracy <= Accuracy && prevAccuracy <= Accuracy) && currentEpoch <= nEpoch)
  prevPrevAccuracy = prevAccuracy;
  prevAccuracy = Accuracy;
  % [MSE Error Rejected] = trainingEpoch(tvecRandom, tlabRandom, nHidden, nOutputs, learningRate);  
  [MSE Accuracy Error Rejected wInputHidden wHiddenOutput] = trainingEpoch(tvecRandom, tlabRandom, wInputHidden, wHiddenOutput, learningRate);  
  learningRate = learningRate0 / (1 + currentEpoch / nSamples);
  currentEpoch += 1;
endwhile
learningTime = mktime(localtime(time()))- mktime(localtime(t));

save "-ascii" wInputHidden.dat wInputHidden 
save "-ascii" wHiddenOutput.dat wHiddenOutput

% --- Test Set --- %
nTestSamples = rows(tstv);
[tstvRandom tstlRandom] = randomSamples(tstv, tstl, nTestSamples);
[testMSE testAccuracy testError testRejected] = accuracyTestSet(tstvRandom, tstlRandom, wInputHidden, wHiddenOutput);

% --- Training Set --- #
[MSE Accuracy Error Rejected]
% --- Test Set --- %
[testMSE testAccuracy testError testRejected]
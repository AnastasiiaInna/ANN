[tvec tlab tstv tstl] = readSets(); 
[mu trmx] = prepTransform(tvec, 40);
tstv = pcaTransform(tstv, mu, trmx);
tstl = tstl + 1;

% --- Setup the parameters --- %
nInputs  = 40;  % 20x20 Input Images of Digits
nHidden = 40;   % 25 hidden units
nOutputs = 10;  % 10 labels, from 1 to 10  

wInputHidden = load(wInputHidden.dat);   % first row of W handles the "bias" terms
wHiddenOutput = load(wHiddenOutput.dat); % first row of W handles the "bias" terms

nTestSamples = rows(tstv);
[tstvRandom tstlRandom] = randomSamples(tstv, tstl, nTestSamples);
[testMSE testAccuracy testError testRejected] = accuracyTestSet(tstvRandom, tstlRandom, wInputHidden, wHiddenOutput);

[testMSE testAccuracy testError testRejected]
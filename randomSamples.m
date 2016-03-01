function [tvecRandom tlabRandom] = randomSamples(tvec, tlab, N)
  
% Function for creating the set consisted of random samples from the initial dataset;
% N is a desired number of samples;

  rndIDX = randperm(rows(tvec)); 
  tvecRandom = tvec(rndIDX(1:N), :); 
  tlabRandom = tlab(rndIDX(1:N), :); 
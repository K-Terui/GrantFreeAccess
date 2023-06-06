function [MD,FA,AER] = Compute_MDandFA(D,D_hat,N,K)
% Compute Miss detection and False alarm
% Input: (D: Active user index, D_hat: Estimate active user index,N: # of
% user, K: # of active user)
% Output: [MD: Miss detection probability, FA: False alarm probability
MD = 1 - sum(ismember(D,D_hat))/K;
FA = (numel(D_hat) - sum(ismember(D,D_hat)))/(N - K);
AER = (MD*K + FA*(N-K))/N;
end
function [Xhat, SetEst] = MMVOMP_nzknown(Y, A, K)
% MMVOMP algorithm (Case: num. non-zero elements is known at the receiver)
% min. ||Y - A\hat{X}||_0, s.t. Y = AX + N, ||x_i||_0 = K (forall i) 
% input
% Y      : received signal (M, J)
% A      : mesurement matrix (M, N)
% K      : num. non-zero elements at the each column vector of X
% output
% Xhat   : reconstructed block sparse matrix (N, J)
% SetEst : indeces set of reconstructed sparse vector
% variable
% Xtilde : estimated non-sparse matrix (K, J)


% initialize
Xhat   = zeros(size(A, 2), size(Y, 2)); %reconstructed vector (block sparse)   
R      = zeros(size(Y));    %residual (M, J)
SetEst = int16.empty;       %index set of estimated active UEs
A_hat  = double.empty;      %selected support

% main loop
for s = 1 : K
    % residual update
    R = Y - A * Xhat;

    % select the maximum correlated index over all column spaces
    [~, p] = max(sum(abs(A' * R), 2));
    
    % add index to the index set
    SetEst(s) = p;

    % derive the least square solution
    A_hat(:, s) = A(:, SetEst(s));
    Xtilde  = pinv(A_hat) * Y;

    % sparse reconstruction
    Xhat(SetEst, :) = Xtilde;

end

end
function [xhat, SetEst] = OMP_nzknown(y, A, K)
% OMP algorithm (Case: num. non-zero elements is known at the receiver)
% min. ||y - A\hat{x}||_0, s.t. y = Ax + n, ||x||_0 = K 
% input
% y      : received signal
% A      : mesurement matrix
% K      : num. non-zero elements 
% output
% xhat   : reconstructed sparse vector
% SetEst : indeces set of reconstructed sparse vector


% initialize
xhat   = zeros(size(A, 2), 1); %reconstructed vector (sparse)   
r      = zeros(size(y));    %residual
SetEst = int16.empty;       %index set of estimated active UEs
A_hat  = double.empty;      %selected support

% main loop
for s = 1 : K
    % residual update
    r = y - A * xhat;

    % select the maximum correlated index
    [~, p] = max(abs(A' * r));
    
    % add index to the index set
    SetEst(s) = p;
    

    % derive the least square solution
    A_hat(:, s) = A(:, SetEst(s));
    xtilde_GRF  = pinv(A_hat) * y;

    % sparse reconstruction
    xhat(SetEst) = xtilde_GRF;

end

end
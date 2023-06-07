function NMSE = ChannelEstimation_MMSE(A, Y, H, Xhat, SetEst, nvar)
% Channel estimation method with MMSE for grant free access
% After the channel estimation, calculate the NMSE

% initialize
M = size(A, 1);
J = size(Y, 2);

for j = 1 : J
    % initialize
    Aest = A(:, SetEst(:, j));
    y    = Y(:, j);
    
    % MMSE
    xmmse = Aest' * inv(Aest * Aest' + nvar * eye(M)) * y;
    
    % sparse vector reconstruction
    Xhat(SetEst(:, j), j) = xmmse;

end

% NMSE calculation
NMSE = norm(H - Xhat, 'fro')^2 / norm(H, 'fro')^2;
end
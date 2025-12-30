function sigma_reg = regularize_covariance(sigma, max_cond)
% REGULARIZE_COVARIANCE applies Tikhonov regularization to a covariance matrix.
%
%   sigma_reg = regularize_covariance(sigma, max_cond)
%
%   Ensures the condition number of the matrix does not exceed max_cond
%   by adding a diagonal loading factor (lambda * I).

% 1. Get eigenvalues
eigs = eig(sigma);
min_eig = min(eigs);
max_eig = max(eigs);

% 2. Calculate current condition number
current_cond = max_eig / min_eig;

lambda = 0;
% 3. Check if regularization is needed
if abs(current_cond) > max_cond
    % Analytical solution for lambda to reach target condition number
    lambda = (max_eig - max_cond * min_eig) / (max_cond - 1);
end

% 4. Apply regularization
if lambda > 0
    sigma_reg = sigma + lambda * eye(size(sigma));
else
    sigma_reg = sigma;
end

end

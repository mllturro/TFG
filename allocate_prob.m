function Z_mat = allocate_prob(n,m,p_l_vec)
%*************************************************
% Originally written by Cabrera et al, 2019. Updated Jan 2021 by M.
% Cabrera-Bean and M. Llobet.
% (See Function 2 in the thesis's main document.)
%*************************************************
%    Input:
% n       = no. of instances
% m       = no. of classifiers
% p_l_vec = 2^m x 1 vector containing the probabilities {p_kl} (the
%           probabilities of all possible sets of classification results)
%*************************************************
%   Output:
% Z_mat   = n x m natrix containing n instances labeled by m classifiers
%*************************************************
cdf = cumsum(p_l_vec);
Z_tilde = rand(n,1);
Z_mat = zeros(n,1);

for l = 1:length(cdf)-1
    Z_mat(Z_tilde > cdf(l)) = l;
end
Z_mat = de2bi(Z_mat);
bi_dim = size(Z_mat,2);
if bi_dim < m
    Z_mat = [Z_mat zeros(n,m-bi_dim)];
end
Z_mat = flip(Z_mat,2);

% Random assignation of the first classifier
for ii = 1:size(Z_mat,1)
    rnd_vec = randperm(size(Z_mat,2));
    ii_new = zeros(1,size(Z_mat,2));
    ii_old = Z_mat(ii,:);
    for jj = 1:size(Z_mat,2)
        ii_new(jj) = ii_old(rnd_vec(jj));
    end
    Z_mat(ii,:) = ii_new;
end

end
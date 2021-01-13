function S_mat = scorematrix(R_mat)
%*************************************************
% Written by M. Llobet, Jan 2021.
% (See Eq. 1.12 in the thesis's main document.)
%*************************************************
%    Input:
% R_mat = Estimated inter-classifier covariance matrix (Eq. 1.7)
%*************************************************
%    Output:
% S_mat = Estimated score matrix (Eq. 1.12)
%*************************************************
m = size(R_mat,1); % No. of classifiers
S_mat = zeros(m);
for ii = 1:m
    for jj = 1:m
        sum_ij = 0;
        r_ij = R_mat(ii,jj);
        kvec = [(1:1:ii-1) (ii+1:1:m)]; lvec = [(1:1:jj-1) (jj+1:1:m)]; % i,j indices skipped
        for kk = kvec
            for ll = lvec
                r_kl = R_mat(kk,ll); r_il = R_mat(ii,ll); r_kj = R_mat(kk,jj);
                op_ij = abs(r_ij*r_kl - r_il*r_kj);
                sum_ij = sum_ij + op_ij ;
            end
        end
        S_mat(ii,jj) = sum_ij;
    end
end
S_mat = .5.*(S_mat + S_mat'); % S is forced to be symmetric

end
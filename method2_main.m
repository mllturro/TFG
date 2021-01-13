function Z_mat = method2_main(str,gt_vec,alph_k_vec_0,alph_k_vec_1,eta_vec,phi_vec)
%*************************************************
% Written by M. Llobet in Jan 2021 as an adaptation of the proposed method for artificially
% generating binary correlated data originally presented by Jaffe et al, 2016.
%*************************************************
%    Input:
% str          = 1 x K vector containing the classifiers' correlation 
%                structure (e.g. [5 5 5 1] - 4 correlation groups, the 
%                first three each containing 5 classifiers)
% gt_vec       = n x 1 vector containing the decisions' ground truth
% alph_k_vec_1 = 1 x K vector containing Pr( alph_k = 1 | Y = 1 ), k = 1:K
% alph_k_vec_0 = 1 x K vector containing Pr( alph_k = 0 | Y = 0 ), k = 1:K
% phi_vec      = 1 x m vector containing Pr (f_j = 1 | Y = 1 ), j = 1:m
% eta_vec      = 1 x m vector containing Pr (f_j = 0 | Y = 0 ), j = 1:m
%*************************************************
%  Output:
% Z_mat   = n x m matrix containing n decision results for all m classifiers
%
%*************************************************

    K = length(str); % No. of correlation groups
    m = sum(str); % No. of classifiers
    n = length(gt_vec); % No. of decisions
    NP = sum(gt_vec); % No. of positives
    
    Z_mat = zeros(n,m); Z_alph_k = zeros(n,1);

    c_count = 0; 
    for kk = 1:K
        if NP > 0; Z_alph_k(gt_vec == 1) = binornd(1,alph_k_vec_1(kk),NP,1); end
        if NP ~= n; Z_alph_k(gt_vec == 0) = binornd(1,1-alph_k_vec_0(kk),n-NP,1); end
        NP_k = sum(Z_alph_k);
        
        for nn = c_count+1:str(kk)+c_count
            if NP_k > 0; Z_mat(Z_alph_k == 1,nn) = binornd(1,phi_vec(nn),NP_k,1); end
            if NP ~= n; Z_mat(Z_alph_k == 0,nn) = binornd(1,1-eta_vec(nn),n-NP_k,1); end
        end
        c_count = c_count + str(kk);
    end
    clear kk Z_alph_k NP_k c_count 
    
end
function Z_mat = method1_main(str,gt_vec,eta_vec,psi_vec)
%*************************************************
% Originally written by M. Cabrera et al, 2019. Updated Jan 2021 by M.
% Cabrera-Bean and M. Llobet.
%*************************************************
%    Input:
% str     = 1 x K vector containing the classifiers' correlation structure
%           (e.g. [5 5 5 1] - 4 correlation groups, the first three each
%           containing 5 classifiers)
% gt_vec  = n x 1 vector containing the decisions' ground truth
% eta_vec = 1 x K vector containing prob. of success for classifiers in each
%           correlation group conditioned to Y = 0
% psi_vec = same as eta_vec, but conditioned to Y = 1
%*************************************************
%   Output:
% Z_mat     = n x m natrix containing n instances labeled by m classifiers
%*************************************************

eps = 0.5; % parameter epsilon fixed at 0.5

K = length(str); % no. of correlation groups
m = sum(str); % no. of classifiers
n = length(gt_vec); % no of. decisions
NP = sum(gt_vec); % no. of positives
    
v_cell_0 = cell(1,K);
v_cell_1 = cell(1,K);

Z_mat = zeros(n,m); 

c_count = 0;
for kk = 1:K
    m_k = str(kk); % No. of classifiers im the correlatiom group kk
    
    % Generate the tree of conditional probabilities (Eq 2.2)
    v_j_0 = cell(1,m_k); v_j_0{1} = eta_vec(kk);
    v_j_1 = cell(1,m_k); v_j_1{1} = psi_vec(kk);

    if m_k > 1
        for ii = 2:m_k
            v_i = sort(rand(1,2^(ii-1))); 
            v_i(1) = min(eps*rand,v_i(1)); 
            v_i(end) = max(1-eps*rand,v_i(end)); 
            v_j_0{ii} = flip(v_i);
            
            v_i = sort(rand(1,2^(ii-1)));
            v_i(1) = min(eps*rand,v_i(1)); 
            v_i(end) = max(1-eps*rand,v_i(end)); 
            v_j_1{ii} = v_i;
        end
    end
    
    % Compute probabilities {p_kl} as in Eq (2.1)
    p_kl_0 = compute_prob(v_j_0,0);
    v_cell_0{kk} = p_kl_0;
    p_kl_1 = compute_prob(v_j_1,1);
    v_cell_1{kk} = p_kl_1;
    
    % Generate correlated classification results:
    if NP > 0
        Z_mat(gt_vec==1,c_count+1:c_count+m_k) = allocate_prob(NP,m_k,p_kl_1);
    end
    
    if NP < n
        Z_mat(gt_vec==0,c_count+1:c_count+m_k) = allocate_prob(n-NP,m_k,p_kl_0);
    end
    c_count = c_count + m_k;
end

end
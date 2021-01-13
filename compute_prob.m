function p_l_vec = compute_prob(v_j_cell,gt)
%*************************************************
% Originally written by M. Cabrera et al, 2019. Updated Jan 2021 by M.
% Cabrera-Bean and M. Llobet.
% (See Function 1 in thesis's main document.)
%*************************************************
%    Input:
% v_j_cell = 1 x m ( m = no. of classifiers) cell containing the conditional
% probabilities making up the tree diagram (Eq. 2.2)
% gt       = ground truth label (Y = 0, or Y = 1)
%*************************************************
%   Output:
% p_l_vec  = L x 1 (L = 2^m = no. of possible sets of classification 
%            results)vector containing probabilities {p_l}
%*************************************************
m = size(v_j_cell,2); % No of classifiers 
p_l_vec = [gt*(1-v_j_cell{1})+(1-gt)*v_j_cell{1};gt*v_j_cell{1}+(1-gt)*(1-v_j_cell{1})];
for l1 = 2:m
    v = v_j_cell{l1};
    v1 = [gt*(1-v)+(1-gt)*v;gt*v+(1-gt)*(1-v)];
    v = p_l_vec;
    p_l_vec = [];
    for l2 = 1:length(v)
        p_l_vec = [p_l_vec; v(l2)*v1(:,l2)];
    end
end
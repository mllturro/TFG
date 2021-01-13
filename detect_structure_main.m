function [k_hat,one_c_hat] = detect_structure_main(R_mat)
%*************************************************
% Written by M. Llobet, Jan 2021 as an adaptation of the original algorithm
% presented by Jaffe et al, 2016.
% (See Algorithm 1 in the thesis's main document.)
%*************************************************
%    Input:
% R_mat = Estimated inter-classifier covariance matrix (Eq. 1.7)
%*************************************************
%   Output:
% k_hat = estimated no. of correlation groups
% one_c_hat = estimated indicator function (Eq. 1.9)
%*************************************************
m = size(R_mat,1); % No of. classifiers

%% Score matrix generation (Eq. 1.12):
S_mat = scorematrix(R_mat);

%% Indicator function estimation & matrix completion of V_on / V_off:
% Empty cell arrays and matrices are created for subsequent result history
% record.
res_vec = 1e2.*ones(m,1); one_c_cell = cell(1,m); c_hat_mat = zeros(m); 
V_on_cell = cell(1,m); V_off_cell = cell(1,m);
v_on_mat = zeros(m); v_off_mat = zeros(m);
X_on_cell = cell(1,m); X_off_cell = cell(1,m); 

% The following for-loop iterates for all possible no. of correlated sub-groups of classifiers (k):
for k = 2:m
    % Spectral clustering is performed on score matrix S:
    c_hat = spectralcluster(S_mat,k,'Distance','precomputed'); 
    
    % Known entries of matrices V_on, V_off are recorded.
    % Structure: [i index, j index, entry value; ... ]
    V_on = []; V_off = []; 
    
    one_c_hat = eye(m); 
    for ii = 1:m
        for jj = ii+1:m % symmetric matrix
            if c_hat(ii) == c_hat(jj) % if classifier i belongs to the same group of classifier j...
                one_c_hat(ii,jj) = 1; one_c_hat(jj,ii) = 1; % ...assign 1 to one_c_hat(i,j)... 
                V_on = [V_on; ii jj R_mat(ii,jj)]; % save location & value in V_on.
            elseif c_hat(ii)~= c_hat(jj) % repeat for the opposite case, V_off,..
                V_off = [V_off; ii jj R_mat(ii,jj)]; 
            end
        end
    end
    clear ii jj
    
    % Reconstruct matrices V_on, V_off
    if size(V_on,1) > 0 && size(V_off,1) > 0 % if both matrices are uncompleted...
       [v_on,X_on] = tracemin(m,V_on); % ... perform trace minimization on both
       [v_off,X_off] = tracemin(m,V_off);
    elseif size(V_off,1) == 0 % if all V_on (non-diagonal) entries are known...
        X_on = estimate_rank_1_matrix(R_mat); X_off = []; % ... estimate its diagonal values
        v_on = diag(X_on); 
        v_off = zeros(m,1); 
    elseif size(V_on,1) == 0 % same as previous 'if' case, but with V_off
        X_off = estimate_rank_1_matrix(R_mat); X_on = [];
        v_off = diag(X_off); 
        v_on = zeros(m,1); 
    end
  
    % Residual computation (Eq. 1.11)
    res_ij = 0;
    for ii = 1:m
        for jj = [(1:1:ii-1) (ii+1:1:m)] % i index skipped
            op_ij = (one_c_hat(ii,jj)*(v_on(ii)*v_on(jj)-R_mat(ii,jj))^2)+((1-one_c_hat(ii,jj))*(v_off(ii)*v_off(jj)-R_mat(ii,jj))^2);
            res_ij = res_ij + op_ij;
        end
    end
    
    % Record obtained values
    res_vec(k) = res_ij;
    one_c_cell{k} = one_c_hat; c_hat_mat(:,k) = c_hat;
    V_on_cell{k} = V_on; V_off_cell{k} = V_off;
    v_on_mat(:,k) = v_on; v_off_mat(:,k) = v_off;
    X_on_cell{k} = X_on; X_off_cell{k} = X_off;
end
clear k ii jj op_ij res_ij
[~, k_hat] = min(res_vec); % keep the k minimizing the residual

% Keep the indicator function & assignment function accordingly
one_c_hat = one_c_cell{k_hat}; 
c_hat = c_hat_mat(:,k_hat);
end
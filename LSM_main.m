function y_est = LSM_main(Z_mat,K,c_one,delta)
%*************************************************
% Originally written by Ariel Jaffe and Boaz Nadler, 2015. Update to the
% correlated case & other adaptations by M. Cabrera and M. Llobet, Jan
% 2021.
% (See Algorithm 3 in the thesis's main document.)
%*************************************************
%    Input:
% Z_mat = n x m matrix containing n instances labeled by m classifiers 
% K     = no. of correlation groups
% c_one = indicator function (Eq. 1.9)
% delta = bounds away the class imbalance, psi and eta estimations b_hat in
%         [-1+delta,1-delta], psi,eta in [delta,1-delta]
%*************************************************
%  Output:
% y_est = n x 1 vector containing ground truth label estimates
%*************************************************

Z_mat = 2.*(Z_mat-0.5.*ones(size(Z_mat)))'; % conversion from (0,1) to (-1,1) and dimension switch
n = size(Z_mat,2); % no. of instances

c = assignfun(c_one)'; % obtain the assignment function from the indicator function

% Memory setup for the classifiers' sensitivities and specificities conditioned
% to the latent variables {alpha_k} 
psi_alpha = []; eta_alpha = [];

ZZ = zeros(K,n); % Memory setup for the K latent variables' M classification estimates

for kk = 1:K
    Z_kk = Z_mat(c == kk,:); 
    % get only decision results of classifiers belonging to correlation
    % group 'kk'
    
    m_kk = size(Z_kk,1); % no. of classifiers in correlation group 'kk'
    
    if m_kk > 1
    
        % NOTE: class imbalance computation based on restricted likelihood;
        % performs better than the tensor-based method (Jaffe et al, 2015)
        b_kk = estimate_class_imbalance_restricted_likelihood(Z_kk,delta);

        [V_kk,psi_alpha_kk,eta_alpha_kk] = estimate_ensemble_parameters(Z_kk,b_kk);
        psi_alpha = [psi_alpha psi_alpha_kk'];
        eta_alpha = [eta_alpha eta_alpha_kk'];

        ZZ(kk,:) = sign(V_kk'*Z_kk);
        
    elseif m_kk == 1
        
        ZZ(kk,:) = Z_kk;    
        
    end
    
end
clear kk m_kk

if K > 1
    b = estimate_class_imbalance_restricted_likelihood(ZZ,delta);
    
    [V,alph_1,alph_0] = estimate_ensemble_parameters(ZZ,b);
    alph_1 = alph_1'; alph_0 = alph_0';
    
    y_est = sign(V'*ZZ);
    
elseif K == 1
    y_est = ZZ;
end

y_est = 0.5.*(y_est+ones(size(y_est)))'; % (-1,1) to (0,1) conversion and dimension switch

end



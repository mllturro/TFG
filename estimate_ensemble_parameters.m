function [v_vec,psi_hat,eta_hat] = estimate_ensemble_parameters(Z_mat,b)
%*************************************************
% Originally written by Ariel Jaffe and Boaz Nadler, 2015. Adaptations by
% M. Llobet, Jan 2021.
%*************************************************
%    Input:
% Z_mat = n x m matrix containing n instances labeled by m classifiers 
% K     = no. of correlation groups
%*************************************************
%    Output: 
% v_vec   = first eigenvector of the covariance matrix Z
% psi_hat = estimated sensitivities of m classifiers
% eta_hat = estimated specificities of m classifiers
%*************************************************
    m = size(Z_mat,1);
    
    %estimate mean
    mu = mean(Z_mat,2);
    
    %estimate covariance matrix 
    R = cov(Z_mat');
    
    if m > 2
        % estimate the diagonal values of a single rank matrix
        R = estimate_rank_1_matrix(R);
    elseif m == 2
        M = [1 2 R(1,2); 2 1 R(2,1)];
        [~,R] = tracemin(m,M);
    end  
    
    %get first eigenvector
    [v_vec, ~] = eigs(R,1);
    v_vec = v_vec*sign(sum(sign(v_vec)));
    
    %get constant C for first eigenvector min(C*V*V'-R)
    R_v = v_vec*v_vec';
    Y = R( logical(tril(ones(m))-eye(m)) );
    X = R_v( logical(tril(ones(m))-eye(m)) );
    [~,C] = evalc('lsqr(X,Y)');
    v_vec = v_vec*sqrt(C);
    
    %estimate psi and eta
    psi_hat = 0.5*(1+mu+v_vec*sqrt( (1-b)/(1+b)));
    eta_hat = 0.5*(1-mu+v_vec*sqrt( (1+b)/(1-b)));   
end
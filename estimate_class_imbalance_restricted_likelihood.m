function b_hat = estimate_class_imbalance_restricted_likelihood(Z_mat,delta)
%*************************************************
% Originally written by Ariel Jaffe and Boaz Nadler, 2015. Adaptations by
% M. Llobet, Jan 2021.
%*************************************************
%    Input:
% Z_mat = n x m matrix containing n instances labeled by m classifiers 
% delta = bounds away the class imbalance, psi and eta estimations b_hat in
%         [-1+delta,1-delta], psi,eta in [delta,1-delta]
%*************************************************
%    Output:
% b_hat = Estimated class imbalance Pr(Y=1)-Pr(Y=-1)
%*************************************************
        
    %get number of classifiers
    m = size(Z_mat,1);
    
    %estimate first moment
    mu = mean(Z_mat,2);
        
    %estimate second moment
    R = cov(Z_mat');
    
    if m > 2
        % estimate the diagonal values of a single rank matrix
        R = estimate_rank_1_matrix(R);
    elseif m == 2
        M = [1 2 R(1,2); 2 1 R(2,1)];
        [~,R] = tracemin(m,M);
    end 
    
    %get first eigenvector
    [V, ~] = eigs(R,1);
    V = V*sign(sum(sign(V)));
    
    %get constant C for first eigenvector min(C*V*V'-R)
    R_v = V*V';
    Y = R( logical(tril(ones(m))-eye(m)) );
    X = R_v( logical(tril(ones(m))-eye(m)) );
    [~,C] = evalc('lsqr(X,Y)');
    V = V*sqrt(C);
    
    %Scan over b in [-1+delta, 1-delta]
    b_min = -1+delta;
    b_max =  1-delta;
    res   = 0.01;
    b_tilde_vec = b_min:res:b_max;
    
    pi_tilde = zeros(m,length(b_tilde_vec));
    delta_tilde = zeros(m,length(b_tilde_vec));
    psi_tilde = zeros(m,length(b_tilde_vec));
    eta_tilde = zeros(m,length(b_tilde_vec));
    restricted_ll = zeros(1,length(b_tilde_vec));
        
    for k = 1:length(b_tilde_vec)
        
        %get values of pi,delta,psi and eta which correspond to b_tilde_vec(k)
        pi_tilde(:,k) = ((V/(1-b_tilde_vec(k)^2))+1)/2;
        delta_tilde(:,k) = (mu - (2*pi_tilde(:,k)-1)*b_tilde_vec(k))/2;        
        psi_tilde(:,k) = min(pi_tilde(:,k)+delta_tilde(:,k),1-delta);
        eta_tilde(:,k) = min(pi_tilde(:,k)-delta_tilde(:,k),1-delta);        
        psi_tilde(:,k) = max( psi_tilde(:,k),delta);
        eta_tilde(:,k) = max(eta_tilde(:,k),delta);
        
        %estimate log likelihood function: log(p*Pr(f|y=1)+(1-p)Pr(f|y=-1)
        ll_pos = zeros(size(Z_mat));
        ll_neg = zeros(size(Z_mat));
        p_k = (1+b_tilde_vec(k))/2;
        
        for i = 1:m
            
            %find positive and negative index
            pos_idx = find(Z_mat(i,:)==1);
            neg_idx = find(Z_mat(i,:)==-1);
            
            % find for each element Pr(f|y=1)
            ll_pos(i,pos_idx) = psi_tilde(i,k);
            ll_pos(i,neg_idx) = 1-psi_tilde(i,k);
            ll_neg(i,pos_idx) = 1-eta_tilde(i,k);
            ll_neg(i,neg_idx) = eta_tilde(i,k);
        end
        
        %estimate log likelihood function: log(p*Pr(f|y=1)+(1-p)Pr(f|y=-1)
        ll_vec = log(p_k*prod(ll_pos)+(1-p_k)*prod(ll_neg));        
        restricted_ll(k) = mean(ll_vec);        
    end
    
    
    %get maximum of restricted likelihood function
    [~,max_idx] = max(restricted_ll);
    b_hat = b_tilde_vec(max_idx);
        
end
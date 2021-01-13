function y_est = CEM_main(Z_mat,it,K,one_c)
%*************************************************
% Written by Cabrera et al, 2019.
% (see Algorithm 2 in the thesis's main document.)
%*************************************************
%    Input:
% Z_mat = n x m matrix containing n instances labeled by m classifiers 
% it    = no. of iterations in the EM algorithm
% K     = no. of correlation groups
% one_c = indicator function (Eq. 1.9)
%*************************************************
%    Output:
% y_est = n x 1 vector containing ground truth estimates
%
%*************************************************

n = size(Z_mat,1); % No. of decisions
m = size(Z_mat,2); % No. of classifiers
y_est = sum(Z_mat,2)/m; % Mean of the estimates provided by the m classifiers

c = assignfun(one_c); % obtain assignent function c from indicator function one_c

Z_kk = zeros(n,K); % ensemble estimate for each correlation group
for kk = 1:K
    Z1 = Z_mat(:,c == kk);
    Z1 = bi2de(flip(Z1,2));
    Z_kk(:,kk) = Z1+1;
end
Kmin = 10; % minimum no. of iterations
Epsilon = 1e-5;
Pr = cell(1,K);
Q_funtion = zeros(1,it+1);
Qmax = -inf;


%% EM Algorithm
k_iterations = 1;
% Start iterating via SoftMV
y_est(y_est==1) = 1-0.1*rand(size(find(y_est==1)));
y_est(y_est==0) = 0.1*rand(size(find(y_est==0)));

while k_iterations<it+1
    % M-Step:
    pi1=sum(y_est)/n;
    for kk = 1:K
        Rw_kk = 2^length(c(c == kk));
        % no. of possible combinations of sets of classification results
        
        Bl_0 = zeros(Rw_kk,1); Bl_1 = zeros(Rw_kk,1);
        
        for i_l = 1:Rw_kk
            Vaux=Z_kk(:,kk)==i_l;
            Bl_1(i_l)=sum(y_est(Vaux==1));
            Bl_0(i_l)=sum(Vaux)-sum(y_est(Vaux==1));
        end
        Pr(kk)={[Bl_0/sum(Bl_0) Bl_1/sum(Bl_1)]};
    end
    
    % E-Step:
    for i_m = 1:n
        pdf_1 = 1;
        pdf_0 = 1;
        for kk = 1:K
            pdf_1 = pdf_1*Pr{kk}(Z_kk(i_m,kk),2);
            pdf_0 = pdf_0*Pr{kk}(Z_kk(i_m,kk),1);
        end
        y_est(i_m) = (pdf_1*pi1)/(pdf_1*pi1+pdf_0*(1-pi1));
    end
    
    % Qfunction:
    Qf=0;
    for i_m = 1:n
        if (y_est(i_m) > 0)
            if pi1>0
                Qf = Qf+y_est(i_m)*log(pi1);
            end
            for kk = 1:K
                if Pr{kk}(Z_kk(i_m,kk),2)>0
                    Qf = Qf+y_est(i_m)*log(Pr{kk}(Z_kk(i_m,kk),2));
                end
            end
            if or(isnan(Qf),abs(imag(Qf))>0)
                keyboard
            end
        end
        
        if (y_est(i_m) <1)
            if 1-pi1>0
                Qf = Qf+log(1-pi1)*(1-y_est(i_m));
            end
            for kk=1:K
                if Pr{kk}(Z_kk(i_m,kk),1)>0
                    Qf = Qf+(1-y_est(i_m))*log(Pr{kk}(Z_kk(i_m,kk),1));
                end
            end
            if or(isnan(Qf),abs(imag(Qf))>0)
                keyboard
            end
        end
    end
    Q_funtion(k_iterations) = Qf;
    if Qf>Qmax
        %k_iterations
        Qmax = Qf;
        V_out = y_est;
    end
    
    % Check degree of convergence
    if k_iterations > Kmin
        if ( Q_funtion(k_iterations)- Q_funtion(k_iterations-1)) ...
                < Epsilon*abs( Q_funtion(k_iterations))
            k_iterations=it;
        end
    end
    k_iterations=k_iterations+1;
end

y_est=round(V_out);
end

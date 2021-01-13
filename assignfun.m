function c = assignfun(one_c)
%*************************************************
% Written by M. Llobet, Jan 2021.
%*************************************************
%    Input:
% one_c_hat = indicator function (Eq. 1.9)
%*************************************************
%   Output:
% c         = m x 1 vector-form assignment function c: [m] --> [k]
%*************************************************
m = size(one_c,1); % no. of classifiers
c = zeros(1,m);

kk_count = 1;

find_vec = [];
for ii = 1:m
    if isempty(find(find_vec == ii)) % if classifier ii hasn't been recorded yet
        one_c_kk = find(one_c(ii,:)); find_vec = [find_vec one_c_kk];
        c(one_c_kk) = kk_count;
        kk_count = kk_count + 1;
    end   
end
clear ii c_one_kk kk_count
end
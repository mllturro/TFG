function [v_vec,V_mat] = tracemin(m,M)
%*******************************************************
% Written by M. Llobet, Jan 2021.
% (See Algorithm 1 in the thesis's main document.)
%
% IMPORTANT NOTE: the SDPT3 solver for SDP programming is used. As this is 
% an external resource, make sure to add the corresponding path to the 
% solver's main folder before calling this function. For further
% information, see the solver's documentation.
% 
% Example of a path addition:
addpath(genpath('C:\Users\mlltu\OneDrive\Documentos\Enginyeria Física\TFG\Algorithm 1\SDPT3-4.0')); 
%-------------------------------------------------------
%% tracemin: trace minimization problem.
%
% (primal problem) min  Tr V
%                  s.t.  V_ij = M_ij; V semidefinite positive;
%                              
% The above primal problem may be expressed as a SDP optimization problem. 
% This is a particular case of nuclear norm minimization (see Candès 2009 & Recht 2010)
%-------------------------------------------------------
% [v_vec,V_mat] = tracemin(m,M)
%*******************************************************
%   Input:
% m     = no. of classifiers
% M     = uncompleted matrix data [i index, j index, entry value; ...]
%*************************************************
%   Output:
% v_vec = m x 1 vector generating rank-one (completed) V matrix, V = v_vec*v_vec'
% V = reconstructed matrix M
%%-------------------------------------------------------
%% SDPT3 solver
% The SDPT3 solver (version 4.0) is used - 'sqlp' is the solver's main function.
% [obj,X,y,Z,info,runhist] = sqlp(blk,At,C,b,OPTIONS,X0,y0,Z0)
%*******************************************************
%   Input:
% blk = X's block structure (here, blk = {'s', m}, with 's' standing for 'simple' block structure)
% At  = problem's subjections on X entries (here, X_ij = M_ij) - to be expressed via constraint matrices 
% C   = primal problem expression (here, C = m x m identity matrix)
% b   = no_entries x 1 vector containing external set of values used to obey the problem's subjections (here, b_i = M_ij)
% (rest of input values are not used here.)
%*******************************************************
%   Output:
% X = resulting minimized matrix
% (rest of output values: see solver's documentation)
%% 
%%*****************************************************************
no_entries = size(M,1); % no. of known matrix entries

% Formatting input data for SDPT3 solver (version 4.0):
blk{1,1} = 's';  blk{1,2} = m; % type of block structure
C{1} = eye(m); % trace minimization: min < C, X >

% b is a no_entries x 1 vector containing the values of known entries
b = 2.*M(:,3); % list of values of known matrix elements (m x 1 vector)
% Generation of constraint matrices:
AA = cell(1,no_entries); % there are as many constraint matrices as unknown matrix entries
for kk = 1:no_entries
    AA_kk = zeros(m); % the all-zeros matrix...
    AA_kk(M(kk,1),M(kk,2)) = 1; AA_kk(M(kk,2),M(kk,1)) = 1; %... is replaced with '1's in indices of known entries
    AA_kk = sparse(AA_kk); AA{kk} = AA_kk; % each constraint matrix is stored as a sparse matrix
end 
At = svec(blk,AA,ones(size(blk,1),1)); 
    %{
svec converts cell array of sparse matrices into an acceptable input for
'sqlp'; tbh, I do not really understand how it works, it's weird; It is
based on some similar examples of optimization problems provuded by the
solver's authors.
    %}
[~,obj,V_mat,y,Z,info,runhist] = evalc('sqlp(blk,At,C,b,[])');
V_mat = full(V_mat{1}); Z = full(Z{1}); % X, Z computed as cell arrays by solver

[v_vec, ~] = eigs(V_mat,1);
v_vec = v_vec*sign(sum(sign(v_vec)));

%get constant C for first eigenvector min(C*V*V'-R)
R_v = v_vec*v_vec';
Y_v = V_mat( logical(tril(ones(m))-eye(m)) );
X_v = R_v( logical(tril(ones(m))-eye(m)) );
[~,ctn] = evalc('lsqr(X_v,Y_v)');
v_vec = v_vec*sqrt(ctn);
end

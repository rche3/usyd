function [results]=Estimate_benchmark_NW(Y,X_1,X_2,bw)

% Last modified: 05-15-2012
%
% Estimates the predictive regression model,
%
% r(i,t+1) = b(i,0) + b(i,1)*x(i,1,t) + b(i,2)*x(i,2,t) + e(i,t+1),
%
% where test statistics are computed using a Newey West OLS procedure.
%
% Input
%
% Y   = T-by-N matrix of excess return observations
% X_1 = T-by-N matrix of observations for first predictor
% X_2 = T-by-N matrix of observations for second predictor
% bw  = bandwidth parameter for newey-west
%
% Output
%
% results = (N)-by-4 matrix;
%           First N rows: b(i,1) estimates, t-stats,
%                         b(i,2) estimates, t-stats,
%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

[T,N]=size(Y);
K=3; % number of RHS variables (including intercept)
results=zeros(N,4);

%%%%%%%%%%%%%%%%%%%%%
% Setting up matrices
%%%%%%%%%%%%%%%%%%%%%

Y_stack=zeros(N*(T-1),1);
X_stack=zeros(N*(T-1),N*K);
for t=1:(T-1);
    for i=1:N;
        Y_stack((t-1)*N+i)=Y(t+1,i);
        X_stack((t-1)*N+i,(i-1)*K+1:i*K)=[X_1(t,i) X_2(t,i) 1];
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLS estimation with GMM standard errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nwest_res = nwest(Y_stack, X_stack, bw);
beta_ols=nwest_res.beta;
beta_tstat = nwest_res.tstat;

for i=1:N
    results(i,1) = beta_ols(1);
    results(i,2) = beta_tstat(1);
    results(i,3) = beta_ols(2);
    results(i,4) = beta_tstat(2);
end

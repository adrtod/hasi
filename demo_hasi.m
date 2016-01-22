%% demo for using HASI

% Install matlab-mex files:
% Type in matlab command line (root directory of HASI)
% >> install_mex
% Creates mex files  PROPACK_utils/dbdqr.c    PROPACK_utils/reorth.c   Matlab_files/project_obs_UV.c

clear all
close all

%% Add folders Matlab_files and PROPACK_utils in path
addpath Matlab_files PROPACK_utils

%% Generate data
%%**************************************
fprintf('Generating data...\n');
rng shuffle

m = 100; % nb of rows
n = 100; % nb of col
q = 5; % true rank
density = 0.2; % density of observations
SNR = 10; % signal/noise ratio
sigma = 1; % noise std. deviation

%%% low-rank factorization
sigma_sig = sqrt(SNR*sigma/sqrt(q));
A = sigma_sig*randn(m,q);
B = sigma_sig*randn(n,q);
Z = A*B'; % population outer-product

SNR = std(Z(:))/sigma; % true SNR

%%% sparse-observed matrix Xobs
temp = sprand(m,n,density);
[i_row,j_col] = find(temp);

temp = dot(A(i_row,:), B(j_col,:),  2);
temp = temp + sigma*randn(length(temp),1); % add noise

Xobs = sparse(i_row, j_col, temp, m, n);
mask = sparse(i_row, j_col, true, m, n);

clear temp

%% declare fields of structure OPTS and PARAM
OPTS.TOLERANCE = 1e-3;
% OPTS.SMALL_SCALE = 0;
maxiter = 100;
PARAM.sigma = sigma;

%% Create a path of solutions
%%**************************************
fprintf('Creating a path of solutions...\n');
lambda_max = spectral_norm(Xobs); % approximates the lambda value for which solution is zero
n_lambda = 20;
% sequence of decreasing regularization parameter values
lambda_seq = lambda_max.*logspace(-.1, -3, n_lambda);
% vector of beta values
beta_seq = [100 10 1];
n_beta = numel(beta_seq);

% preallocation
error = NaN(n_lambda, n_beta);
nnorm = NaN(n_lambda, n_beta);
rank = NaN(n_lambda, n_beta);

INIT = [];
t0 = tic;
for i = 1:n_lambda
    t1 = tic;
    fprintf('step %d/%d: lambda=%g', i, n_lambda, lambda_seq(i));
    
    %%% Initialize with Soft Impute
    OPTS.MAXITER = round(maxiter/2); % allow half iterations at most for init
    [U,D,V, out] = soft_impute(Xobs, lambda_seq(i), OPTS, INIT); % warm-start specified via INIT
    
    INIT = struct('U',U,'D',D,'V',V); % specify warm-starts for HASI and smaller lambda value
    
    OPTS.MAXITER = maxiter-out.iter;
    PARAM.lambda = lambda_seq(i)/sigma^2;
    for j = 1:n_beta
        %%% Run HASI
        PARAM.beta = beta_seq(j);
        [U,D,V] = ha_soft_impute(Xobs, PARAM, OPTS, INIT); % warm-start specified via INIT
        Zest = U*D*V';
        
        %%% Process the output, for example compute the relative precition error
        error(i,j) = norm((~mask).*(Zest - Z),'fro')^2/norm((~mask).*(Z),'fro')^2;
        nnorm(i,j) = sum(diag(D));
        rank(i,j) = sum(diag(D)>0);
    end
    
    fprintf(', time=%g sec.\n', toc(t1));
end
toc(t0)

%% Plot results
%%**************************************
set(0,'defaultlinelinewidth', 2)
set(0,'defaultaxesfontsize', 16)

leg = cell(n_beta, 1);
for j=1:n_beta
    leg{j} = sprintf('HASI beta=%g', beta_seq(j));
end
tit = sprintf('%dx%d matrix, %g%% missing entries\n true rank=%d, SNR=%.2g, sigma=%g', ...
    m, n, round((1-density)*100), q, SNR, sigma);

figure
plot(nnorm, error)
xlabel('Nuclear norm')
ylabel('Error')
legend(leg)
legend boxoff
title(tit)
box off

figure 
plot(rank, error)
xlabel('Rank')
ylabel('Error')
legend(leg)
legend boxoff
title(tit)
box off
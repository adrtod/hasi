function [U,D,V,hist] = soft_impute(Xobs, lambda, OPTS, INIT)
% This function performs low rank matrix completion with noisy entries.
%
% (A)  min_Z  1/2*|| Xobs - P(Z) ||_F^2 + lambda*||Z||_*
%
% P(Z)    : size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% Xobs    : observed data matrix with exactly the same sparsity pattern as P(Z).
% lambda  : regularization parameter
% ||Z||_* : nuclear norm of matrix Z (sum of its sing values)
%
% soft_impute(...) solves the problem (A) allowing for warm-start INIT (see below).
% 
% This function is a wrapper to function ha_soft_impute with appropriate arguments.
% See "help ha_soft_impute" for more details.
%
% INPUTS:
%----------
% 1) Xobs (REQUIRED): sparse matrix (nrow x ncol). 0's correspond to the missing values
% 2) lambda (REQUIRED): regularization parameter on the nuclear norm of Z
% 3) Structure OPTS (optional) with fields:
%     BINARY (optional): boolean. Indicates if observed data are binary. In
%     this case we consider the data matrix Xobs is observed from a probit
%     model. default=0
%     SMALL_SCALE (optional): boolean. 1 means small-scale: direct
%          factorization based SVD will be used instead of PROPACK Lanczos
%          bidiagonalization with partial reorthogonalization.
%          default=(min(nrow,ncol) < 2000)
%     CRITERION (optional): string defining the criterion used for
%          convergence. Relative error between two iterates of this
%          criterion falling under the tolerance will stop the algorithm.
%          Possible values are:
%          * 'obj' (default): the objective function
%          * 'Z': the estimated matrix. Only if SMALL_SCALE==1.
%     TOLERANCE (optional): tolerance on the convergence criterion (lack
%          of progress of succ iterates), default=1e-4
%     MAXITER (optional): max nb of iterations requiered for convergence.
%          default=100
%     MAX_RANK (optional): max nb of sing-vectors that can be computed.
%          Large-scale only (SMALL_SCALE==0). default=min(nrow,ncol).
%          * if min(nrow,ncol) > 2000; MAX_RANK=min(500,MAX_RANK)
%          * if nnz(Xobs) > 1e7; MAX_RANK=min(50,MAX_RANK)
%     INCREMENT (optional): increase the number of sing-vectors to be
%          computed as a part of PROPACK by this amount. Large-scale only
%          (SMALL_SCALE==0). default=10.
%     WARNING (optional): boolean. toggle warning and other messages. default=1
% 4) Structure INIT (optional) with fields:
%   U: unitary matrix (nrow x k). left matrix of singular vectors
%   D: diagonal matrix (k x k). matrix of singular values
%   V: unitary matrix (ncol x k). right matrix of singular vectors
%     If provided, ALL of (U,D,V) are required
%     Default: All U,D, V are set to zero.
%
% OUTPUTS:
%----------
%   U: left singular matrix,
%   D: singular values (vector)
%   V: right singular matrix.
%   hist: structure with fields
%      * obj  : sequence of objective values across iterations
%      * nnorm: sequence of nuclear norms across iterations
%      * rank : sequence of ranks across iterations
%      * time : total elapsed time
%
% For any questions/suggestions/comments/bugs please report to
% <adrien.todeschini@inria.fr>
%--------------------------------------------------------------------------
%   
% Matlab code written by Adrien Todeschini <adrien.todeschini@inria.fr>
% Reference: "Probabilistic Low-Rank Matrix Completion with Adaptive 
% Spectral Regularization Algorithms" 
% by Adrien Todeschini, Francois Caron, Marie Chavent (NIPS' 2013)
%
% Adapted from Matlab code written by Rahul Mazumder <rahulm@stanford.edu>
% Reference: "Spectral Regularization Algorithms for Learning Large 
% Incomplete Matrices"
% by Rahul Mazumder, Trevor Hastie, Rob Tibshirani (JMLR vol 11, 2010)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check arguments
% set default output
U=[];D=[];V=[];hist=[];

% check number of required arguments
if nargin<2
    disp('Error: requires at least 2 inputs');
    return
end

% check lambda
if ~isnumeric(lambda) || ~isscalar(lambda) || (lambda<0)
    disp('Error: lambda must be a scalar >= 0.');
    return
end

PARAM = [];
PARAM.lambda = lambda;
PARAM.sigma = 1;

% check type of OPTS
if  (nargin>=3) && ~isempty(OPTS)
    if ~isstruct(OPTS);
        disp('Error: OPTS must be a structure');
        return
    else
        OPTS.VARIANT = 'soft';
        OPTS.ESTIM_SIGMA = 0;
    end
else
    OPTS=[];
    OPTS.VARIANT = 'soft';
end

if nargin<4
    INIT = [];
end

% check number of output arguments
FLAG_hist = nargout>=4;

if FLAG_hist
    [U,D,V,hist] = ha_soft_impute(Xobs, PARAM, OPTS, INIT);
else
    [U,D,V] = ha_soft_impute(Xobs, PARAM, OPTS, INIT);
end
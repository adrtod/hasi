function [U,D,V,hist,Z] = hard_impute(Xobs, k, OPTS, INIT)
% This function performs low rank matrix completion with noisy entries.
%
% (A)  min_Z  1/(2*sigma^2)*|| Xobs - P(Z) ||_F^2 + rank(Z)
%
% P(Z)   : size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% Xobs   : observed data matrix with exactly the same sparsity pattern as P(Z).
% sigma  : standard deviation of the noise : Xobs_ij ~ N(Z_ij, sigma^2)
% rank(Z): rank of matrix Z. nb of nonzero singular values.
%
% hard_impute(..) solves the problem (A) allowing for warm-start
% INIT (see below).
%
% INPUTS:
%----------
% 1) Xobs (REQUIRED) : sparse matrix (nrow x ncol). 0's correspond to the missing values
% 2) k: nb of sing-vectors to be computed.
%          * if min(nrow,ncol) > 2000; k=min(500,MAX_RANK)
%          * if nnz(Xobs) > 1e7; k=min(50,MAX_RANK)
% 3) Structure OPTS (optional) with fields:
%     BINARY (optional): boolean. Indicates if observed data are binary. In
%     this case we consider the data matrix Xobs is observed from a probit
%     model. default=0
%     SMALL_SCALE (optional): boolean. 1 means small-scale: direct
%          factorization based SVD will be used instead of PROPACK Lanczos
%          bidiagonalization with partial reorthogonalization.
%          default=(min(nrow,ncol) < 2000)
%     TOLERANCE (optional): tolerance on the convergence criterion (lack
%          of progress of succ iterates), default=1e-4
%     MAXITER (optional): max nb of iterations requiered for convergence.
%          default=100
%     WARNING (optional): boolean. toggle warning messages. default=1
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
%      * nnorm: sequence of nuclear norms across iterations
%      * time : total elapsed time
%
% For any questions/suggestions/comments/bugs please report to
% <adrien.todeschini@inria.fr>
%--------------------------------------------------------------------------
%
% Matlab code written by Adrien Todeschini <adrien.todeschini@inria.fr>
% Reference: "Probabilistic Low-Rank Matrix Completion with Adaptive Spectral Regularization Algorithms" 
% by Adrien Todeschini, Francois Caron, Marie Chavent (NIPS' 2013)
%
% Adapted from Matlab code written by Rahul Mazumder <rahulm@stanford.edu>
% Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
% by Rahul Mazumder, Trevor Hastie, Rob Tibshirani (JMLR vol 11, 2010)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global INCREMENT MAX_RANK

%% Check arguments
% set default output
U=[];D=[];V=[];hist=[];Z=[];

% check number of required arguments
if nargin<2
    disp('Error: requires at least 2 inputs');
    return
end

% check number of output arguments
FLAG_hist = nargout>=4;

% check required arguments are not empty
if isempty(Xobs) || isempty(k)
    disp('Error: requires observed data AND rank');
    return
end

% check type of OPTS
if  (nargin>=3) && ~isempty(OPTS)
    if ~isstruct(OPTS);
        disp('Error: OPTS must be a structure');
        return
    end
else
    OPTS=[];
end
% check type of INIT
if (nargin>=4) && ~isempty(INIT)
    if ~isstruct(INIT)
        disp('Error: INIT must be a structure');
        return
    end
else
    INIT=[];
end
% check type of Xobs
if ~issparse(Xobs)
    disp('Error: requires Input observed matrix to be a sparse matrix');
    return
end
% check dimensions of Xobs
if ~ismatrix(Xobs)
    disp('Error: Incorrect dimensions for Observed matrix');
    return
end
[nrow,ncol] = size(Xobs);
nmin = min(nrow,ncol);

% check rank
if ~isscalar(k) || (k<=0) || (k>nmin)
    disp('Error: k must be a scalar >0 and <min(size(Xobs))');
    return
end

%%%% BINARY case
% default sigma parameter
sigma = 1;
%%%%

%% Set options
% set default options
BINARY = 0;
SMALL_SCALE = (nmin < 2000);
CRITERION = 'Z';
TOLERANCE = 1e-4;
MAXITER = 100;
WARNING = 1;

% check if options are supplied
if isstruct(OPTS)
    % parse OPTS struct
    c = fieldnames(OPTS);
    for i=1:length(c)
        if any(strcmpi(c(i),'BINARY')); BINARY = double(OPTS.BINARY); end
        if any(strcmpi(c(i),'SMALL_SCALE')); SMALL_SCALE = double(OPTS.SMALL_SCALE); end
        if any(strcmpi(c(i),'TOLERANCE')); TOLERANCE = double(OPTS.TOLERANCE);  end
        if any(strcmpi(c(i),'MAXITER')); MAXITER = double(OPTS.MAXITER); end
        if any(strcmpi(c(i),'WARNING')); WARNING = double(OPTS.WARNING); end
    end
end
clear OPTS c

% check BINARY
if ~isscalar(BINARY) || (BINARY~=1 && BINARY~=0)
    disp('Error: BINARY must be a boolean.');
    return
end
if BINARY
    %%%%%%%%%%%%%% TODO : allow large scale for binary %%%%%%%%%%%%%%%%%%%
    SMALL_SCALE = 1;
end
% check SMALL_SCALE
if ~isscalar(SMALL_SCALE) || (SMALL_SCALE~=0 && SMALL_SCALE~=1)
    disp('Error: SMALL_SCALE must be a boolean.');
    return
end
% check CRITERION
if isempty(CRITERION)
    disp('Error: CRITERION must be a string');
    return
end
switch CRITERION
    case 'Z'
        rel_err_fun = @rel_err_Z;
    otherwise
        disp('Error: CRITERION value is not allowed');
        return
end
% check TOLERANCE
if ~isscalar(TOLERANCE) || (TOLERANCE<0)
    disp('Error: TOLERANCE must be a scalar >= 0');
    return
end
% check MAXITER
if ~isscalar(MAXITER) || (MAXITER <1)
    disp('Error: MAXITER must be a scalar >= 1.');
    return
end

if ~SMALL_SCALE
    % update k to safe-guard against large ranks
    % If you really need a large-rank solution, then change the default/comment this part out.
    if (nmin > 2000) && (k>500)
        if WARNING
            warning('Safe-guard against large ranks: k value was decreased to 500.');
        end
        k = 500;
    end
    if (nnz(Xobs)>1e7) && (k>50)
        if WARNING
            warning('Safe-guard against large ranks: k value was decreased to 50.');
        end
        k = 50;
    end
end

INCREMENT = 0;
MAX_RANK = k;

%% Set starting point factorization
% check if starting point factorization is supplied
if isstruct(INIT)
    if ~isempty(INIT.U) && ~isempty(INIT.D) && ~isempty(INIT.V)
        U =INIT.U; V=INIT.V; D=INIT.D;
        clear INIT
        dim_check1=size(U); dim_check2=size(V); dim_check3= size(D);
        % check dimensions
        if ( (dim_check1(1)~=nrow) || (dim_check2(1)~=ncol) || (dim_check2(2)~= dim_check2(2) ) )  || ( dim_check3(1) ~= dim_check3(2) )
            disp('Error: wrong dimensions in Input starting point.');
            return;
        end
        % check max rank
        if ~SMALL_SCALE && (dim_check2(2) > MAX_RANK)
            disp('Error: Input starting point has rank larger than MAX_RANK.');
            return;
        end
        clear dim_check1 dim_check2 dim_check3
    else
        disp('Error: Not proper starting point.');
        return
    end
else
    U = zeros(nrow,1);
    V = zeros(ncol,1);
    D = 0;
    
end
clear INIT

%% Intialization
hist = [];
hist.obj = zeros(MAXITER,1);
hist.nnorm = zeros(MAXITER,1);
hist.rank = zeros(MAXITER,1);

[i_row, j_col, data] = find(Xobs);
n_obs = length(data);

if BINARY %%%% BINARY case
    mask_neg = Xobs<0;
    mask_pos = Xobs>0;
end

t0 = tic;
U = U*D;

if (n_obs < 1e6)
    temp = sum(U(i_row,:).*V(j_col,:),2);
else
    temp = project_obs_UV(U, V, i_row, j_col, n_obs);
end

Zobs = sparse(i_row, j_col, temp, nrow, ncol);
Z = U*V';

crit = eval(CRITERION);
tol_curr = Inf;
i = 0;

%% Iterations
while (tol_curr>TOLERANCE) && (i<MAXITER)
    i=i+1;
    crit_old = crit;
    
    % M step
    % svd
    if SMALL_SCALE
        if BINARY %%%% BINARY case
            Xcomp = Z;
            temp = Xcomp(mask_pos);
            Xcomp(mask_pos) = temp + normpdf(temp,0,sigma)./(1+normcdf(-temp,0,sigma));
            temp = Xcomp(mask_neg);
            Xcomp(mask_neg) = temp - normpdf(temp,0,sigma)./normcdf(-temp,0,sigma);
        else
            Xcomp = Xobs+Z-Zobs;
        end
        
        [U,D,V] = svds(Xcomp, k);
    else
        if BINARY %%%% BINARY case
            error('large scale not available for binary data')
            %%%%%%%%%%%%%% TODO : allow large scale for binary %%%%%%%%%%%%%%%%%%%
        else
            Front_multi = @(x)A_multiply_fun_handle(x,Xobs,U,V,Zobs);
            Front_Transpose_multi = @(x)At_multiply_fun_handle(x,Xobs,U,V,Zobs);
            
            [U,D,V] = lansvd_thres(Front_multi, Front_Transpose_multi, nrow, ncol, k, 0);
        end
    end
    
    sv = diag(D);
    
    D = spdiags(sv, 0, k, k);
    U = U*D; % mix singular values inside U, temporarily
    
    if n_obs < 1e5
        temp = sum(U(i_row,:).*V(j_col,:),2);
    else
        temp = project_obs_UV(U, V, i_row, j_col, n_obs);
    end
    
    Zobs = sparse(i_row,j_col,temp, nrow, ncol);
    Z = U*V';
    
    crit = eval(CRITERION);
    tol_curr = rel_err_fun(crit, crit_old);
    
    % save history
    hist = save_history(hist,i);
end

hist = end_history(hist,i);

% divide U by singular values
U = U*spdiags(1./sv, 0, k, k);

% end_warn();


%% Function rel_err_Z
% relative error of the estimated matrix Z
    function e = rel_err_Z(Z_new, Z_old)
        e = norm(Z_new-Z_old,'fro')^2/norm(Z_old,'fro')^2;
    end

%% Function save_history
% store the history at iteration i
    function hist = save_history(hist, i)
        if FLAG_hist
            hist.nnorm(i) = sum(sv);
        end
    end
%% Function end_history
% resize the hsitory at the end of the iterations
    function hist = end_history(hist, i)
        if FLAG_hist
            hist.time = toc(t0);
            hist.iter = i;
            hist.nnorm = hist.nnorm(1:i);
        end
    end
%% Function end_warn
% warn if limits have been reached at the end of the iterations
    function end_warn()
        if i==MAXITER && WARNING>=2
            warning('MAXITER limit has been reached.');
        end
    end

end % main function
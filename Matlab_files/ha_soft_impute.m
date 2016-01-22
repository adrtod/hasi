function [U,D,V,hist] = ha_soft_impute(Xobs, PARAM, OPTS, INIT)
% This function performs low rank matrix completion with noisy entries.
%
% (A)  min_Z  1/(2*sigma^2)*|| Xobs - P(Z) ||_F^2 + sum_i(pen(d_i))
%
% P(Z)    : size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% Xobs    : observed data matrix with exactly the same sparsity pattern as P(Z).
% sigma   : standard deviation of the noise : Xobs_ij ~ N(Z_ij, sigma^2)
% d_i     : singular values of Z, i=1:min(nrow, ncol).
% pen(d_i): penalization term over d_i.
%
% ha_soft_impute(...) solves the problem (A) allowing for warm-start INIT (see below).
% We advocate initializing with the output of Soft-Impute algorithm with parameter lambda*sigma^2:
%   soft_impute(Xobs, lambda*sigma^2, OPTS)
%
% INPUTS:
%----------
% 1) Xobs (REQUIRED): sparse matrix (nrow x ncol). 0's correspond to the missing values
% 2) Structure PARAM (REQUIRED), defines the parameters of the model
% distributions. Depending on the value taken by OPTS.VARIANT, its fields must be:
%     If (OPTS.VARIANT=='soft'):
%          * lambda (required): regularization parameter
%     If (OPTS.VARIANT=='gamma'):
%          * lambda (required): scalar. location parameter of the prior p(gamma_i)
%          * beta   (required): scalar. rate parameter of the prior p(gamma_i)
%               beta->Inf => p(gamma_i) is concentrated and the problem is close to SoftImpute
%               beta->0 => p(gamma_i) is flat and the problem is close to HardImpute
%     If (OPTS.VARIANT=='igauss') :
%          * lambda (required): scalar>0. location parameter of the prior p(gamma_i)
%          * delta   (required): scalar>0. shape parameter of the prior p(gamma_i)
%     If (OPTS.VARIANT=='gig'):
%          * lambda (required): scalar>0. location parameter of the prior p(gamma_i)
%          * beta   (required): scalar>0. parameter of the prior p(gamma_i)
%          * delta  (required): scalar>0. parameter of the prior p(gamma_i)
%     Optional:
%          * sigma (optional): scalar value of standard deviation of the
%          noise, default=1.
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
%     VARIANT (optional): string defining the prior distribution
%     p(gamma_i) on the regularization parameters of each singular value.
%     Possible values are:
%          * 'soft': SoftImpute. All gamma_i are equal to a constant
%          * 'gamma' (default): Hierarchical with Gamma(gamma_i; lambda*beta, beta) prior
%          * 'igauss': Hierarchical with inverse Gaussian prior
%          * 'gig': Hierarchical with 3 parameters Generalized inverse Gaussian prior
%             %%%%%%%%% TODO %%%%%%%%%%
%          * 'jeffreys' (NOT IMPLEMENTED): Hierarchical with Jeffreys prior
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

% declare global variables
% used in function lansvd_thres
global MAX_RANK INCREMENT

%% Check arguments
% set default output
U=[];D=[];V=[];hist=[];

% check number of required arguments
if nargin<2
    disp('Error: requires at least 2 inputs');
    return
end

% check number of output arguments
FLAG_hist = nargout>=4;

% check required arguments are not empty
if isempty(Xobs) || isempty(PARAM)
    disp('Error: requires observed data AND prior parameters');
    return
end
% check type of PARAM
if ~isstruct(PARAM);
    disp('Error: PARAM must be a structure');
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
if length(size(Xobs))~= 2
    disp('Error: Incorrect dimensions for Observed matrix');
    return
end
[nrow,ncol] = size(Xobs);
nmin = min(nrow,ncol);

%% Parse PARAM struct
% set default parameters
sigma = 1;

%%% Experimental: hyperparameters of the prior over sigma in case
%%% ESTIM_SIGMA=1 (see below)
a_sigma = 1;
b_sigma = 1;

c = fieldnames(PARAM);
for i=1:length(c)
    if any(strcmpi(c(i),'sigma')); sigma = double(PARAM.sigma); end
    if any(strcmpi(c(i),'lambda')); lambda = double(PARAM.lambda);  end
    if any(strcmpi(c(i),'beta')); beta = double(PARAM.beta); end
    if any(strcmpi(c(i),'delta')); delta = double(PARAM.delta); end
end

% check sigma
if ~isscalar(sigma) || (sigma<=0)
    disp('Error: sigma must be a scalar > 0');
    return
end

%% Set options
% set default options
BINARY = 0;
SMALL_SCALE = (nmin < 2000);
CRITERION = 'obj';
TOLERANCE = 1e-4;
MAXITER = 100;
MAX_RANK = nmin;
INCREMENT = 10;
VARIANT = 'gamma';
WARNING = 1;

%%% Experimental: toggle estimation of the observation noise standard deviation
%%% Unsatisfactory results yet.
ESTIM_SIGMA = 0;

% check if options are supplied
if isstruct(OPTS)
    % parse OPTS struct
    c = fieldnames(OPTS);
    for i=1:length(c)
        if any(strcmpi(c(i),'BINARY')); BINARY = double(OPTS.BINARY); end
        if any(strcmpi(c(i),'SMALL_SCALE')); SMALL_SCALE = double(OPTS.SMALL_SCALE); end
        if any(strcmpi(c(i),'CRITERION')); CRITERION = char(OPTS.CRITERION); end
        if any(strcmpi(c(i),'TOLERANCE')); TOLERANCE = double(OPTS.TOLERANCE);  end
        if any(strcmpi(c(i),'MAXITER')); MAXITER = double(OPTS.MAXITER); end
        if any(strcmpi(c(i),'MAX_RANK'));  MAX_RANK = double(OPTS.MAX_RANK); end
        if any(strcmpi(c(i),'INCREMENT')); INCREMENT = double(OPTS.INCREMENT); end
        if any(strcmpi(c(i),'VARIANT')); VARIANT = char(OPTS.VARIANT); end
        if any(strcmpi(c(i),'WARNING')); WARNING = double(OPTS.WARNING); end
%         if any(strcmpi(c(i),'ESTIM_SIGMA')); ESTIM_SIGMA = logical(OPTS.ESTIM_SIGMA); end
    end
end
clear OPTS c

% check BINARY
if ~isscalar(BINARY) || (BINARY~=1 && BINARY~=0)
    disp('Error: BINARY must be a boolean.');
    return
end
if BINARY
    %%%%%%%%%%%%%% TODO: allow large scale for binary %%%%%%%%%%%%%%%%%%%
    SMALL_SCALE = 1;
end
% check SMALL_SCALE
if ~isscalar(SMALL_SCALE) || (SMALL_SCALE~=0 && SMALL_SCALE~=1)
    disp('Error: SMALL_SCALE must be a boolean.');
    return
end
if ~SMALL_SCALE
    % disable ESTIM_SIGMA
    ESTIM_SIGMA = 0;
end
% check CRITERION
if isempty(CRITERION)
    disp('Error: CRITERION must be a string');
    return
end
switch CRITERION
    case 'obj'
        rel_err_fun = @rel_err_obj;
    case 'Z'
        if ~SMALL_SCALE
            if WARNING
                warning('CRITERION ''Z'' not allowed in large-scale: switching to ''obj''.')
            end
            CRITERION = 'obj';
            rel_err_fun = @rel_err_obj;
        else
            rel_err_fun = @rel_err_Z;
        end
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
    % check MAX_RANK
    if ~isscalar(MAX_RANK) || (MAX_RANK < 1)
        disp('Error: MAX_RANK must be a scalar >= 1.');
        return
    end
    % update MAX_RANK to safe-guard against large ranks
    % If you really need a large-rank solution, then change the default/comment this part out.
    if (nmin > 2000) && (MAX_RANK>500)
        if WARNING
            warning('Safe-guard against large ranks: MAX_RANK value was decreased to 500.');
        end
        MAX_RANK = 500;
    end
    if (nnz(Xobs)>1e7) && (MAX_RANK>50)
        if WARNING
            warning('Safe-guard against large ranks: MAX_RANK value was decreased to 50.');
        end
        MAX_RANK = 50;
    end
    % check INCREMENT
    if ~isscalar(INCREMENT) || (INCREMENT <1)
        disp('Error: INCREMENT must be a scalar >= 1.');
        return
    end
end
% check VARIANT
if isempty(VARIANT)
    disp('Error: VARIANT must be a string.');
    return
end
switch VARIANT
    case 'soft'
        % check parameters are available
        if ~exist('lambda', 'var')
            disp('Error: Soft variant requires lambda parameter in PARAM struct.');
            return
        end
        % check lambda
        if ~isscalar(lambda) || (lambda<0)
            disp('Error: lambda must be a scalar >= 0');
            return
        end
        % set E step function
        e_step_fun = @(sv) lambda;
        % set penalization function
        pen_fun = @(sv) lambda*sum(sv);
        % disable ESTIM_SIGMA
        ESTIM_SIGMA = 0;
    case 'gamma'
        % check parameters are available
        if ~exist('lambda', 'var') || ~exist('beta', 'var')
            disp('Error: Gamma prior variant requires lambda AND beta parameters in PARAM struct.');
            return
        end
        % check lambda
        if ~isscalar(lambda) || (lambda<0)
            disp('Error: lambda must be a scalar >= 0.');
            return
        end
        % check beta
        if ~isscalar(beta) || (beta <=0)
            disp('Error: beta must be a scalar > 0.');
            return
        end
        % set E step function
        e_step_fun = @(sv) (lambda*beta + 1)./(beta + sv);
        % set penalization function
        pen_fun = @(sv) (lambda*beta + 1)*sum(log(beta + sv));
    case 'igauss'
        % check parameters are available
        if ~exist('lambda', 'var') || ~exist('delta', 'var')
            disp('Error: Inverse Gaussian prior variant requires lambda AND delta parameters in PARAM struct.');
            return
        end
        % check lambda
        if ~isscalar(lambda) || (lambda<0)
            disp('Error: lambda must be a scalar >= 0.');
            return
        end
        % check delta
        if ~isscalar(delta) || (delta <=0)
            disp('Error: delta must be a scalar > 0.');
            return
        end
        
        gamma = delta/lambda;
        
        e_step_fun = @(sv) exp(log(delta) - 0.5*log(gamma^2+2*sv) ...
            + log(1 + 1./(delta*sqrt(gamma^2+2*sv))) );
        % set penalization function
        pen_fun = @(sv) sum(-log(delta)+0.5*log(gamma^2+2*sv)) ...
            - delta*sum(gamma-sqrt(gamma^2+2*sv));
    case 'gig'
        % check parameters are available
        if ~exist('lambda', 'var') || ~exist('beta', 'var') || ~exist('delta', 'var')
            disp('Error: GiG prior variant requires lambda AND beta AND delta parameters in PARAM struct.');
            return
        end
        % check lambda
        if ~isscalar(lambda) || (lambda<0)
            disp('Error: lambda must be a scalar >= 0.');
            return
        end
        % check beta
        if ~isscalar(beta) || (beta <=0)
            disp('Error: beta must be a scalar > 0.');
            return
        end
        % check delta
        if ~isscalar(delta) || (delta <=0)
            disp('Error: delta must be a scalar > 0.');
            return
        end
        % set E step function
%         nu = lambda*beta;
        nu = find_nu(lambda, delta, beta);
        e_step_fun = @(sv) exp(log(delta)-0.5*log(2*beta+2*sv) ...
            + log(besselk(nu+2, delta*sqrt(2*beta+2*sv), 1)) ...
            - log(besselk(nu+1, delta*sqrt(2*beta+2*sv), 1)));
        % set penalization function
        gamma = sqrt(2*beta);
        C1 = log(delta*gamma^nu/besselk(nu, delta*gamma));
        pen_fun = @(sv) -sum( C1 + log(besselk(nu+1, delta*sqrt(2*beta+2*sv), 1)) ...
            - delta*sqrt(2*beta+2*sv) - 0.5*(nu+1)*log(2*beta+2*sv) );
    otherwise
        disp('Error: VARIANT value is not allowed.');
        return
end

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
if ESTIM_SIGMA
    hist.sigma = zeros(MAXITER,1);
end

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
if SMALL_SCALE
    Z = U*V';
    Xcomp = Xobs;
end
sv = diag(D);
if length(sv)<nmin
    sv(nmin,1) = 0;
end
ind_sv = sv>0;
nb_sv = sum(ind_sv);

temp = data-temp;
f_norm_sq = temp'*temp;
obj = .5/sigma^2*f_norm_sq + pen_fun(sv);
if ESTIM_SIGMA
    obj = obj + 2*log(sigma);
end

crit = eval(CRITERION);
tol_curr = Inf;
i = 0;
warn_MAX_RANK = 0;

%% Iterations
while (tol_curr>TOLERANCE) && (i<MAXITER)
    i=i+1;
    crit_old = crit;
    nb_sv_old = nb_sv;
    
    if ESTIM_SIGMA
        sigma = sqrt(a_sigma + norm(Z-Xcomp, 'fro')^2) / sqrt(b_sigma + nrow*ncol);
    end
    
    % E step
    weights = sigma^2.*e_step_fun(sv);
    
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
        
        [U,D,V] = svd(Xcomp,'econ');
    else
        if BINARY %%%% BINARY case
            error('large scale not available for binary data')
            %%%%%%%%%%%%%% TODO : allow large scale %%%%%%%%%%%%%%%%%%%
        else
            Front_multi = @(x)A_multiply_fun_handle(x,Xobs,U,V,Zobs);
            Front_Transpose_multi = @(x)At_multiply_fun_handle(x,Xobs,U,V,Zobs);
            
            [U,D,V] = lansvd_thres(Front_multi, Front_Transpose_multi, nrow, ncol, max(nb_sv,5), weights);
        end
    end
    
    sv = diag(D);
    if length(sv)<nmin
        sv(nmin,1) = 0;
    end
    
    % weighted soft-threshold of singular values
    sv = max(sv-weights, 0);
    ind_sv = sv>0;
    nb_sv = sum(ind_sv);
    if ~SMALL_SCALE && nb_sv==MAX_RANK
        warn_MAX_RANK = 1;
    end
    
    % early exit if iterate turns to be zero.
    if nb_sv==0
        U = zeros(nrow,1);
        D = 0;
        V = zeros(ncol,1);
        if nb_sv_old == 0
            if WARNING
                disp('Early exit: threshold too large.');
            end
            obj = .5/sigma^2*(data'*data) + pen_fun(sv);
            if ESTIM_SIGMA
                obj = obj + 2*log(sigma);
            end
            % save history
            hist = save_history(hist,i);
            hist = end_history(hist,i);
            end_warn();
            return
        end
    else
        % compress
        D = spdiags(sv(ind_sv), 0, nb_sv, nb_sv);
        U = U(:,ind_sv);
        U = U*D; % mix singular values inside U, temporarily
        V = V(:,ind_sv);
    end
    
    if n_obs < 1e5
        temp = sum(U(i_row,:).*V(j_col,:),2);
    else
        temp = project_obs_UV(U, V, i_row, j_col, n_obs);
    end
    
    Zobs = sparse(i_row,j_col, temp, nrow, ncol);
    if SMALL_SCALE
        Z = U*V';
    end
    
    temp = data-temp;
    obj = .5/sigma^2*(temp'*temp) + pen_fun(sv);
    if ESTIM_SIGMA
        obj = obj + 2*log(sigma);
    end
    
    crit = eval(CRITERION);
    tol_curr = rel_err_fun(crit, crit_old);
    
    % save history
    hist = save_history(hist,i);
end

hist = end_history(hist,i);

% divide U by singular values
if nb_sv~=0
    U = U*spdiags(1./sv(ind_sv), 0, nb_sv, nb_sv);
end

end_warn();


%% Function rel_err_obj
% relative error of the objective function
    function e = rel_err_obj(obj_new, obj_old)
        e = abs(obj_new-obj_old)/(obj_old+1e-6);
    end

%% Function rel_err_Z
% relative error of the estimated matrix Z
    function e = rel_err_Z(Z_new, Z_old)
        e = norm(Z_new-Z_old,'fro')^2/norm(Z_old,'fro')^2;
    end

%% Function save_history
% store the history at iteration i
    function hist = save_history(hist, i)
        if FLAG_hist
            hist.obj(i) = obj;
            hist.nnorm(i) = sum(sv(ind_sv));
            hist.rank(i) = nb_sv;
            if ESTIM_SIGMA
                hist.sigma(i) = sigma;
            end
        end
    end
%% Function end_history
% resize the hsitory at the end of the iterations
    function hist = end_history(hist, i)
        if FLAG_hist
            hist.time = toc(t0);
            hist.iter = i;
            hist.obj = hist.obj(1:i);
            hist.nnorm = hist.nnorm(1:i);
            hist.rank = hist.rank(1:i);
            if ESTIM_SIGMA
                hist.sigma = hist.sigma(1:i);
            end
        end
    end
%% Function end_warn
% warn if limits have been reached at the end of the iterations
    function end_warn()
        if warn_MAX_RANK && WARNING
            warning('MAX_RANK limit has been reached.');
        end
        if i==MAXITER && WARNING>=2
            warning('MAXITER limit has been reached.');
        end
    end

end % main function
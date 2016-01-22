function [U,B_k,V,p,ierr,work] = lanbpro_fnhandle(varargin)

% LANBPRO Lanczos bidiagonalization with partial reorthogonalization.
%from original PROPACK, modified by RM for specific function-handles,
% required by soft_impute.m to perform Soft-Impute

% expected calling sequence for function-handles
% [U,B,V,p,ierr,w] = lanbpro_fnhandle(A,Atrans,m,n,j,p,options,U,B,V,anorm);

% Warning!!! : DO NOT use as a stand-alone function, use "lanbpro" instead



global LANBPRO_TRUTH
LANBPRO_TRUTH=0;

if LANBPRO_TRUTH==1
  global MU NU MUTRUE NUTRUE
  global MU_AFTER NU_AFTER MUTRUE_AFTER NUTRUE_AFTER
end

narg=length(varargin);

A = varargin{1};
%% read off function handle
  Atrans = varargin{2};
  m = varargin{3};
  n = varargin{4};
  k=varargin{5};
  p=varargin{6}; options=varargin{7}; 
  U = varargin{8}; B_k = varargin{9}; V = varargin{10}; anorm=varargin{11}; 

% Set options.
m2 = 3/2;
n2 = 3/2;
delta = sqrt(eps/k); % Desired level of orthogonality.
eta = eps^(3/4)/sqrt(k);    % Level of orth. after reorthogonalization.
cgs = 0;             % Flag for switching between iterated MGS and CGS.
elr = 2;             % Flag for switching extended local
% reorthogonalization on and off.
gamma = 1/sqrt(2);   % Tolerance for iterated Gram-Schmidt.
onesided = 0; t = 0; waitb = 0;

%skip: Parse options struct (empty passed from lansvd_try1)

if isempty(anorm)
    anorm = []; est_anorm=1;
else
    est_anorm=0;
end

% Conservative statistical estimate on the size of round-off terms. 
% Notice that {\bf u} == eps/2.
FUDGE = 1.01; % Fudge factor for ||A||_2 estimate.
npu = 0; npv = 0; ierr = 0;
p = p(:);


% Prepare for Lanczos iteration.
if isempty(U)
  V = zeros(n,k); U = zeros(m,k);
  beta = zeros(k+1,1); alpha = zeros(k,1);
  beta(1) = norm(p);
  % Initialize MU/NU-recurrences for monitoring loss of orthogonality.
  nu = zeros(k,1); mu = zeros(k+1,1);
  mu(1)=1; nu(1)=1;
  
  numax = zeros(k,1); mumax = zeros(k,1);
  force_reorth = 0;  nreorthu = 0; nreorthv = 0;
  j0 = 1;
else
  j = size(U,2); % Size of existing factorization
  % Allocate space for Lanczos vectors
  U = [U, zeros(m,k-j)];
  V = [V, zeros(n,k-j)];
  alpha = zeros(k+1,1);  beta = zeros(k+1,1);
  alpha(1:j) = diag(B_k); if j>1 beta(2:j) = diag(B_k,-1); end
  beta(j+1) = norm(p);
  % Reorthogonalize p.
  if j<k & beta(j+1)*delta < anorm*eps,
    fro = 1;
    ierr = j;
  end
  int = [1:j]';
  [p,beta(j+1),rr] = reorth(U,p,beta(j+1),int,gamma,cgs);
  npu =  rr*j;  nreorthu = 1;  force_reorth= 1;  

  % Compute Gerscgorin bound on ||B_k||_2
  if est_anorm
    anorm = FUDGE*sqrt(norm(B_k'*B_k,1));
  end
  mu = m2*eps*ones(k+1,1); nu = zeros(k,1);
  numax = zeros(k,1); mumax = zeros(k,1);
  force_reorth = 1;  nreorthu = 0; nreorthv = 0;
  j0 = j+1;
end


if delta==0
  fro = 1; % The user has requested full reorthogonalization.
else
  fro = 0;
end

if LANBPRO_TRUTH==1
  MUTRUE = zeros(k,k); NUTRUE = zeros(k-1,k-1);
  MU = zeros(k,k); NU = zeros(k-1,k-1);
  
  MUTRUE_AFTER = zeros(k,k); NUTRUE_AFTER = zeros(k-1,k-1);
  MU_AFTER = zeros(k,k); NU_AFTER = zeros(k-1,k-1);
end

% Perform Lanczos bidiagonalization with partial reorthogonalization.
for j=j0:k

  if beta(j) ~= 0
    U(:,j) = p/beta(j);
  else
    U(:,j) = p;
  end

  % Replace norm estimate with largest Ritz value.
  if j==6
    B = [[diag(alpha(1:j-1))+diag(beta(2:j-1),-1)]; ...
      [zeros(1,j-2),beta(j)]];
    anorm = FUDGE*norm(B);
    est_anorm = 0;
  end

%%%%%%%%%% Lanczos step to generate v_j. %%%%%%%%%%%%%
if j==1
   % if isnumeric(A)
   %   r = At*U(:,1);    
   % elseif isstruct(A)
   %   r = A.R\U(:,1);          
   % else
      r = feval(Atrans,U(:,1));
   % end
    
    alpha(1) = norm(r);
    if est_anorm
      anorm = FUDGE*alpha(1);
    end
else    
    %if isnumeric(A)
    %  r = At*U(:,j) - beta(j)*V(:,j-1);
    %elseif isstruct(A)
    %  r = A.R\U(:,j) - beta(j)*V(:,j-1);      
    %else
      r = feval(Atrans,U(:,j))  - beta(j)*V(:,j-1);
    %end
    alpha(j) = norm(r); 

    % Extended local reorthogonalization    
    if alpha(j)<gamma*beta(j) & elr & ~fro
      normold = alpha(j);
      stop = 0;
                while ~stop
	        t = V(:,j-1)'*r;
	        r = r - V(:,j-1)*t;
	        alpha(j) = norm(r);
	        if beta(j) ~= 0
	        beta(j) = beta(j) + t;
	        end
	        if alpha(j)>=gamma*normold
	        stop = 1;
	        else
	        normold = alpha(j);
        	end
    end
end

    if est_anorm
      if j==2
	anorm = max(anorm,FUDGE*sqrt(alpha(1)^2+beta(2)^2+alpha(2)*beta(2)));
      else	
	anorm = max(anorm,FUDGE*sqrt(alpha(j-1)^2+beta(j)^2+alpha(j-1)* ...
	    beta(j-1) + alpha(j)*beta(j)));
      end			     
    end
    
    if ~fro & alpha(j) ~= 0
      % Update estimates of the level of orthogonality for the
      %  columns 1 through j-1 in V.
      nu = update_nu(nu,mu,j,alpha,beta,anorm);
      numax(j) = max(abs(nu(1:j-1)));
    end

    if j>1 & LANBPRO_TRUTH
      NU(1:j-1,j-1) = nu(1:j-1);
      NUTRUE(1:j-1,j-1) = V(:,1:j-1)'*r/alpha(j);
    end
    
    if elr>0
      nu(j-1) = n2*eps;
    end
    
    % IF level of orthogonality is worse than delta THEN 
    %    Reorthogonalize v_j against some previous  v_i's, 0<=i<j.
    if onesided~=-1 & ( fro | numax(j) > delta | force_reorth ) & alpha(j)~=0
      % Decide which vectors to orthogonalize against:
      if fro | eta==0
	int = [1:j-1]';
      elseif force_reorth==0
	int = compute_int(nu,j-1,delta,eta,0,0,0);
      end
      % Else use int from last reorth. to avoid spillover from mu_{j-1} 
      % to nu_j.
      
      % Reorthogonalize v_j 
      [r,alpha(j),rr] = reorth(V,r,alpha(j),int,gamma,cgs);
      npv = npv + rr*length(int); % number of inner products.
      nu(int) = n2*eps;  % Reset nu for orthogonalized vectors.

      % If necessary force reorthogonalization of u_{j+1} 
      % to avoid spillover
      if force_reorth==0 
	force_reorth = 1; 
      else
	force_reorth = 0; 
      end
      nreorthv = nreorthv + 1;
    end
  end


 % Check for convergence or failure to maintain semiorthogonality
  if alpha(j) < max(n,m)*anorm*eps & j<k, 
    % If alpha is "small" we deflate by setting it
    % to 0 and attempt to restart with a basis for a new 
    % invariant subspace by replacing r with a random starting vector:
    %j
    %disp('restarting, alpha = 0')
    alpha(j) = 0;
    bailout = 1;
    for attempt=1:3    
      r = rand(m,1)-0.5;  
%      if isnumeric(A)
%	r = At*r;    
%      elseif isstruct(A)
%	r = A.R\r;    
%      else
	r = feval(Atrans,r);
%      end
      nrm=sqrt(r'*r); % not necessary to compute the norm accurately here.
      int = [1:j-1]';
      [r,nrmnew,rr] = reorth(V,r,nrm,int,gamma,cgs);
      npv = npv + rr*length(int(:));        nreorthv = nreorthv + 1;
      nu(int) = n2*eps;
      if nrmnew > 0
	% A vector numerically orthogonal to span(Q_k(:,1:j)) was found. 
	% Continue iteration.
	bailout=0;
	break;
      end
    end
    if bailout
      j = j-1;
      ierr = -j;
      break;
    else
      r=r/nrmnew; % Continue with new normalized r as starting vector.
      force_reorth = 1;
      if delta>0
	fro = 0;    % Turn off full reorthogonalization.
      end
    end       
  elseif  j<k & ~fro & anorm*eps > delta*alpha(j)
%    fro = 1;
    ierr = j;
  end

  if j>1 & LANBPRO_TRUTH
    NU_AFTER(1:j-1,j-1) = nu(1:j-1);
    NUTRUE_AFTER(1:j-1,j-1) = V(:,1:j-1)'*r/alpha(j);
  end

  
  if alpha(j) ~= 0
    V(:,j) = r/alpha(j);
  else
    V(:,j) = r;
  end

  %%%%%%%%%% Lanczos step to generate u_{j+1}. %%%%%%%%%%%%%
  % if waitb
  %  waitbar((2*j+1)/(2*k),waitbarh)
  %end
  
%  if isnumeric(A)
%    p = A*V(:,j) - alpha(j)*U(:,j);
%  elseif isstruct(A)
%    p = A.Rt\V(:,j) - alpha(j)*U(:,j);
%  else
    p = feval(A,V(:,j)) - alpha(j)*U(:,j);
%  end
  beta(j+1) = norm(p);
  % Extended local reorthogonalization
  if beta(j+1)<gamma*alpha(j) & elr & ~fro
    normold = beta(j+1);
    stop = 0;
    while ~stop
      t = U(:,j)'*p;
      p = p - U(:,j)*t;
      beta(j+1) = norm(p);
      if alpha(j) ~= 0 
	alpha(j) = alpha(j) + t;
      end
      if beta(j+1) >= gamma*normold
	stop = 1;
      else
	normold = beta(j+1);
      end
    end
  end

  if est_anorm
    % We should update estimate of ||A||  before updating mu - especially  
    % important in the first step for problems with large norm since alpha(1) 
    % may be a severe underestimate!  
    if j==1
      anorm = max(anorm,FUDGE*hypot(alpha(1),beta(2))); 
    else
      anorm = max(anorm,FUDGE*sqrt(alpha(j)^2+beta(j+1)^2 + alpha(j)*beta(j)));
    end
  end
  
  
  if ~fro & beta(j+1) ~= 0
    % Update estimates of the level of orthogonality for the columns of V.
    mu = update_mu(mu,nu,j,alpha,beta,anorm);
    mumax(j) = max(abs(mu(1:j)));  
  end

  if LANBPRO_TRUTH==1
    MU(1:j,j) = mu(1:j);
    MUTRUE(1:j,j) = U(:,1:j)'*p/beta(j+1);
  end
  
  if elr>0
    mu(j) = m2*eps;
  end
  
  % IF level of orthogonality is worse than delta THEN 
  %    Reorthogonalize u_{j+1} against some previous  u_i's, 0<=i<=j.
  if onesided~=1 & (fro | mumax(j) > delta | force_reorth) & beta(j+1)~=0
    % Decide which vectors to orthogonalize against.
    if fro | eta==0
      int = [1:j]';
    elseif force_reorth==0
      int = compute_int(mu,j,delta,eta,0,0,0); 
    else
      int = [int; max(int)+1];
    end
    % Else use int from last reorth. to avoid spillover from nu to mu.

%    if onesided~=0
%      fprintf('i = %i, nr = %i, fro = %i\n',j,size(int(:),1),fro)
%    end
    % Reorthogonalize u_{j+1} 
    [p,beta(j+1),rr] = reorth(U,p,beta(j+1),int,gamma,cgs);    
    npu = npu + rr*length(int);  nreorthu = nreorthu + 1;    

    % Reset mu to epsilon.
    mu(int) = m2*eps;    
    
    if force_reorth==0 
      force_reorth = 1; % Force reorthogonalization of v_{j+1}.
    else
      force_reorth = 0; 
    end
  end
  
  % Check for convergence or failure to maintain semiorthogonality
  if beta(j+1) < max(m,n)*anorm*eps  & j<k,     
    % If beta is "small" we deflate by setting it
    % to 0 and attempt to restart with a basis for a new 
    % invariant subspace by replacing p with a random starting vector:
    %j
    %disp('restarting, beta = 0')
    beta(j+1) = 0;
    bailout = 1;
    for attempt=1:3    
      p = rand(n,1)-0.5;  
     % if isnumeric(A)
     %p = A*p;
     % elseif isstruct(A)
%	p = A.Rt\p;
 %     else
	p = feval(A,p);
%      end
      nrm=sqrt(p'*p); % not necessary to compute the norm accurately here.
      int = [1:j]';
      [p,nrmnew,rr] = reorth(U,p,nrm,int,gamma,cgs);
      npu = npu + rr*length(int(:));  nreorthu = nreorthu + 1;
      mu(int) = m2*eps;
      if nrmnew > 0
	% A vector numerically orthogonal to span(Q_k(:,1:j)) was found. 
	% Continue iteration.
	bailout=0;
	break;
      end
    end
    if bailout
      ierr = -j;
      break;
    else
      p=p/nrmnew; % Continue with new normalized p as starting vector.
      force_reorth = 1;
      if delta>0
	fro = 0;    % Turn off full reorthogonalization.
      end
    end       
  elseif  j<k & ~fro & anorm*eps > delta*beta(j+1) 
%    fro = 1;
    ierr = j;
  end  
  
  if LANBPRO_TRUTH==1
    MU_AFTER(1:j,j) = mu(1:j);
    MUTRUE_AFTER(1:j,j) = U(:,1:j)'*p/beta(j+1);
  end  
end
%if waitb
%  close(waitbarh)
%end

if j<k
  k = j;
end

B_k = spdiags([alpha(1:k) [beta(2:k);0]],[0 -1],k,k);
if nargout==1
  U = B_k;
elseif k~=size(U,2) | k~=size(V,2)  
  U = U(:,1:k);
  V = V(:,1:k);
end
if nargout>5
  work = [[nreorthu,npu];[nreorthv,npv]];
end



function mu = update_mu(muold,nu,j,alpha,beta,anorm)

% UPDATE_MU:  Update the mu-recurrence for the u-vectors.
%
%   mu_new = update_mu(mu,nu,j,alpha,beta,anorm)

%  Rasmus Munk Larsen, DAIMI, 1998.

binv = 1/beta(j+1);
mu = muold;
eps1 = 100*eps/2;
if j==1
  T = eps1*(hypot(alpha(1),beta(2)) + hypot(alpha(1),beta(1)));
  T = T + eps1*anorm;
  mu(1) = T / beta(2);
else
  mu(1) = alpha(1)*nu(1) - alpha(j)*mu(1);
%  T = eps1*(hypot(alpha(j),beta(j+1)) + hypot(alpha(1),beta(1)));
  T = eps1*(sqrt(alpha(j).^2+beta(j+1).^2) + sqrt(alpha(1).^2+beta(1).^2));
  T = T + eps1*anorm;
  mu(1) = (mu(1) + sign(mu(1))*T) / beta(j+1);
  % Vectorized version of loop:
  if j>2
    k=2:j-1;
    mu(k) = alpha(k).*nu(k) + beta(k).*nu(k-1) - alpha(j)*mu(k);
    %T = eps1*(hypot(alpha(j),beta(j+1)) + hypot(alpha(k),beta(k)));
    T = eps1*(sqrt(alpha(j).^2+beta(j+1).^2) + sqrt(alpha(k).^2+beta(k).^2));
    T = T + eps1*anorm;
    mu(k) = binv*(mu(k) + sign(mu(k)).*T);
  end
%  T = eps1*(hypot(alpha(j),beta(j+1)) + hypot(alpha(j),beta(j)));
  T = eps1*(sqrt(alpha(j).^2+beta(j+1).^2) + sqrt(alpha(j).^2+beta(j).^2));
  T = T + eps1*anorm;
  mu(j) = beta(j)*nu(j-1);
  mu(j) = (mu(j) + sign(mu(j))*T) / beta(j+1);
end  
mu(j+1) = 1;


function nu = update_nu(nuold,mu,j,alpha,beta,anorm)

% UPDATE_MU:  Update the nu-recurrence for the v-vectors.
%
%  nu_new = update_nu(nu,mu,j,alpha,beta,anorm)

%  Rasmus Munk Larsen, DAIMI, 1998.

nu = nuold;
ainv = 1/alpha(j);
eps1 = 100*eps/2;
if j>1
  k = 1:(j-1);
%  T = eps1*(hypot(alpha(k),beta(k+1)) + hypot(alpha(j),beta(j)));
  T = eps1*(sqrt(alpha(k).^2+beta(k+1).^2) + sqrt(alpha(j).^2+beta(j).^2));
  T = T + eps1*anorm;
  nu(k) = beta(k+1).*mu(k+1) + alpha(k).*mu(k) - beta(j)*nu(k);
  nu(k) = ainv*(nu(k) + sign(nu(k)).*T);
end
nu(j) = 1;


%{
function x = pythag(y,z)
%PYTHAG Computes sqrt( y^2 + z^2 ).
%
% x = pythag(y,z)
%
% Returns sqrt(y^2 + z^2) but is careful to scale to avoid overflow.

% Christian H. Bischof, Argonne National Laboratory, 03/31/89.

[m n] = size(y);
if m>1 | n>1
  y = y(:); z=z(:);
  rmax = max(abs([y z]'))';
  id=find(rmax==0);
  if length(id)>0
    rmax(id) = 1;
    x = rmax.*sqrt((y./rmax).^2 + (z./rmax).^2);
    x(id)=0;
  else
    x = rmax.*sqrt((y./rmax).^2 + (z./rmax).^2);
  end
  x = reshape(x,m,n);
else
  rmax = max(abs([y;z]));
  if (rmax==0)
    x = 0;
  else
    x = rmax*sqrt((y/rmax)^2 + (z/rmax)^2);
  end
end
  
%}





























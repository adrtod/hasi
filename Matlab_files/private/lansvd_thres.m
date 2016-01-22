function [U,S,V]=lansvd_thres(varargin)
% WARNING!!!: DO NOT USE as a standalone program
% Modified to suit being called by HASI (ie ha_soft_impute.m) 
% and also allows for the no.singular vectors to be increased dynamically

% Matlab code written by Adrien Todeschini <adrien.todeschini@inria.fr>
%
% This file is adapted from lansvd_lambda.m of the Soft-Impute
% Matlab code written by Rahul Mazumder <rahulm@stanford.edu>
% This is essentially LANSVD() of R.M. Larsen's PROPACK.


global MAX_RANK INCREMENT


options = [];
A = varargin{1};
Atrans = varargin{2};
m = varargin{3};   n = varargin{4};
k=varargin{5}; 
thres=varargin{6}; 


%skipping the check condn in lansvd for min(m,n)<=1 || k<1  

% skip: A is the matrix of all zeros (not detectable if A is defined by an m-file)

lanmax = min(m,n);
tol = 16*eps;
p = rand(m,1)-0.5;

%%s_target = inf;
%%skip:  Parse options struct
%%skip: Protect against absurd options, since they are all @ default

%%%%%%%%%%%%%%%%%%%%% Here begins the computation  %%%%%%%%%%%%%%%%%%%%%%
%%skip: column permutatn, since we dont need to compute the smallest singular values
%%IMPLICIT = isstr(A) || isa(A,'function_handle');

ksave = k;
neig = 0; nrestart=-1;
 j= min(k+max(8,k)+1,lanmax);
U = []; V = []; B = []; anorm = []; work = zeros(2,2);

while neig < k
    
 %%%%%%%%%%%%%%%%%%%%% Compute Lanczos bidiagonalization %%%%%%%%%%%%%%%%%

% if isnumeric(A) 
%    [U,B,V,p,ierr,w] = lanbpro(A,j,p,options,U,B,V,anorm);
%    [U,B,V,p,ierr,w] = lanbpro_try1(A,j,p,options,U,B,V,anorm);
%    [U,B,V,p,ierr,w] = lanbpro_fnhandle(A,j,p,options,U,B,V,anorm);

%  else
%    [U,B,V,p,ierr,w] = lanbpro(A,Atrans,m,n,j,p,options,U,B,V,anorm);
%    [U,B,V,p,ierr,w] = lanbpro_try1(A,Atrans,m,n,j,p,options,U,B,V,anorm);
    [U,B,V,p,ierr,w] = lanbpro_fnhandle(A,Atrans,m,n,j,p,options,U,B,V,anorm);

%  end

  work= work + w;

    if ierr<0 % Invariant subspace of dimension -ierr found.
        j = -ierr;
    end


 %%%%%%%%%%%%%%%%%% Compute singular values and error bounds %%%%%%%%%%%%%%%%
   % Analyze B
   resnrm = norm(p);
   
   [S,bot] = bdsqr(diag(B),[diag(B,-1); resnrm]);
   
% Use Largest Ritz value to estimate ||A||_2. This might save some
 % reorth. in case of restart.

  anorm=S(1);
  
  % Set simple error bounds
  bnd = resnrm*abs(bot);
  
  % Examine gap structure and refine error bounds
  bnd = refinebounds(S.^2,bnd,n*eps*anorm);

%%%%%%%%%%%%%%%%%%% Check convergence criterion %%%%%%%%%%%%%%%%%%%%
  i=1;
  neig = 0;
  while i<=min(j,k) 
    if (bnd(i) <= tol*abs(S(i)))
      neig = neig + 1;
      i = i+1;
    else
      i = min(j,k)+1;
    end
  end

%%%%%%%%%% Check whether to stop or to extend the Krylov basis? %%%%%%%%%%
  if ierr<0 % Invariant subspace found
    if j<k
      warning(['Invariant subspace of dimension ',num2str(j-1),' found.'])
    end
    j = j-1;
    break;
  end
  if j>=lanmax % Maximal dimension of Krylov subspace reached. Bail out
    if j>=min(m,n)
      neig = ksave;      
      break;
    end
    if neig<ksave
      warning(['Maximum dimension of Krylov subspace exceeded prior',...
	    ' to convergence.']);
    end
    break;
  end
  
 % Increase dimension of Krylov subspace
  if neig>0
    % increase j by approx. half the average number of steps pr. converged
    % singular value (j/neig) times the number of remaining ones (k-neig).
    j = j + min(100,max(2,0.5*(k-neig)*j/(neig+1)));
  else
    % As long a very few singular values have converged, increase j rapidly.
    %    j = j + ceil(min(100,max(8,2^nrestart*k)));
    j = max(1.5*j,j+10);
  end
  j = ceil(min(j+1,lanmax));
  nrestart = nrestart + 1;

 
   %Modified by RM based on SRB@caltech adding: check if smallest singular value is less than
   %the threshold; if it isn't, then increase k
    min_sv_thres = min(abs( S(1: min(j,k)))-thres(1: min([j,k, length(thres)])));
    if (min_sv_thres > 0) && (k< MAX_RANK)
        k = min(k + INCREMENT,MAX_RANK);  ksave = k;
        j2 = ceil( min(k+max(8,k)+1,lanmax) );
        j2 = min(j2, lanmax) ;
        j = max( j, j2 );
    end
   


end


%%%%%%%%%%%%%%%% Lanczos converged (or failed). Prepare output %%%%%%%%%%%%%%%
k = min(ksave,j);

%%if nargout>2 && min(S(1:k))<s_target
  j = size(B,2);
  % Compute singular vectors
  [P,S,Q] = svd(full([B;[zeros(1,j-1),resnrm]]),0); 
  S = diag(S);
  if size(Q,2)~=k
    Q = Q(:,1:k); 
    P = P(:,1:k); 
  end
  % Compute and normalize Ritz vectors (overwrites U and V to save memory).
  if resnrm~=0
    U = U*P(1:j,:) + (p/resnrm)*P(j+1,:);
  else
    U = U*P(1:j,:);
  end
  V = V*Q;
  for i=1:k     
    nq = norm(V(:,i));
    if isfinite(nq) && nq~=0 && nq~=1
      V(:,i) = V(:,i)/nq;
    end
    nq = norm(U(:,i));
    if isfinite(nq) && nq~=0 && nq~=1
      U(:,i) = U(:,i)/nq;
    end
  end

%%%end

% Pick out desired part the spectrum
S = S(1:k);
bnd = bnd(1:k);
  
S = diag(S);

% if min(S)>s_target
%    U=[];
%    V=[];
%  end























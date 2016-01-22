function e=spectral_norm(S)
% computes the largest singular value of a sparse matrix.


%% calls lansvd to compute the largest singular value

e=lansvd(S,1,'L');

%{
% the following is based on the MATLAB normest function, modified by
% RM (rahulm@stanford.edu)
%function to estimate the spectral norm of a (sparse) matrix

tol = 1.e-6; 
maxiter = 100; % should never take this many iterations. 
x = sum(abs(S),1)';
cnt = 0;
e = norm(x);
if e == 0, return, end
x = x/e;
e0 = 0;
while (abs(e-e0) > tol*e) & (cnt <=maxiter)
   e0 = e;
   Sx = S*x;

   x = S'*Sx;
   normx = norm(x);
   e = normx/norm(Sx);
   x = x/normx;
   cnt = cnt+1;

end
%}

Software package for soft-impute for nuclear norm minimzation problems for matrix completion with noisy entries.

% min_X  0.5*|| GXobs - P(X) ||_F^2 + lambda ||X||_* --------------------- (A)
% P(X) with size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% GXobs is the observed data matrix with exactly the same sparsity pattern as P(X). 
% lambda: tuning parameter
% ||X||_* = nuclear norm of X ie sum(singular values of X)

1. Please see "demo_soft_impute.m" to see how to use soft_impute on some examples.

2. To run the programs, folders 'Matlab_files' and 'PROPACK_utils' need to be in path.

3. Install mex files by running install_mex (details in "install_mex.m")


Software written by Rahul Mazumder <rahulm@stanford.edu> with considerable input from 
Trevor Hastie <hastie@stanford.edu> 
Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
by Rahul Mazumder, Trevor Hastie, Rob Tibshirani (JMLR vol 11, 2010)


Any bugs/ errors/ comments/ improvements/ suggestions are most welcome. In that case
please email: rahulm@stanford.edu



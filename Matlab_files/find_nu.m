function nu = find_nu(lambda, delta, beta)
% find nu parameter of GiG s.t. expectation is lambda
% with delta and gamma parameters fixed
% !!! returns NaN for large lambda*beta !!!
gamma = sqrt(2*beta);

%% Expectation of GiG distribution w.r.t nu, delta and gamma parameters
lambda_fun = @(nu) lambda-delta/gamma*exp(log(besselk(nu+1,gamma*delta))-log(besselk(nu,gamma*delta)));


%% dichotomy algorithm
tol = 1e-6;
maxiter = 100;

nu_min = 0;
nu_max = lambda*beta;
nu = 0.5*(nu_min+nu_max);

for i=1:maxiter
    if lambda_fun(nu)>0
        nu_min = nu;
    else
        nu_max = nu;
    end
    if (nu_max-nu_min)/nu_max < tol
        break;
    end
    nu = 0.5*(nu_min+nu_max);
end
 function dot_prod=project_obs_UV(U,V,i_row,j_col,no_obs)

disp('Warning: slow matlab "project_obs_UV.m" being called')
disp('Install mex file: "install_mex.m" for better performance')

dot_prod=zeros(no_obs,1);

for i =1 : no_obs 
dot_prod(i)= U(i_row,:)*(V(j_col,:)');

end


function y=A_multiply_fun_handle(x,GXobs,Glr_mat_u,Glr_mat_v,GPmZ_old)
% this function multiplies A*x and returns
% this is required to be used becuse of the special structure of the matrix

y= Glr_mat_v'*x ; clear Glr_mat_v; 
y=Glr_mat_u*y; clear Glr_mat_u  % Low-rank*x

x=sparse(x); 
y=y + GXobs*x; clear GXobs; %% sparse_part1 times x
y=y- GPmZ_old*x; clear GPmZ_old

%%%global nrow ncol
%%%f2=fopen('new_mat_u.txt','r'); Glr_mat_u=fread(f2,[nrow,inf],'double'); fclose(f2); clear f2
%%%Glr_mat_u=double(Glr_mat_u);
%%%load  mat_d.mat
%%%y2=Glr_mat_d*sparse(Glr_mat_u'*x);

%%%%clear Glr_mat_u Glr_mat_d
%%%f1=fopen('new_mat_v.txt','r'); Glr_mat_v=fread(f1,[ncol,inf],'double'); fclose(f1); clear f1
%%%Glr_mat_v=double(Glr_mat_v);
%%%y1=(Glr_mat_v*y2); clear Glr_mat_v y2
%%%load  mat_XPm.mat



%%%clear GXobs
%%%load  mat_ZPm.mat
 




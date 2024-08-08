function [Y, Obj, J_maxtrix] = ECMCAD(B,num_cluster,gamma,lambda)
num_view = size(B, 2);
tol = 1e-6;
max_iter = 10;
tol_iter = 2;
% initialization
U=cell(1,num_view);
H=cell(1,num_view);
Delta=cell(num_view);
K=cell(num_view);
W=cell(num_view);
alpha =1/num_view*ones(1,num_view);
% beta =1/num_view*ones(1,num_view);
Bsum=[];
for v=1:num_view
num_sample = size(B{v}, 1);
num_anchor = size(B{v}, 2);
Bsum = [Bsum B{v}];
U{v} = initializeF(num_sample,num_cluster);
W{v} = initializeF(num_cluster,num_cluster);
H{v} = randn(num_anchor,num_cluster);
Delta{v}=1e0*sqrt(sum(sum((B{v}-U{v}*H{v}').^2)/(2*num_anchor)));
end
C=initializeF(num_cluster,num_cluster);
[y0,~] = litekmeans(Bsum,num_cluster);
Y=sparse(1:num_sample,y0,1,num_sample,num_cluster,num_sample);

J_maxtrix=[];
for iter = 1 : max_iter
    fprintf('----processing iter %d--------\n', iter);
    J=[];
    %% update Wv
    for v=1:num_view
         tempe1=(B{v}-U{v}*H{v}').^2;
         tempe2=(sum(tempe1,2))./(2*Delta{v}^2);
         K{v}=sparse(diag(exp(-tempe2)./(Delta{v}^2)));
    end
    %% update Uv
    for v=1:num_view
        Utemp = 2*((alpha(v))^gamma)*(K{v}*B{v}*H{v}+lambda*Y*C*W{v}');
        [AA1,~,CC1] = svd(Utemp','econ');
        U{v} = (AA1*CC1')';
    end
     %% update Hv
     for v = 1:num_view
        H{v} = B{v}'*U{v};
     end
    %% update Wv;
     for v=1:num_view
        Wtemp =C'*Y'*U{v};
        [AA2,~,CC2] = svd(Wtemp,'econ');
        W{v} = (AA2*CC2')';
     end
    %% update C
     Ctemp=0;
     for v=1:num_view
        Ctemp=Ctemp+(alpha(v))^gamma*U{v}*W{v};
     end
     Ctemp=Y'*Ctemp;
     [AA3,~,CC3] = svd(Ctemp,'econ');
     C = AA3*CC3';  
     %% update Y
      Ytemp=0;
     for v=1:num_view
        Ytemp=Ytemp+(alpha(v))^gamma*U{v}*W{v}*C';
     end
     [~ , yind] = max(Ytemp, [], 2);
     Y=sparse(1:num_sample,yind,1,num_sample,num_cluster,num_sample);
    %% update alpha
    q=zeros(num_view,1);
    for v = 1:num_view
        qtemp = trace((B{v}-U{v}*H{v}')'*K{v}*(B{v}-U{v}*H{v}')+lambda*trace(C'*Y'*U{v}*W{v}));
        r = 1/(1-gamma);
        q(v) = (gamma*qtemp)^r; 
    end
    alpha = q./(sum(q,1));

    %% obj
    objnmf = 0;
    for v = 1:num_view
        objnmf = objnmf + (alpha(v)^gamma)* trace((B{v}-U{v}*H{v}')'*K{v}*(B{v}-U{v}*H{v}')+lambda*trace(C'*Y'*U{v}*W{v}));
    end
    Obj(iter) =objnmf; 
%     if iter > tol_iter &&  abs((Obj(iter) - Obj(iter-1) )/Obj(iter-1))< tol
%         break; 
%     end
    % convergence

end
disp(['iter:', num2str(iter)]);

end



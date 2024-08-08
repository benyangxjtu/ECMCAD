function [x, val,a] = SimplexQP_ALM(A, b, mu, beta, pic, x)
% solve:
% min     x'Ax - x'b
% s.t.    x'1 = 1, x >= 0
% paras:
% mu    - mu > 0
% beta  - 1 < beta < 2
%

NITER = 10000;
THRESHOLD = 1e-8;

val = 0;

if nargin < 6
    x = zeros(size(b));
end
% v = ones(size(x));
v = zeros(size(x));
% TODO: consider the initialization of multipliers
lambda = ones(size(x));
cnt = 0;

if pic
    rec = [];
end

for iter = 1:NITER
    x0=v-1/mu*( lambda+v*A-b);
    x = EProjSimplex_new(x0 );
    v = x + 1/mu*(lambda + x*A);
    lambda = lambda + mu*(x - v);
    mu = beta*mu;
    
    val_old = val;
    val = x*A*x' + b*x';
    
    if pic
        rec = [rec val];
    end
%     obj(iter)=norm(x-v,'fro').^2;
%     if obj(iter) < THRESHOLD
    if (val_old-val) < THRESHOLD%&&(x-v)'*(x-v)
        if cnt >= 5
            break;
        else
            cnt = cnt + 1;
        end
    else
        cnt = 0;
    end
end
a=iter;
%fprintf('Using SimplexQP, relax gap: %.5f, iter: %d, mu: %.3f\n', norm(x-v), iter, mu);
% if pic
%     plot(rec);
% end

end
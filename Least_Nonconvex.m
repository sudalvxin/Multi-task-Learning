%% FUNCTION Least_Trace
%   Log Regularized Learning with Sqrt Least Squares Loss.
%
%% OBJECTIVE
%   argmin_W { sum_i^t (0.5 * sqrt(norm (Y{i} - X{i}' * W(:, i))^2))
%            + rho1 \|W\|_Log}
%% INPUT
%   X: {n * d} * t - input matrix£¬ 
%   Y: {n * 1} * t - output matrix
%   rho1: trace norm regularization parameter
%
%% OUTPUT
%   W: model: d * K
%   funcVal: function value vector.
%% Thanks
%   Thanks  Dr.Jiayu Zhou for providing MATLAB softwar Malsar
%
%% Code starts here
function [W, funcVal] = Least_Nonconvex(X, Y, rho1,opts)

X = multi_transpose(X);  

% initialize options.
task_num  = length (X); 
dimension = size(X{1}, 1); 
funcVal = [];

% precomputation. 
XY = cell(task_num, 1);
W0_prep = [];  
for t_idx = 1: task_num
    XY{t_idx} = X{t_idx}*Y{t_idx};
    W0_prep = cat(2, W0_prep, XY{t_idx}); % cat(2,A,B) is the same as [A,B]. cat(1,A,B) is the same as [A;B].
end
W0 = W0_prep;
I = eye(dimension);
%==============================
st = 10;  % for calculation stability, generaly, a large value generates better performance
%=============================
W = W0;
iternum = opts.maxIter;
iter = 1;
d = ones(1,task_num);
mu = rho1;
while iter < iternum
    
    % update D
    temp = W*W'+ st*I;
    [U S1 V] = mySVD(temp,size(W,1));
    S1 = diag(S1);
    tempd = Lin(S1,1);
    S = diag(tempd);
    D = U*S*V';
    % update W
    for t_ii = 1:task_num
        %=========================================
        % size(X{t_ii},2) is the number of training samples in task T_i
           rho1 = mu*sqrt(size(X{t_ii},2))*d(t_ii);  % no calibration rho1 = mu for each t_ii
        %=========================================
           XXt = X{t_ii}*X{t_ii}';
           XYT =  XY{t_ii};
           temp = (XXt + 2*rho1*D+0.001*I);
           w = temp\XYT;
           W(:,t_ii) = w;
           %===================
           e = Y{t_ii}-X{t_ii}'*w;
           % update v_i
           d(t_ii) = norm(e);
           %===================
    end
    funcVal(iter) = funVal_eval(W) + mu*sum(log(sqrt(S1)+1));
    iter = iter+1;
end
% private functions
 function [funcVal] = funVal_eval(W)
        funcVal = 0;
            for i = 1: task_num
                funcVal = funcVal + 0.5*sqrt(norm(Y{i} - X{i}' * W(:, i),'fro')/size(X{i},2));
            end
    end

end
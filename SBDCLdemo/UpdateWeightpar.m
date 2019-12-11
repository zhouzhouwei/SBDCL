function [ Z, delete_atom_idx, valCost] = UpdateWeightpar(Psi, Y, NumPerClass)
C = length(NumPerClass);
initers = 100 ;
eta = 1e-10; 
tau1 = 1e-2;     % the threshold for pruning the small entries
[M, N] = size(Y) ;
M = M-C;
if ~(sum(NumPerClass)==N)
    disp('The number of instances in each category is wrong!')
    exit
end
K0 = size(Psi,2) ;
D0 = Psi;
Z = cell(1,C) ;
for c=1:C
    Nc = NumPerClass(c) ;
    class_idx(c,:) = sum(NumPerClass(1:c))-Nc+1:sum(NumPerClass(1:c));
end
vals = zeros(C,1);

% main loop
parfor c = 1:C    
    Psi = D0; 
    K = K0;
    lambdas = zeros(initers,1);
    Tc_idx = class_idx(c,:) ;
    Yc = Y(:,Tc_idx) ;
    gamma = 1*ones(K,1);
    lambda = 1e-2 ;
    reserved_idx = (1:K)' ;
    
    for iter = 1:initers
        K = size(Psi,2);
        
        %using the woodbury identity to reduce computational time 
        if M+C<=K         
            Theta = inv(lambda*eye(M+C)+Psi*diag(gamma)*Psi') ;
        else
            Theta = 1/lambda*eye(M+C)-1/lambda*Psi*((diag(lambda./gamma)+Psi'*Psi)\Psi') ;
        end
        Zc = diag(gamma)*Psi'*Theta*Yc;
        
        % set small theta to zero
        Zc(abs(Zc)./vecnorm(Zc)<tau1) = 0;
        
        % update gamma and lambda 
        gamma_numer = vecnorm(Zc,2,2) ;
        gamma = gamma_numer./sqrt(Nc*diag(Psi'*Theta*Psi)) ;
        lambda = norm(Yc-Psi*Zc,'fro')/sqrt(Nc*trace(Theta)) ;
        lambdas(iter,1) = lambda ;
        
        % prune the atoms corresponding to small gamma
        deleted_idx = find(gamma<=eta);
        reserved_idx(deleted_idx)=[];
        gamma(deleted_idx) = [];
        Psi(:,deleted_idx) = [];
        
        % stopping criteria 
        if isempty(deleted_idx)
            if iter>2
                err1 = abs(lambdas(iter,1)-lambdas(iter-1,1));
                if err1<1e-7
                    % fprintf('the iter is %0.1d \n',iter) ;
                    vals(c,1)= trace(Zc'*diag(1./gamma)*Zc) + sum(log(gamma))*size(Zc,2);
                    break;
                end
            end
        end    
    end
    Zc_est = zeros(K0,Nc);
    Zc_est(reserved_idx,:) = Zc;
    Z{c} = Zc_est;
end

% the sparse coding matrix  
Z = cell2mat(Z);
 
% delete the atoms which have on contribution for representing all
% instances
delete_atom_idx = find(~sum(Z,2));
valCost = sum(vals);
end
% =========================================================================
%  The GSBL method of SBDCL algorithm proposed in
%
%   Wei Zhou, Yue Wu, Junlin Li, Maolin Wang, Hai-Tao Zhang,
%   "A Bayesian Approach for Joint Discriminative Dictionary and Classifier
%    Learning", 
%   IEEE Transactions on Systems, Man, and Cybernetics: Systems.
%
%    Copyright (c) 2020 Wei Zhou
%    All rights reserved.
%
% Code by Wei Zhou (946958036@qq.com)
% Date: 04-01-2020
% =========================================================================

function Z = SBLgroup(Phi,test_data, paras)
[M, Ntest] = size(test_data) ;
K0 = size(Phi,2);
iters = 500;
normolized = paras.normlized ;
thre1 = 0.01 ;  % the threshold value for setting the small coefficients to zero
thre2 = 0.4 ;  % the threshold value for deleting the redundant atoms 
lambdas = zeros(iters,Ntest);
Z = zeros(K0,Ntest);


% main loop
for i=1:Ntest
    feats = test_data(:,i);
    Phitest = Phi;
    group = paras.group ;
    numGroup = size(group,1) ;
    gamma = 1e-2*ones(K0,1) ;
    lambda = 1e-2 ;
    reserved_ids = (1:K0)';
    zi = zeros(K0,1);
    if normolized == 1
        y_max = max(abs(feats)) ;
        feats = feats/ y_max ;
        Phi_norm = vecnorm(Phitest) ;
        Phitest = Phitest ./ vecnorm(Phitest) ;
    end
    
    for iter = 1:iters
        [~,KN] = size(Phitest);
        
        % if Phi is over-determined using Woodbury identity to calculate
        % Theta
        if M < KN
            Theta = inv(lambda*eye(M)+Phitest*(gamma.*Phitest')) ;
        else
            Theta = 1/lambda*eye(M) - 1/lambda*Phitest*((diag(lambda./gamma) + Phitest'*Phitest)\Phitest') ;
        end
        zhat = gamma.*(Phitest'*Theta*feats) ;
        
        % % prune the small terms in z 
        zhat(abs(zhat)./norm(zhat) < thre1) = 0;
        
        %update lambda and gamma
        lambda = norm(feats-Phitest*zhat,2)/sqrt(trace(Theta));
        gamma = zeros(KN,1);
        Groupgamma = zeros(numGroup,1);
        temp = diag(Phitest'*Theta*Phitest) ;
        for c=1:numGroup
            idsG = group{c} ;
            gamma_num = norm( zhat(idsG) ) ;
            gamma_den = sqrt(sum( temp(idsG) ));
            Groupgamma(c,1) = gamma_num / gamma_den ;
            gamma(idsG) = Groupgamma(c) ;
        end
        
        % delete redundant atoms
        idsDeleteGroups = find(Groupgamma/max(Groupgamma)<thre2);       
        delete_ids = cell2mat(group(idsDeleteGroups));
        reserved_ids(delete_ids) = [];
        Phitest(:,delete_ids) = [] ;
        gamma(delete_ids) = [];
        group(idsDeleteGroups) = [];
        lambdas(iter,i) = lambda ;
        numGroup = size(group,1) ;
        
        % update the indexes in group
        if ~isempty(idsDeleteGroups)
            ids = (1:KN)';
            ids(delete_ids) = [];
            numGroup = size(group,1) ;
            for c=1:numGroup
                group{c} = find(ismember(ids,group{c})) ;
            end
        end
        
        % stopping condition
        if iter>1
            if isempty(delete_ids)
                if abs(lambdas(iter-1,i)-lambdas(iter,i))<1e-6
                    break;
                end
            end
        end
        
    end
    
    if length(reserved_ids)==length(zhat)
        zi(reserved_ids) = zhat ;
    else
        fprintf('The index of test samples is %d with %d iterations\n',i,iter);
        fprintf('Error! The algorithm has not converged')
    end
    if normolized ==1
        zi = zi * y_max ./Phi_norm' ;
    end
    Z(:,i) = zi ;
end

fprintf('\nThe average sparsityness of the testing samples is %.2f \n', mean(sum(logical(Z))))

end
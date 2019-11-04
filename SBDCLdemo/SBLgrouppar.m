function Z = SBLgrouppar(Phi,test_data,paras)
[M, Ntest] = size(test_data) ;
K0 = size(Phi,2);
iters = 200;
eta = 1e-10;
tau1 = paras.tau1;
Z = zeros(K0,Ntest);

% main loop
parfor i=1:Ntest
    feats = test_data(:,i);
    Phitest = Phi;
    group = paras.group ;
    numGroup = size(group,1) ;
    Gamma = 1e-2*eye(K0) ;
    lambda = 1e-2 ;
    lambdas = zeros(iters,1);
    reserved_ids = (1:K0)';
    theta = zeros(K0,1);
    
    for iter = 1:iters
        [~,KN] = size(Phitest);
        
        % % if Phi is over-determined using Woodbury identity to calculate Dic
        if M < KN
            Theta = inv(lambda*eye(M)+Phitest*Gamma*Phitest') ;
        else
            Theta = 1/lambda*eye(M) - 1/lambda*Phitest*((lambda*inv(Gamma) + Phitest'*Phitest)\Phitest') ; %#ok<MINV>
        end
        zhat = Gamma*Phitest'*Theta*feats;
        
        % % prune the small terms in x
        zhat(abs(zhat)./norm(zhat)<tau1) = 0;
        
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
        idsDeleteGroups = (Groupgamma<eta);
        group(idsDeleteGroups) = [];
        
        % delete redundant atoms
        delete_ids = find(gamma<eta);
        reserved_ids(delete_ids) = [];
        Phitest(:,delete_ids) = [] ;
        gamma(delete_ids) = [];
        Gamma = diag(gamma) ;
        lambdas(iter,1) = lambda ;
        
        % update the indices in group
        ids = (1:KN)';
        ids(delete_ids) = [];
        numGroup = size(group,1) ;
        for c=1:numGroup
            group{c} = find(ismember(ids,group{c})) ;
        end
        
        % stopping condition
        if iter>1
            if isempty(delete_ids)
                if abs(lambdas(iter-1,1)-lambdas(iter,1))<1e-7
                    break;
                end
            end
        end
        
    end
    
    if M < KN
        Theta = inv(lambda*eye(M)+Phitest*Gamma*Phitest') ;
    else
        Theta = 1/lambda*eye(M) - 1/lambda*Phitest*((lambda*inv(Gamma) + Phitest'*Phitest)\Phitest') ; %#ok<MINV>
    end
    zhat = Gamma*Phitest'*Theta*feats;
    zhat(abs(zhat)./norm(zhat)<tau1) = 0;
    if length(reserved_ids)==length(zhat)
        theta(reserved_ids) = zhat ;
    else
        fprintf('The index of test samples is %d with %d iterations\n',i,iter);
        fprintf('Error! The algorithm has not converged\n')
    end

    Z(:,i) = theta ;

end


end
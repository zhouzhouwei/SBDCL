function [ Z, delete_atom_idx, valCost] = UpdateWeightpar(Psi, Y, NumPerClass)
C = length(NumPerClass);
initers = 500 ;
thre2 = 1e-2 ; 
thre1 = 1e-2 ;
[Mbar, N] = size(Y) ;
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
        if Mbar<= K            
            Dic = (lambda*eye(Mbar)+Psi*diag(gamma)*Psi') ;
            PsiDic = Psi'/Dic ;
            Zc = (gamma).*(PsiDic*Yc);
            % set small coefficients to zeros
            Zc(abs(Zc)./vecnorm(Zc)<thre1) = 0;
            % update gamma and lambda
            gamma_numer = vecnorm(Zc,2,2) ;
            gamma = gamma_numer./sqrt(Nc*diag(PsiDic*Psi)) ;
            lambda = norm(Yc-Psi*Zc,'fro')/sqrt(Nc*trace_inv(Dic)) ;
        else
            PsitPsi = Psi'*Psi;
            Dic = diag(lambda./gamma) + PsitPsi ;
            Zc = Dic\Psi'*Yc ;            
            % set small theta to zero
            Zc(abs(Zc)./vecnorm(Zc)<thre1) = 0;
            gamma_numer = vecnorm(Zc,2,2) ;
            % update gamma and lambda
            gamma = gamma_numer./sqrt(Nc*diag(PsitPsi/(lambda*eye(K)+gamma.*PsitPsi)));
            lambda_den = Mbar/lambda - 1/lambda*trace(Psi/Dic*Psi') ;
            lambda = norm(Yc-Psi*Zc,'fro')/sqrt(Nc*lambda_den) ;
        end
        
        lambdas(iter,1) = lambda ;
               
        % prune the atoms corresponding to small gammas
        deleted_idx = find(gamma/max(gamma)<=thre2);
        reserved_idx(deleted_idx)=[];
        gamma(deleted_idx) = [];
        Psi(:,deleted_idx) = [];
               
        % stopping criteria 
        if isempty(deleted_idx)
            if iter>2
                err1 = abs(lambdas(iter,1)-lambdas(iter-1,1));
                if err1<1e-6
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

% the mean sparsity  
fprintf('The average sparsityness of the training samples is %.2f \n', mean(sum(logical(Z))))

% delete the atoms which have on contribution for representing all instances
delete_atom_idx = find(~sum(Z,2));
valCost = sum(vals);
end
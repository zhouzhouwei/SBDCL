function [ Z, delete_atom_idx, valCost] = UpdateWeight(Psi, Y, NumPerClass )
C = length(NumPerClass);
initers = 500 ;
thre1 = 1e-2 ;  % the threshold value for setting the small coefficients to zero
thre2 = 1e-2 ;  % the threshold value for deleting the redundant atoms 
[Mbar, N] = size(Y) ;
if ~(sum(NumPerClass)==N)
    disp('The number of instances in each category is wrong!')
    exit
end
K0 = size(Psi,2) ;
Psi0 = Psi;
Z = zeros(K0,N) ;
lambdas = zeros(initers,C);
vals = zeros(C,1);

% main loop
for c = 1:C
    Nc = NumPerClass(c) ;
    Psi = Psi0; 
    K = K0;
    class_idx = sum(NumPerClass(1:c))-Nc+1:sum(NumPerClass(1:c));
    Yc = Y(:,class_idx) ;
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
        
        lambdas(iter,c) = lambda ;
               
        % prune the atoms corresponding to small gammas
        deleted_idx = find(gamma/max(gamma)<=thre2);
        reserved_idx(deleted_idx)=[];
        gamma(deleted_idx) = [];
        Psi(:,deleted_idx) = [];
               
        % stopping criteria 
        if isempty(deleted_idx)
            if iter>2
                err1 = abs(lambdas(iter,c)-lambdas(iter-1,c));
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
    Z(:,class_idx) = Zc_est;
end
  
% the mean sparsity  
fprintf('The average sparsityness of the training samples is %.2f \n', mean(sum(logical(Z))))

% delete the atoms which have on contribution for representing all instances
delete_atom_idx = find(~sum(Z,2));
valCost = sum(vals);
end
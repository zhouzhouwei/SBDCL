function [Psi_new, E_new, valCost, Betas, lambda_all] = UpdateDic( Y, Psi, Z )
iters = 50;
[M, N] = size(Y);
ids = vecnorm(Psi)<1e-2;
Psi(:,ids) =[];
K = size(Psi,2);
Betas = zeros(1,K);
lambda_all = zeros(1,K);
Vk = Y - Psi * Z + Psi(:,1)*Z(1,:);
 
for k=1:K  
    beta = 1e-2;
    lambda = 1e-2;
    sk = Z(k,:) ;
    % update the contribution matrix 
    if k>1
        Vk = Vk - Psi(:,k-1)*Z(k-1,:) + Psi(:,k)*sk ;
    end
    lambdas = zeros(iters,1);  
    lambda_psis = zeros(iters,1);
    b = norm(sk)^2 ;
    for iter =1:iters
        psi = Vk*sk' / ( b +lambda/beta) ;
        if b==0
            den_psi = M * b + 1e-8;
        else
            den_psi = M * b ;
        end
        num_psi = b*beta + lambda ;
        a = norm(Vk-psi*sk,'fro')^2 ;
        num_lambda = lambda*(b*beta+lambda)*a;
        den_lambda = M*N*lambda + M*(N-1)*b*beta ;
        beta = sqrt( num_psi/den_psi ) * norm(psi,2) ;
        lambda = sqrt(num_lambda / den_lambda) ;
        lambdas(iter,1) = lambda;
        lambda_psis(iter,1) =  beta;
        
        % stopping criteria 
        if iter>1
            err1 = abs(lambdas(iter,1)-lambdas(iter-1,1)) ;
            err2 = abs(lambda_psis(iter,1)-lambda_psis(iter-1,1)) ;
            if max(err1, err2)<1e-7
                break;
            end
        end
    end
    
    % update k^th atom
    Psi(:,k) = psi/norm(psi);
    lambda_all(1,k) = lambda ;
    Betas(1,k) = beta ;
end

ids = vecnorm(Psi)<1e-2;
Psi(:,ids) =[];
Psi_new = Psi;
E_new = Y - Psi * Z ;
valCost = Betas*diag(Psi_new'*Psi_new) + M*sum(log(Betas)) ...
    + M*N*log(lambda)+norm(Y-Psi_new*Z,'fro')^2/lambda ;

end
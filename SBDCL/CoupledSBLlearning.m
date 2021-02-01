function [Phi,W,DicSize,Psiall,Zall,trainingTime,ConErr,group,valCost] = CoupledSBLlearning(train_samples, train_label, Phiinit, Winit, paras)

iters = paras.iters ;
Y =[train_samples ; train_label];
Psi = normc([Phiinit; Winit]);
DicSize = zeros(1,iters);
numPerClass = sum(train_label,2); % the number of training samples in each category
M = size(train_samples,1);  % the number of features
Psiall = cell(1,iters);
Eall = cell(1,iters);
Zall = cell(1,iters);
ConErr = zeros(iters,1);
valCost = zeros(iters,1);

tic
for iter = 1:iters
     
    fprintf('Begining %d iteration... \n',iter) ;
    
    % update sparse coding matrix Z and delete the redundant atoms
    [ Z, deleted_idx, valCost1] = UpdateWeight(Psi, Y, numPerClass) ;
    
    % if has matlab parallel computing toolbox, one can calculate the sparse
    % codes in parallel to reduce the testing time 
    % [ Z, deleted_idx, valCost1] = UpdateWeightpar(Psi, Y, numPerClass) ;
    
    Psi(:,deleted_idx) = [];
    Z(deleted_idx,:) = [];
    DicSize(iter) = size(Psi,2);
    Zall{iter} = Z;
    
    % update dictionary matrix Psi
    [Psi, E, valCost2] = UpdateDic( Y, Psi, Z) ;
    
    % store the results
    Eall{iter} = E;
    Psiall{iter} = Psi;
    ConErr(iter,1) = norm(E,'fro')/norm(Y,'fro');
    valCost(iter,1) = valCost1 + valCost2 ;

    % stopping criteria: reconstruction error     
    if ConErr(iter,1)<=paras.tau
        break;
    end 
end

% training time
trainingTime = toc;

% dictionary and classifier
Phi = Psi(1:M,:);
W = Psi(M+1:end,:);
normPhi = vecnorm(Phi);
Phi = normc(Phi) ;
W = W./ normPhi;

% divide the dictionary atoms into different groups 
C = size(W,1);
group=cell(C,1);
for c=1:C
    coff = sum( logical(Z(:,train_label(c,:)==1)), 2) ;
    group{c} = find(coff);
end
A = tabulate(cell2mat(group));
idsSharedAtoms = find(A(:,2)>1);
Nullidx=[];
for c=1:C+length(idsSharedAtoms)
    if c<C+1
        temp = setdiff(group{c},idsSharedAtoms);
        group{c} = temp;
        if isempty(temp)
            Nullidx = [Nullidx c];
        end
    else
        group{c} =  idsSharedAtoms(c-C);
    end
end
group(Nullidx)=[];
end



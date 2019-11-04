function [Phiinit, Winit] = initialization(train_samples, train_label, Para)
[M, N] = size(train_samples) ;
    iterations = 20 ;
    numClass = size(train_label,1) ;
    numPerClass = round(Para.initDictSize/numClass);
    num0 = N/numClass ;
    Psiinit = [];
if Para.initDictSize>N
    for k=1:numClass
        col_ids = find(train_label(k,:)==1) ;
        data_ids = vecnorm(train_samples(:,col_ids))>1e-6 ;
        col_ids = col_ids(data_ids);
        Phipart0 =[train_samples(:,col_ids) train_samples(:,col_ids(1:numPerClass-num0))] ;
        Phipart1 =[train_label(:,col_ids) train_label(:,col_ids(1:numPerClass-num0))] ;
        Phipart = [Phipart0 ; Phipart1 ];
        Psiinit = [Psiinit Phipart];
    end
else
    for k=1:numClass
        col_ids = find(train_label(k,:)==1) ;
        data_ids = (vecnorm(train_samples(:,col_ids))>1e-6) ;
        col_ids = col_ids(data_ids);
        Phipart0 = train_samples(:,col_ids(1:numPerClass)) ;
        Phipart1 = train_label(:,col_ids(1:numPerClass)) ;
        Phipart = [Phipart0 ; Phipart1 ];
        data1 = train_samples(:,col_ids) ;
        data2 = train_label(:,col_ids) ;
        para.data = [data1; data2];
        para.Tdata = Para.sparsity;
        para.iternum = iterations;
        para.memusage = 'high';
        % normalization
        para.initdict = Phipart./vecnorm(Phipart) ;
        [Phipart,~,~] = ksvd(para,'');
        Psiinit = [Psiinit Phipart];       
    end    
end

% normalization
Psiinit = normc(Psiinit) ;
Phiinit = Psiinit(1:M,:) ;
normPhi = vecnorm(Phiinit) ;
Phiinit = Phiinit./normPhi;
Winit = Psiinit(M+1:end,:)./normPhi;
end
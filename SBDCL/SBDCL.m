% =========================================================================
%  The SBDCL algorithm proposed in
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

function [accuracy, err, trainingTime, testTime, results] = SBDCL(DataBase, paras)
% The proposed SBDCL method 

%normalize the columns of data 
train_samples = normc(DataBase.training_samples);  
test_samples = normc(DataBase.test_samples);          
train_label = DataBase.training_label;
test_label = DataBase.test_label;


% Initialization using K-SVD
[Phiinit, Winit] = initialization(train_samples, train_label, paras);


%  joint Dictionary and Classifier Learning
[Phi, W, DicSize, Psiiters, Ziters, trainingTime, ConErr, group, valCost] = CoupledSBLlearning(train_samples, train_label, Phiinit, Winit, paras) ;


% OMP based classification
[~, accuracy(1), err2, testTime(1)] = classification2(Phi, W, test_samples, test_label, paras.sparsity) ;


%group Sparse Bayesian Learning based classification
paras.group = group ;
[predictors, accuracy(2), err,testTime(2),Ztest] = classificationGSBL(Phi, W, test_samples, test_label, paras) ;

% if has matlab parallel computing toolbox, one can calculate the sparse
% codes in parallel to reduce the testing time 
% [predictors, accuracy(2), err,testTime(2),Ztest] = classificationGSBLpar(Phi, W, test_samples, test_label, paras) ;


fprintf('\nFinal recognition rate for SBDCL+OMP is : %.02f \n', accuracy(1)*100);
fprintf('\nFinal recognition rate for SBDCL is : %.02f \n', accuracy(2)*100);


% Store the results 
Psiiters = Psiiters(1:10:end);
Ziters = Ziters(1:10:end);
results.DicSize = DicSize ;
results.ConErr = ConErr;
results.Diters = Psiiters ;
results.Siters = Ziters ;
results.predictors = predictors;
results.group = group ;
results.costFunVals = valCost ;
results.Stest = Ztest;
end
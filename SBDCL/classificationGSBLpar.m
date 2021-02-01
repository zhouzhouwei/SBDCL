% =========================================================================
%  The GSBL classification scheme (parallel version) proposed in
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

function [prediction, accuracy, err,testTime,Ztest] = classificationGSBLpar(Phi, W, test_data, labels, paras)

paras.normlized = 0 ;

tic

% solve the sparse code using GSBL 
Ztest = SBLgrouppar(Phi,test_data, paras);

% classification process
score_est = W*Ztest;
[~,ids] = max(score_est);
[ids_true,~] = find(labels);
test_num = size(test_data,2);
true_num = sum(ids_true'==ids);
err_num = test_num - true_num ;
err_ids = find(~(ids_true==ids'));
err_true = ids_true(err_ids);
err_est = ids(err_ids)';
err = [(1:err_num)' err_ids  err_true err_est];
prediction = ids ;
accuracy = true_num/test_num ;
testTime = toc ; 
 
end
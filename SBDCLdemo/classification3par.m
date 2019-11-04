function [prediction, accuracy, err,testTime,Stest] = classification3par(Phi, W, test_data, labels, paras)

paras.normlized = 0 ;

tic

% solve the sparse code using group SBL 
Stest = SBLgrouppar(Phi,test_data, paras);

% classify process
score_est = W*Stest;
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
function [accuracy, err, trainingTime, testTime, results] = SBDCL(DataBase, paras)
% The proposed SBDCL method 

%normalize if not already done
training_samples = normc(DataBase.training_samples);  
test_samples = normc(DataBase.test_samples);          
training_label = DataBase.training_label;
test_label = DataBase.test_label;


% Initializations using K-svd
[Phiinit, Winit] = initialization(training_samples, training_label, paras);


%  joint learning Dictionary and classifier
[Phi, W, DicSize, Psiiters, Ziters, trainingTime, ConErr, group, valCost] = CoupledSBLlearning(training_samples, training_label, Phiinit, Winit, paras) ;


% OMP based classification
[~, accuracy(1), err2, testTime(1)] = classification2(Phi, W, test_samples, test_label, paras.sparsity) ;
%group Sparse Bayesian Learning based classification
paras.group = group ;
[predictors, accuracy(2), err,testTime(2),Stest] = classification3par(Phi, W, test_samples, test_label, paras) ;


fprintf('\nFinal recognition rate for OMP based SBDCL is : %.03f \n', accuracy(1));
fprintf('\nFinal recognition rate for SBDCL is : %.03f \n', accuracy(2));


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
results.Stest = Stest;
end
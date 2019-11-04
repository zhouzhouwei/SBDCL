clear ;  clc ;
addpath('..\ksvdbox')
addpath('..\ompbox')

% load data
exnums = 1;
filename = ['..\Dataset\ExtendedYaleBdata\DataBase',num2str(exnums),'.mat'];
DB = load(filename) ;
accuracy = cell(exnums,1);
err = cell(exnums,1);
trainTime = cell(exnums,1);
testTime = cell(exnums,1);
Results = cell(exnums,1);


% parameters for our approach 
paras1.initDictSize = 988;  % dictionary size 
paras1.sparsity = 80;  % Required for K-SVD initialization 
paras1.iters = 30 ;    % the iterations     
paras1.tau1 = 1e-2;    % the threshold for pruning the small entries
paras1.tau2 = 1e-2;    % the threshold of stopping criteria  


% our method 
[accuracy{1},err{1},trainTime{1},testTime{1},Results{1}] = SBDCL(DB, paras1) ;
acc = cell2mat(accuracy);

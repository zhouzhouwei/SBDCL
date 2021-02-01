% =========================================================================
% An example code for the SBDCL algorithm proposed in
%
%   Wei Zhou, Yue Wu, Junlin Li, Maolin Wang, Hai-Tao Zhang,
%   "A Bayesian Approach for Joint Discriminative Dictionary and Classifier
%    Learning", 
%  submitted to IEEE Transactions on Systems, Man, and Cybernetics: Systems.
%
%    Copyright (c) 2020 Wei Zhou
%    All rights reserved.
%
% Code by Wei Zhou (946958036@qq.com)
% Date: 04-01-2020
% =========================================================================

clear ;  clc ;
addpath('..\ksvdbox')
addpath('..\ompbox')
addpath('..\SBDCL')

%% load data
exnums = 1;
filename = ['..\Dataset\ExtendedYaleBdata\DataBase',num2str(exnums),'.mat'];
DB = load(filename) ;
accuracy = cell(exnums,1);
err = cell(exnums,1);
trainTime = cell(exnums,1);
testTime = cell(exnums,1);
Results = cell(exnums,1);

% the parameters of our approach 
paras1.initDictSize = 988;   % dictionary size 
paras1.sparsity = 80;        % the sparsity constraint factor required for K-SVD initialization 
paras1.iters = 30 ;          % the iterations, which is usually set as 10     
paras1.tau = 1e-2;          % the threshold of stopping criteria  

%% SBDCL algorithm 

[accuracy{1},err{1},trainTime{1},testTime{1},Results{1}] = SBDCL(DB, paras1) ;
acc = cell2mat(accuracy);



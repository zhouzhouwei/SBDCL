# SBDCL
The Sparse Bayesian based joint Discriminative Dictionary and Classifier Learning algorithm

This code is for paper titled 'A Bayesian approach for joint discriminative dictionary and classifier learning'. 

This code is implemented in Matlab 2019a.

In 2017a and before, there is no function "vecnorm", you should add function vecnorm to calculate the norm of the columns of a matrix.

In our algorithm, K-svd is employed to initilize the dictionary matrix, which is obtained from http://www.cs.technion.ac.il/~ronrubin/software.html. 

The Extended YaleB data is from http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html. 

The implement steps: 

Step 1: install the OMPbox  

Step 2: install the ksvdbox following the readme.txt 

Step 3: go to demo file, run the main function 

if you have any questions, please contact zhouwei@hust.edu.cn 

if you use any part of our code, please cite our paper:

  Wei Zhou, Yue Wu, Junlin Li, Maolin Wang, Hai-Tao Zhang,
 "A Bayesian Approach for Joint Discriminative Dictionary and Classifier  Learning", 
 IEEE Transactions on Systems, Man, and Cybernetics: Systems, doi: 10.1109/TSMC.2022.3170443.

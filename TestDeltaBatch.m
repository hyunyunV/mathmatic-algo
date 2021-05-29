clear all; clc;

X = [ 0 0 1; 0 1 1 ; 1 0 1 ; 1 1 1; ];

D = [ 0; 0; 1; 1;];

W = 2*rand(1,3) ;  % init cond
for epoch = 1:300000 % trian for Batch
   W = DeltaBatch(W, X, D) ;
end 

N = 4 ; 
for k = 1:N
    x = X(k,:)';
    v = W*x;
    y(k) = Sigmoid(v) % y(k) = Sigmoid(x)    
end




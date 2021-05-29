clear all; clc;

X = [ 0 0 1; 0 1 1 ; 1 0 1 ; 1 1 1; ];
D = [ 0; 0; 1; 1;];
Ntest=1000;
E1 = zeros(Ntest,1);
E2 = zeros(Ntest,1);

W1 = 2*rand(1,3) - 1; % same init cond for compare
W2 = W1;

for epoch = 1:Ntest
    W1 = DeltaSGD(W1,X,D);
    W2 = DeltaBatch(W2,X,D);
    
    es1 = 0 ;
    es2 = 0 ;
    N = 4;
    for k = 1:N
        x = X(k,:)';
        d = D(k);

        % deltaSGD
        v1 = W1*x;
        y1 = Sigmoid(v1);
        es1 = es1 +(d-y1)^2; % 편차넣을려고 제곱
        
        % deltaBatch
        v2 = W2*x;
        y2 = Sigmoid(v2);
        es2 = es2 + (d-y2)^2;
    end
    
    E1(epoch) = es1/N;
    E2(epoch) = es2/N;

end

plot(E1,'r')
hold on
plot(E2,'b:')
xlabel('Epoch')
ylabel('Average of Training error')
legend('SGD','Batch')
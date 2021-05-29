function [W1, W2] = BackpropMmt(W1, W2, X, D)
    alpha = 0.9;
    beta = 0.9;


    mmt1 = zeros(size(W1));
    mmt2 = zeros(size(W2));



    N = 4;
    for k = 1:N
        x = X(k,:)';
        d = D(k);
        v1 = W1*x ;
        y1 = Sigmoid(v1);
        v = W2*y1;
        y = Sigmoid(v);

        e = d -y;
        delta = y.*(1-y).*e;

        e1 = W2'*delta;
        delta1 = y1.*(1-y1).*e1;

        dW1 = alpha*delta1*x';
        mmt1 = dW1 + beta*mmt1; % CNN고칠때 여기서 뒤에 beta mmt1 만 없으면댐 그럼 돌아감  이게 2번째 프로젝트 
        W1 = W1 + mmt1 ;  % CNN 할때 모멘텀 없애라 ! 모멘텀이랑 SGD만 쓴거 중 어느것이 더 나은가를 해라 이게 두번째 과제 

        dW2 = alpha*delta*y1';
        mmt2 = dW2 + beta*mmt2;
        W2 = W2 + mmt2;
        
    end
end
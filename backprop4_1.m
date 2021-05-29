function [W1 W2] = BackpropXOR(W1, W2, X ,D)
    %https://deep-eye.tistory.com/16 여기참조 근데 노테이션이 다름(알파 델타가 다름) 
    alpha = 0.9 
    N = 4
    for k = 1:N
        x = X(k,:)' % 입력값을 가짐
        d = D(k)  % 에러구하기 위해 실제 값

        v1 = W1 * x
        y1 = Sigmoid(v1)
        v = W2 * y1
        y = Sigmoid(v) 

        e = d - y ;
        delta = y1.*(1-y1).*e; % 매트릭스 곱

        dW1 = alpha.*delta1.*x' ;
        W1 = W1 + dW1;

        dW2 = alpha*delta1.*x' ;
        W1 = W1 + dW2;
        
    end 
end 
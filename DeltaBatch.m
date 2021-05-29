function W = DeltaBatch(W, X, D) % 매트랩은 여기서 리턴값을 내려주는구나, W 로써,
    alpha = 0.9 ; % Learning rate 
    
    dWsum = zeros(3,1);
    
    N = length(X(:,1)); % 4 데이터 셋이 몇개냐 하고 묻는 것 
    for k = 1:N
        x = X(k, :)'; % input
        d = D(k) ; % answer

        v = W*x ;
        y = Sigmoid(v) ;
        % 활성화 함수에는 시그모이드 항등함수 등등
        
        e = d - y ;% error 얘는 계속 측정해야하니깐 ; 을 뺸 것으로 판단됨.  
        delta = y*(1-y)*e ; %델타 값
        
        dW = alpha*delta*x ; % delta rule

        dWsum = dWsum + dW;        
    end
    dWavg = dWsum/N ;
    W(1) = W(1) + dWavg(1);
    W(2) = W(2) + dWavg(2);
    W(3) = W(3) + dWavg(3);
end
function W = DeepReLUfor(W, X, D)
    alpha = 0.01;
    
    % 파이썬 list 없어서 mat cell 쓰긴하는데 너무 불편함 그래도 행렬연산이 자동으로 풀림 아무리 많아도 
    leng = length(W) ;
    Y = cell(1,leng) ;
    V = cell(1,leng) ;
    De = cell(1,leng) ;
    E = cell(1,leng) ;
    N = 5;
    for k = 1:N
        x = reshape(X(:,:,k), 25, 1) ;
        
        % Y,V 계산
        for i = 1:leng
            Y(i) = {x} ; % 이 자리는 x y1 y2 y3 뒤에 쓸거 담아두는 곳  뒤에씀
            V(i) = { cell2mat(W(i))*x} ; % 여기에는 V 를 담아야함 뒤에씀
            x = ReLU( cell2mat( V(i) ) ) ; 
        end
        
        % y-hat, y-target
        y = Softmax( cell2mat( V(leng) ) ) ;
        d = D(k,:)';
        
        % error, delta 계산, 근데 첫번째꺼는 for문 연산안에 넣기가 어려움 안똑같은 것 같은데
        E(leng) = {d - y} ;
        De(leng) = { cell2mat(E(leng)) } ;
        for i = leng-1:-1:1
            E(i) = { cell2mat(W(i+1))'*cell2mat( De(i+1) ) } ;
            De(i) =  { (cell2mat( V(i) ) > 0).*cell2mat( E(i) ) } ;
        end
  
        % W 업데이트
        for i = leng:-1:1
            dW = alpha*cell2mat( De(i) )*cell2mat( Y(i) )';
            W(i) = {cell2mat( W(i) ) + dW} ;            
        end
    end   
end
function [W_CNN,W_D]  = MnistConvMMTFOR(W_CNN, W_D, X, D, alpha, beta )
    % Momentum 하이퍼파라미터들
    
    
    % 자동화에 사용할 cell들 설정 여기서 틀리면 암걸릴 듯 조심조심
    W = [W_CNN, W_D];
    M = cell(1,length(W));
    lenC = length(W_CNN)*3 ; % 이거는 총장
    lenD = length(W_D) ;
    dW_C = cell(1,length(W_CNN)) ;
    dW_D = cell(1,lenD) ;
    Des = cell(1,length(W_CNN)*2 + lenD);
    E = cell(1,length(W_CNN)*2 + lenD+1 );
    V = cell(1,lenD);
    Y = cell(1,lenC + lenD+1);

    % 모멘텀 뚫어 놓기
    for i = 1:length(W)
       M(i) = {zeros(size(cell2mat(W(i))))};
    end
    
    %batch 정의하기
    N = length(D);   
    bsize = 100; % batch size
    blist = 1:bsize:(N-bsize+1); % 80개 나옴 배치해서 돌릴게 80개 있다 100x80 8000 8000개데잉터 쓴다 이거지 linspace 같은 것인 듯 


    for batch = 1:length(blist) % 80 번 하겠다 
       for i = 1:length(W_CNN)
           dW_C(i) = {zeros(size(cell2mat(W_CNN(i))))} ;
       end
       for i = 1:lenD
           dW_D(i) = {zeros(size(cell2mat(W_D(i))))} ;
       end
       
       % Mini-Batch loop
       begin  = blist(batch);
       for k = begin:begin+bsize-1 %  batch 마다 100개씩 있으니깐 여튼 그 갯수만큼 하겠다
           % forward pass = inference?
           
           x = X(:,:,k); % input 28x 28
           Y(lenC) = {x} ;
           for i = 0:lenC/3-1
               j = i*3 ;
               Y(1+j) = {Conv(cell2mat(Y(3+j)),cell2mat(W(i+1)))} ;
               Y(2+j) = {ReLU(cell2mat(Y(1+j)))} ;
               Y(3+j) = {Pool(cell2mat(Y(2+j)))} ;
           end
           Y(lenC+1) = {reshape( cell2mat(Y(lenC)), [], 1)}; % 이전까지가.. 그 ConV ReLU Pool 친구들 여기서 좀 익숙한(?) 걸로 바뀜 뒤에껀 자동화 가능일 듯 
           for i = lenC+1:lenC+lenD
                j = i - lenC ;
                V(j) = {cell2mat(W_D(j))*cell2mat(Y(i))};
                Y(i+1) = {ReLU(cell2mat(V(j)))} ;
           end
 
           y = Softmax(cell2mat(V(lenD))) ;
          
           % One-hot encoding
           d = zeros(10,1);
           d(sub2ind(size(d),D(k),1)) = 1;

           % BroadCasting
           E(length(E)) = {d - y} ;
           Des(length(Des)) = {d - y} ;
           for i = length(E)-1:-1:length(E)-length(W_CNN)*2
               E(i) = {cell2mat(W_D(i - length(W_CNN)*2))'* cell2mat(Des(i))} ;
               Des(i-1) = {(cell2mat(Y(i+1))>0).*cell2mat(E(i))} ;  
           end
           for i = length(W_CNN):-1:1
               % E 참조 인덱스
               p = length(W_CNN)+1 - i;
               j = 2 + (i-1)*3 ; % y 참조 인덱스
               E(length(E)-lenD-2*p+1) = {  reshape( cell2mat(E(length(E)-lenD - (p-1)*2 )), size(cell2mat(Y(j+1))))  };
               nowY = cell2mat(Y(j)) ;
               E(length(E)-lenD-2*p) = {zeros(size(nowY))} ;
               TempW = ones(size(nowY)) / (2*2) ; % 2*2는 pool 이 2x2 평균인가? 그렇기 때문 max는 어캐하노...
               Tempsize = size(nowY) ;
               
               eR = cell2mat(E(length(E)-lenD-2*p +1 ));
               eL = cell2mat(E(length(E)-lenD-2*p ));
               for c = 1:Tempsize(3)
                   eL(:,:,c) = kron(eR(:,:,c), ones([2,2])) .* TempW(:,:,c);    
               end
               
               Tempdelta = ( nowY>0 ).*eL ; 
               savedelta = zeros(size(cell2mat(W_CNN(i)))) ;
               nowY = cell2mat(Y(j-1)) ;
               Tempsize = size(nowY) ;
               for c = 1:Tempsize(3)
                   savedelta(:,:,c) = conv2(x(:,:), rot90(Tempdelta(:,:,c),2), 'valid');
               end
               Des(i) = {savedelta};
           end

           for i = 1:length(dW_C)
               dW_C(i) = {cell2mat(dW_C(i)) + cell2mat(Des(i))};
           end
           for i = 1:lenD
               dW_D(i) = {cell2mat(dW_D(i)) + cell2mat(Des(length(W_CNN)*2+i)) * cell2mat(Y(lenC+i))'};
           end
       end
        % update weight
        
        for i = 1:length(dW_C)
            dW_C(i) = {cell2mat(dW_C(i)) / bsize };
        end
        for i = 1:length(dW_D)
            dW_D(i) = {cell2mat(dW_D(i)) / bsize };
        end
        
        for i = 1:length(dW_C)
            M(i) = {alpha * cell2mat(dW_C(i)) + beta * cell2mat(M(i))} ;
            W_CNN(i) = {cell2mat(W_CNN(i)) + cell2mat(M(i))} ;
        end
        
        for i = 1:length(dW_D)
            M(i+length(W_CNN)) = {alpha * cell2mat(dW_D(i)) + beta * cell2mat(M(i+length(W_CNN)))} ;
            W_D(i) = {cell2mat(W_D(i)) + cell2mat(M(i+length(W_CNN)))} ;
        end
        
    end

end
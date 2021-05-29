function [W1,W5, Wo] = MnistConv(W1,W5,Wo,X,D)
%
%
alpha = 0.01;
beta = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D); % 8,000
bsize = 100; % batch size
blist = 1:bsize:(N-bsize+1); % 80개 나옴 배치해서 돌릴게 80개 있다 100x80 8000 8000개데잉터 쓴다 이거지 linspace 같은 것인 듯 

% one epoch loop
%

for batch = 1:length(blist) % 80 번 하겠다 
   dW1 = zeros(size(W1));
   dW5 = zeros(size(W5));
   dWo = zeros(size(Wo));

   % Mini-Batch loop
   %
   begin  = blist(batch);
   for k = begin:begin+bsize-1 %  batch 마다 100개씩 있으니깐 여튼 그 갯수만큼 하겠다
       % forward pass = inference
       %
       
       x = X(:,:,k); % input 28x 28
       y1 = Conv(x,W1); % conv 20x20x20; pcolor(y1(:,:,1)) Conv함수이해하기
       y2 = ReLU(y1);
       y3 = Pool(y2); % pooling, 10x10x20
       y4 = reshape(y3,[],1); % 이전까지가.. 그 ConV ReLU Pool 친구들 여기서 좀 익숙한(?) 걸로 바뀜 뒤에껀 자동화 가능일 듯 
       v5 = W5*y4; 
       y5 = ReLU(v5);
       v = Wo*y5;
       y = Softmax(v);
       
       % One-hot encoding
       
       d = zeros(10,1);
       d(sub2ind(size(d),D(k),1)) = 1; % 답인 부분에만 인덱스 찍는 것 
       
       % BroadCasting
       
       e = d - y;  % Output layer
       delta = e;
       
       e5 = Wo'* delta ;
       delta5 =(y5>0).*e5; % hidden(ReLU) later
       
       e4 = W5'*delta5; % Pooling layer  여기서 에러남 근데 inf가 있어도 대는건가
       
       e3 = reshape(e4, size(y3));
       
       e2 = zeros(size(y2));
       W3 = ones(size(y2))/ (2*2) ; % 이쪽 위아래 부분이 이해가 되면 콘볼루션 풀링 여러번 하는 것을 이해할 수 있음 컨볼루션 여러번하는거를 쉽게할 수 있음 
       % 얘는 노테이션이 W3가 되면 안댄다 햇갈린다 물론 3번째에서 한다는 것은 이해가능
       for c = 1:20
           e2(:,:,c) = kron(e3(:,:,c), ones([2,2])) .* W3(:,:,c);
       end
       
       delta2 = (y2>0).*e2; % ReLU layer
       
       delta1_x = zeros(size(W1)); % Convolution layer  % 이게 이렇게하는게 얘들이 역전파를 3차원에서 바뀌니깐 이 사단이 남 
       for c = 1:20 % 3차원의 차원인 듯
           delta1_x(:,:,c) = conv2(x(:,:), rot90(delta2(:,:,c),2), 'valid');
       end
       
       dW1 = dW1 + delta1_x ;
       dW5 = dW5 + delta5 * y4' ;
       dWo = dWo + delta * y5' ; % 이까지 하면 처음 100개 배치에 대해서 훈련이 끝난거임 
   end
    % update weight
    
    dW1 = dW1 / bsize ;
    dW5 = dW5 / bsize ;
    dWo = dWo / bsize ;
    % 100개의 평균 부분임
    
    % 아래는 모멘텀 부분 그래디언트가 아니라 모멘터마
    momentum1 = alpha * dW1 + beta*momentum1 ;
    W1 = W1 + momentum1;
    
    momentum5 = alpha * dW5 + beta*momentum5 ;
    W5 = W5 + momentum5;
    
    momentumo = alpha * dWo + beta*momentumo ;
    Wo = Wo + momentumo;
    

end
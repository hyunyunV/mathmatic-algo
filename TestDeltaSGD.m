clear all;
clc; 

X= [ 0 0 1;
     0 1 1;
     1 0 1;
     1 1 1;
    ];

D = [ 0
      0
      1
      1
     ];
W = 2 * rand(1,3) - 1  % rand 유니폼하게 디스트리 뷰트된 랜덤넘버
disp('가중치 초기화') 
% doc rand 이런식으로 하면 구글연결시켜줘서 다큐먼트 떤져줌
% 2*[0 1 ] = [0 2] ->
tic % 시간 잼 여기서 시작 toc까지 센다
for epoch = 1:10000
    W = DeltaSGD(W,X,D);
end
disp('upgraded weight')
W 
toc % 이까지 시간 잼 시간을 알아서 말해줌
% 여기까지가 훈련하는 거임 

N = 4 ;
for k = 1:N
    x = X(k,:)'; % 여기서 트랜스포즈해줌
    v = W*x;
    y(k) = Sigmoid(v) ;
end

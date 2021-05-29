clear all;
% 컴 사양 너무 딸릴 것 같으니깐 pack해야함 --> 메모리 다 비우기 
Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte_');
Images = reshape(Images, 28,28,[]);
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte_');
Labels(Labels == 0) = 10 ; % 0--> 10


% Learning
%
%W1 = 1e-2*randn([9,9,20]); % 이게 2000이랑 연관이 있다고 함 
%W5 = (2*rand(100,2000) -1 ) * sqrt(6) / sqrt(360 +2000); % 2000개 나오면 100개에 보낼거다
%Wo = (2*rand(10,100) -1) * sqrt(6) / sqrt(10+100) ;

W1 = 1e-1*randn([9,9,20]); % 이게 2000이랑 연관이 있다고 함 장기적으로 근데 이게 보니깐 
W5 = (2*rand(50,2000) -1 ) * sqrt(6) / sqrt(360 +2000); % 2000개 나오면 100개에 보낼거다
Wo = (2*rand(10,50) -1) * sqrt(6) / sqrt(10+100) ;
X = Images(:,:,1:8000);
D = Labels(1:8000);

for epoch = 1:1 %8000 x 3 = 24000개 썻다 생각하면 댐
   epoch
   [W1, W5, Wo] = MnistConv(W1,W5,Wo,X,D) ;
end

%save('MnistConv.mat');
% 여기서 기본적 트레이닝 끝 위에거는 데이터 저장하는 듯 이거하면 데이터가 save가 댐
save('MyData.mat')
% Test 
%
X = Images(:,:,8001:10000);
D = Labels(8001:10000);


acc = 0 ;
N = length(D);
for k = 1:N
    x = X(:,:,k);
    
    y1 = Conv(x,W1);
    y2 = ReLU(y1);
    y3 = Pool(y2);
    y4 = reshape(y3,[],1);
    v5 = W5*y4;
    y5 = ReLU(v5);
    v = Wo*y5;
    y = Softmax(v);
    
    [~,i]  = max(y);
    if i == D(k)
        acc = acc +1;
    end
end

acc = acc / k ;
fprintf('Accuracy is %f\n', acc) ;

clear all;
Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte_');
Images = reshape(Images, 28,28,[]);
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte_');
Labels(Labels == 0) = 10 ; % 0--> 10
%rng(1); 

% fold 설정
n = length(Labels);
fold = 10  ;
c = cvpartition(Labels,'KFold',fold,'Stratify',true);

% 결과 담는 것
ACCS_MMT = zeros(1,fold) ;
ACCS_SGD = zeros(1,fold) ; 

for i = 1 : 10
    % 변수 설정하기 ** CNN에 쓸꺼랑 그냥 ANN에 쓸꺼 고려해서 반영 
    W1 = 1e-2*randn([9,9,20]); % 이게 2000이랑 연관이 있다고 함 
    W5 = (2*rand(100,2000) -1 ) * sqrt(6) / sqrt(360 +2000); % 나누고 곱하기 왜하는 거더라 ???
    Wo = (2*rand(10,100) -1) * sqrt(6) / sqrt(10+100) ;
    W_CNN = {W1} ; % W_CNN Weights
    W_D = {W5,Wo} ; % W_Default(?) Weights name..shit
    
    X = Images(:,:,c.training(i));
    D = Labels(c.training(i));
    alpha = 0.01 ; 
    beta = 0.95 ;
    for epoch = 1:i %8000 x 3 = 24000개 썻다 생각하면 댐
       epoch
       [W_CNN, W_D] = MnistConvMMTFOR(W_CNN, W_D, X, D, alpha, beta ) ;
    end

    % Test 
    
    X = Images(:,:,c.test(i));
    D = Labels(c.test(i));


    acc = 0 ;
    N = length(D);
    
 
    for k = 1:N
        x = X(:,:,k);
        y_C = x ; 
        for n = 1:length(W_CNN)
            y_C = Conv(y_C,cell2mat(W_CNN(n)));
            y_C = ReLU(y_C);
            y_C = Pool(y_C);
        end
        y_A = reshape(y_C, [],1) ;
        for n = 1:length(W_D)
            v = W_D(n)*y_A ;
            y_A = ReLU(v);
        end
        y = Softmax(v);
        
        [~,m]  = max(y);
        if m == D(k)
            acc = acc +1;
        end
    end
    acc = acc / k ;
    fprintf('Accuracy is %f\n', acc) ;
    ACCS_MMT(i) = acc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    W1 = 1e-2*randn([9,9,20]); % 이게 2000이랑 연관이 있다고 함 
    W5 = (2*rand(100,2000) -1 ) * sqrt(6) / sqrt(360 +2000); % 나누고 곱하기 왜하는 거더라 ???
    Wo = (2*rand(10,100) -1) * sqrt(6) / sqrt(10+100) ;
    W_CNN = {W1} ; % W_CNN Weights
    W_D = {W5,Wo} ; % W_Default(?) Weights name..shit
    
    X = Images(:,:,c.training(i));
    D = Labels(c.training(i));
    alpha = 0.01 ; 
    beta = 0 ;
    for epoch = 1:i %8000 x 3 = 24000개 썻다 생각하면 댐
       epoch
       [W_CNN, W_D] = MnistConvMMTFOR(W_CNN, W_D, X, D, alpha, beta ) ;
    end

    % Test 
    
    X = Images(:,:,c.test(i));
    D = Labels(c.test(i));


    acc = 0 ;
    N = length(D);
    
 
    for k = 1:N
        x = X(:,:,k);
        y_C = x ; 
        for n = 1:length(W_CNN)
            y_C = Conv(y_C,cell2mat(W_CNN(n)));
            y_C = ReLU(y_C);
            y_C = Pool(y_C);
        end
        y_A = reshape(y_C, [],1) ;
        for n = 1:length(W_D)
            v = W_D(n)*y_A ;
            y_A = ReLU(v);
        end
        y = Softmax(v);
        
        [~,m]  = max(y);
        if m == D(k)
            acc = acc +1;
        end
    end
    acc = acc / k ;
    fprintf('Accuracy is %f\n', acc) ;
    ACCS_SGD(i) = acc;
    
    
end
subplot(2,1,1);
bar(ACCS_MMT);
xlabel('i-번째 결과','Fontsize',13);
ylabel('Accracy','Fontsize',13);
title('Momentum Used', 'Fontsize', 15,'FontWeight', 'bold');

subplot(2,1,2);
bar(ACCS_SGD);
xlabel('i-번째 결과','Fontsize',13);
ylabel('Accracy','Fontsize',13);
title('Only SGD Used', 'Fontsize', 15,'FontWeight', 'bold');


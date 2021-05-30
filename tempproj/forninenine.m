clear all;
Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte_');
Images = reshape(Images, 28,28,[]);
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte_');
Labels(Labels == 0) = 10 ; % 0--> 10

% fold 설정
n = length(Labels);
fold = 10  ;
c = cvpartition(Labels,'KFold',fold,'Stratify',true);

% 기본설정들
epochs = 2 ;

% 결과 담는 것
ACCS_MMT = zeros(1,fold) ;
ACCS_SGD = zeros(1,fold) ; 

for i = 1 : 1
    fprintf('%d - th iteration', i) ;
    % 변수 설정하기 ** CNN에 쓸꺼랑 그냥 ANN에 쓸꺼 고려해서 반영 
    W1 = 1e-2*randn([3,3,1,32]); % 이게 2000이랑 연관이 있다고 함 
    W2 = 1e-2*randn([3,3,32,64]);
    W5 = (2*rand(10,3136) -1 ) * sqrt(6) / sqrt(3136) ;  % weight initialize 일단 ReLU 쓰니깐 Xavier 말고 He initialize써보자  
    W_CNN = {W1, W2} ; % W_CNN Weights
    W_D = {W5} ; % W_Default(?) Weights name..shit
    
    X = Images(:,:,c.training(i));
    D = Labels(c.training(i));
    alpha = 0.001 ; 
    beta = 0.95 ;
    for epoch = 1:epochs %8000 x 3 = 24000개 썻다 생각하면 댐
       [W_CNN, W_D] = CNNver1(W_CNN, W_D, X, D, alpha, beta ) ;
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
            v = cell2mat(W_D(n))*y_A ;
            y_A = ReLU(v);
        end
        y = Softmax(v);
        
        [~,m]  = max(y);
        if m == D(k)
            acc = acc +1;
        end
    end
    acc = acc / k ;
    ACCS_MMT(i) = acc;
    fprintf('%d - ACCS - %f - MMT \n ', i, accs) ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    W1 = 1e-2*randn([3,3,1,32]); % 이게 2000이랑 연관이 있다고 함 
    W2 = 1e-2*randn([3,3,32,64]);
    W5 = (2*rand(10,3136) -1 ) * sqrt(6) / sqrt(3136) ;  % weight initialize 일단 ReLU 쓰니깐 Xavier 말고 He initialize써보자  
    W_CNN = {W1, W2} ; % W_CNN Weights
    W_D = {W5} ; % W_Default(?) Weights name..shit
    
    X = Images(:,:,c.training(i));
    D = Labels(c.training(i));
    alpha = 0.001 ; 
    beta = 0 ;
    for epoch = 1:epochs %8000 x 3 = 24000개 썻다 생각하면 댐
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
            v = cell2mat(W_D(n))*y_A ;
            y_A = ReLU(v);
        end
        y = Softmax(v);
        
        [~,m]  = max(y);
        if m == D(k)
            acc = acc +1;
        end
    end
    acc = acc / k ;
    ACCS_SGD(i) = acc;
    fprintf('%d - ACCS - %f - SGD \n ', i, accs) ;

end
ment = "평균 : ";
subplot(2,1,1);
bar(ACCS_MMT);
MeanAccs = strcat(ment, num2str(mean(ACCS_MMT))) ;
xlabel(MeanAccs,'Fontsize',13);
ylabel('i-번째 결과','Fontsize',13);
title('Momentum Used', 'Fontsize', 15,'FontWeight', 'bold');

subplot(2,1,2);
bar(ACCS_SGD);
MeanAccs = strcat(ment, num2str(mean(ACCS_SGD))) ;
xlabel(MeanAccs,'Fontsize',13);
ylabel('i-번째 결과','Fontsize',13);
title('Only SGD Used', 'Fontsize', 15,'FontWeight', 'bold');
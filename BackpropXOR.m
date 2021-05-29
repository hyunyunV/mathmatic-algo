function [W1 W2] = BackpropXOR(W1, W2, X ,D)
    alpha = 0.9 ; % 훈련 정도를 설정 
    N = 4; % 참조 인덱스 설정
    for k = 1:N
        x = X(k,:)' ;% 입력값 가져옴
        d = D(k) ; % 에러구하기 위해 실제 값 가져옴

        v1 = W1 * x ; % v1 = 입력값에다가 첫번째 레이어에서 두번째레이어로 가는 Weight곱해서 2번째 레이어 z값 벡터생성
        y1 = Sigmoid(v1) ; % Sigmoid 비선형화해서 3번째 레이어로 보낼 a값 벡터 생성 
        v = W2 * y1 ; % 2번째레이어에서 받은 입력값과 2,3번째 사이의 Weight를 곱해서 z 값 생성
        y = Sigmoid(v) ;  % 값을 sigmoid 비선형화해서 아웃풋값 생성

        e = d - y ; % 예측치와 실체치 사이의 비용을 계산
        delta = y.*(1-y).*e; % 2,3번째 사이에서 역전파 훈련시키기 위해서 델타생성 
        %y=Sigmoid(W2*y1=z3)이고 dC/dz를 구하고 에러만큼 곱해줌(틀린정도만큼 훈련) 
        %2,3번째 레이어 사이의 weight들 훈련에 사용
        
        e1 = W2'*delta ; % 1,2번째 사이의 Weight들 훈련시키기위한 2번째 에러 생성 dz/da형식으로 이어 붙임  
        delta1 = y1.*(1-y1).*e1 ; % 마찬가지로 역전파 훈련위한 델타 생성
        % e1 의 역전파들에다가 da/dz 하여 sigmoid ft을 미분하여 연결
        
        
        dW1 = alpha*delta1*x' ; % 업데이트 시킬 dW1생성 학습률 X 역전파 X 활용input(grad값알기위해)
        W1 = W1 + dW1; % W1 업데이트

        dW2 = alpha*delta*y1' ; % 업데이트 시킬 dW2생성 학습률 X 역전파 X 활용input 
        W2 = W2 + dW2; % W2 업데이트
        
    end 
end 
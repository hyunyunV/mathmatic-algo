function y = Padding(x)
% 0으로 패딩시켜서 내보내기 
%
[xrow, xcol, nFiliters         ] = size(x); % 알아서 타타타탁 계산 벡터 컬큘레이션 해준다함

%yrow = xrow - wrow +1 ; % 전체 사이즈 계산
%ycol = xcol - wcol + 1; % 궁금하면 doc conv2 여기 다큐먼트에 나와있음

y = zeros(xrow+2, xcol+2, nFiliters) ;
for k = 1:nFiliters
    y(:,:,k) =  padarray(x(:,:,k),[1 1], 0, 'both') ;% 이거 valid를 빼면 stride가 생기는 것임 
end

end
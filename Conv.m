function y = Conv(x,W)
%
%
[wrow, wcol, numFilters] = size(W);
[xrow, xcol, ~         ] = size(x); % 알아서 타타타탁 계산 벡터 컬큘레이션 해준다함

yrow = xrow - wrow +1 ; % 전체 사이즈 계산
ycol = xcol - wcol + 1; % 궁금하면 doc conv2 여기 다큐먼트에 나와있음

y = zeros(yrow, ycol, numFilters) ;

for k = 1:numFilters
    filter = W(:,:,k);
    filter = rot90(squeeze(filter),2); % 28,28,2인데 스토리즈  줄일때 sqeueeze 를 씀
    y(:,:,k) = conv2(x,filter, 'valid');
end

end
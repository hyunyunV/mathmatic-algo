function e = JumpError(W, eL)
% 일단 conv에 쓰이는 W로 back propa해서 14x14x32만들거 이거를 백프로파 시캬ㅕ서 28x28x만들자 
W = cell2mat(W);
[ ~  , ~   , nFilters, ~] = size(W);
[xrow, xcol, ~         ] = size(eL); % 알아서 타타타탁 계산 벡터 컬큘레이션 해준다함

e = zeros(xrow, xcol, nFilters) ;
for k = 1:nFilters
    filter = W(:,:,k,:);
    filter = rot90(squeeze(filter),2); % 28,28,2인데 스토리즈  줄일때 sqeueeze 를 씀
    e(:,:,k) = sum(convn(Padding(eL),filter, 'valid'),3); % 이거 valid를 빼면 stride가 생기는 것임 
end

e = {e} ;







end
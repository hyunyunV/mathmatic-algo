function y= Pool(x)
%max pooling 

dlx = dlarray(x,'SSC');
[y, ~, ~] = maxpool(dlx,2,'Stride',2);
end
%{
% 2x2 mean pooling
[xrow, xcol, numFilters] = size(x);
y = zeros(xrow/2, xcol/2, numFilters);
for k = 1:numFilters
    
    filter = ones(2) / (2*2) ; % 여기다가 max를 해야지 max pool인데음.. 맥스풀링을 할 수 없으니 지수를 이용해서 맥스풀링을 하도록 하자 
    image = conv2(x(:,:,k), filter, 'valid'); %여기서 음.. 
    % 음.. 일단 max 비슷하게 값을 뽑아서 보고 거기서 가장 큰 것을 인데스 찾아서  2x2안에서 찾아서 넣어주자 그럼 댈
    % 듯??? 일단.. 음 ... 
    
    
    y(:,:,k) = image(1:2:end , 1:2:end); % 1에서 시작해서 2씩 전진해서 끝까지 
    
end

end
%}
%{

miniBatchSize = 128;
inputSize = [28 28];
numChannels = 3;
X = rand(inputSize(1),inputSize(2),numChannels,miniBatchSize);
dlX = dlarray(X,'SSCB');
[dlY,indx,dataSize] = maxpool(dlX,2,'Stride',2);

%}
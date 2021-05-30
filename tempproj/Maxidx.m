function  W = Maxidx(x)
%max pooling 

dlx = dlarray(x,'SSC');
[~, idx, ~] = maxpool(dlx,2,'Stride',2);
W = zeros(size(x));
W(idx) = 1;
end
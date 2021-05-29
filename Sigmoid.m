function y = Sigmoid(x)
   y = 1 ./ (1 + exp(-x)); % 점을찍어줘야지 애가 벡터로 드갈 수 있음
end
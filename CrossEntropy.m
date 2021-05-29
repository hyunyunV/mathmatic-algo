function J = CrossEntropy(D, X) % d x 참값 J는 아우풋
    J = -D * log(X) - ( 1 - D ) * log(1-X) ;
end
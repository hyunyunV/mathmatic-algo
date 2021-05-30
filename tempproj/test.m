A = rand(2,2,2);
B = 0.25*ones(2,2,2);
C = convn(A,B,'same')
